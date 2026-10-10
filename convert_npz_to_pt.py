import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


def discover_npz_files(data_dir: Path) -> List[Path]:
    files = []
    for path in sorted(data_dir.rglob("*.npz")):
        if any(part in {"ricci_flow_pt", "networkx", "ricci_flow_graphs"} for part in path.relative_to(data_dir).parts):
            continue
        with np.load(path, allow_pickle=True) as payload:
            if "edges" in payload or "edge_index" in payload:
                files.append(path)
    return files


def _maybe_scalar_to_str(value: np.ndarray) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def infer_dataset_name(data_dir: Path, npz_path: Path, payload: Dict[str, np.ndarray]) -> str:
    if "dataset_name" in payload:
        return _maybe_scalar_to_str(payload["dataset_name"])

    rel = npz_path.relative_to(data_dir)
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return npz_path.stem


def normalize_edge_index(edge_index: np.ndarray) -> np.ndarray:
    arr = np.asarray(edge_index)
    if arr.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got shape={arr.shape}")

    if arr.shape[0] == 2:
        fixed = arr
    elif arr.shape[1] == 2:
        fixed = arr.T
    else:
        raise ValueError(f"edge_index must be [2, E] or [E, 2], got shape={arr.shape}")

    return fixed.astype(np.int64, copy=False)


def normalize_time_series(name: str, values: np.ndarray, num_edges: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be 1D/2D, got shape={arr.shape}")

    if arr.shape[1] == num_edges:
        fixed = arr
    elif arr.shape[0] == num_edges:
        fixed = arr.T
    else:
        raise ValueError(
            f"{name} edge dimension mismatch, expected {num_edges}, got shape={arr.shape}"
        )

    return fixed.astype(np.float32, copy=False)


def build_edge_list(edge_index: np.ndarray) -> List[Tuple[int, int]]:
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    return list(zip(src, dst))


def validate_optional_snapshot(
    payload: Dict[str, np.ndarray],
    curvatures: np.ndarray,
    weights: np.ndarray,
) -> str:
    warnings: List[str] = []
    tol = 1e-6

    if "final_curvatures" in payload:
        final_curv = np.asarray(payload["final_curvatures"]).reshape(-1).astype(np.float32, copy=False)
        if final_curv.shape[0] != curvatures.shape[1]:
            warnings.append("final_curvatures length mismatch")
        elif not np.allclose(final_curv, curvatures[-1], atol=tol, rtol=tol):
            warnings.append("final_curvatures differs from curvatures[-1]")

    if "final_weights" in payload:
        final_w = np.asarray(payload["final_weights"]).reshape(-1).astype(np.float32, copy=False)
        if final_w.shape[0] != weights.shape[1]:
            warnings.append("final_weights length mismatch")
        elif not np.allclose(final_w, weights[-1], atol=tol, rtol=tol):
            warnings.append("final_weights differs from weights[-1]")

    return "; ".join(warnings)


def build_flow_data(
    dataset_name: str,
    edge_index: np.ndarray,
    curvatures: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, object]:
    return {
        "iterations": int(curvatures.shape[0]) - 1,
        "num_states": int(curvatures.shape[0]),
        "edge_index": torch.as_tensor(edge_index, dtype=torch.long),
        "edge_list": build_edge_list(edge_index),
        "curvatures": torch.as_tensor(curvatures, dtype=torch.float32),
        "weights": torch.as_tensor(weights, dtype=torch.float32),
        "dataset_name": dataset_name,
    }


def output_pt_path(data_dir: Path, npz_path: Path, dataset_name: str) -> Path:
    return data_dir / "ricci_flow_pt" / npz_path.relative_to(data_dir).with_suffix(".pt")


def infer_num_nodes(payload: Dict[str, np.ndarray], edge_index: np.ndarray) -> int:
    if "num_nodes" in payload:
        return int(np.asarray(payload["num_nodes"]).reshape(-1)[0])
    if "node_features" in payload:
        return int(np.asarray(payload["node_features"]).shape[0])
    if edge_index.size == 0:
        return 0
    return int(edge_index.max()) + 1


def _edge_attr(graph, u: int, v: int, key: str, default: float) -> float:
    data = graph.get_edge_data(u, v)
    if data is None:
        data = graph.get_edge_data(v, u)
    if data is None:
        return default
    return float(data.get(key, default))


def compute_ricci_from_edges(
    edge_index: np.ndarray,
    num_nodes: int,
    flow_iterations: int,
    alpha: float,
    method: str,
    step: float = 0.1,
    sinkhorn_reg: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import networkx as nx
    from run_ricci_flow_and_save import run_flow_no_merge
    if method != "Sinkhorn":
        raise ValueError("Only Sinkhorn is supported")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        raise ValueError("Edge endpoint outside the node range")
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    graph.remove_edges_from(nx.selfloop_edges(graph))
    result = run_flow_no_merge(graph, "ollivier", flow_iterations, alpha=alpha, step=step, sinkhorn_reg=sinkhorn_reg)
    return (result["edge_index"].numpy(), result["curvatures"].numpy(), result["weights"].numpy())


def load_or_compute_ricci(
    payload: Dict[str, np.ndarray],
    flow_iterations: int,
    alpha: float,
    method: str,
    step: float = 0.1,
    sinkhorn_reg: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if "edge_index" in payload and "curvatures" in payload and "weights" in payload:
        edge_index = normalize_edge_index(payload["edge_index"])
        num_edges = edge_index.shape[1]
        curvatures = normalize_time_series("curvatures", payload["curvatures"], num_edges)
        weights = normalize_time_series("weights", payload["weights"], num_edges)
        if str(np.asarray(payload.get("flow_version", "")).item()) != "linear-rho-sinkhorn-v1":
            raise ValueError("NPZ histories use an unsupported flow definition")
        if str(np.asarray(payload.get("ot_method", "")).item()) != "Sinkhorn" or str(np.asarray(payload.get("method", "")).item()) != "ollivier":
            raise ValueError("Existing NPZ histories must declare method=ollivier and ot_method=Sinkhorn")
        if curvatures.shape != weights.shape:
            raise ValueError(
                f"curvatures and weights shape mismatch: {curvatures.shape} vs {weights.shape}"
            )
        return edge_index, curvatures, weights, "from_npz_fields"

    if "edges" not in payload:
        raise KeyError("missing required key: edge_index/curvatures/weights or edges")

    edge_index = normalize_edge_index(payload["edges"])
    num_nodes = infer_num_nodes(payload, edge_index)
    edge_index, curvatures, weights = compute_ricci_from_edges(
        edge_index=edge_index,
        num_nodes=num_nodes,
        flow_iterations=flow_iterations,
        alpha=alpha,
        method=method,
        step=step,
        sinkhorn_reg=sinkhorn_reg,
    )
    return edge_index, curvatures, weights, "computed_from_edges"


def convert_one(
    data_dir: Path,
    npz_path: Path,
    flow_iterations: int,
    alpha: float,
    method: str,
    step: float = 0.1,
    sinkhorn_reg: float = 0.1,
) -> Dict[str, str]:
    record = {
        "dataset_name": "",
        "input_npz": str(npz_path),
        "output_pt": "",
        "num_nodes": "",
        "num_edges": "",
        "iterations": "",
        "elapsed_sec": "",
        "source_mode": "",
        "status": "failed",
        "error_msg": "",
        "warning_msg": "",
    }

    start = time.time()
    try:
        with np.load(npz_path, allow_pickle=True) as payload_npz:
            payload = {k: payload_npz[k] for k in payload_npz.files}

        dataset_name = infer_dataset_name(data_dir, npz_path, payload)
        edge_index, curvatures, weights, source_mode = load_or_compute_ricci(
            payload=payload,
            flow_iterations=flow_iterations,
            alpha=alpha,
            method=method,
            step=step,
            sinkhorn_reg=sinkhorn_reg,
        )
        num_edges = edge_index.shape[1]
        num_nodes = infer_num_nodes(payload, edge_index)

        warning_msg = ""
        if source_mode == "from_npz_fields":
            warning_msg = validate_optional_snapshot(payload, curvatures, weights)
        if curvatures.shape[0] == 0 or not np.isfinite(curvatures).all() or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Flow histories must be nonempty and finite, with positive weights")
        flow_data = build_flow_data(dataset_name, edge_index, curvatures, weights)
        flow_data.update(flow_version="linear-rho-sinkhorn-v1", measure="linear_weight", curvature_denominator="shortest_path", method="ollivier", ot_method="Sinkhorn", curvature_attr="ricciCurvature", num_nodes=num_nodes, num_edges=num_edges, is_directed=False)
        if source_mode == "computed_from_edges":
            flow_data.update(alpha=alpha, step=step, sinkhorn_reg=sinkhorn_reg)
        else:
            for key in ("alpha", "step", "sinkhorn_reg"):
                if key in payload:
                    flow_data[key] = float(np.asarray(payload[key]).item())

        out_path = output_pt_path(data_dir, npz_path, dataset_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(flow_data, out_path)

        record["dataset_name"] = dataset_name
        record["output_pt"] = str(out_path)
        record["num_nodes"] = str(num_nodes)
        record["num_edges"] = str(num_edges)
        record["iterations"] = str(flow_data["iterations"])
        record["source_mode"] = source_mode
        record["status"] = "success"
        record["warning_msg"] = warning_msg
    except Exception as exc:  # noqa: BLE001
        record["error_msg"] = str(exc)
    finally:
        record["elapsed_sec"] = f"{time.time() - start:.4f}"

    return record


def write_summary(data_dir: Path, rows: List[Dict[str, str]]) -> Path:
    summary_path = data_dir / "pt_export_summary.csv"
    fieldnames = [
        "dataset_name",
        "input_npz",
        "output_pt",
        "num_nodes",
        "num_edges",
        "iterations",
        "elapsed_sec",
        "source_mode",
        "status",
        "error_msg",
        "warning_msg",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def write_json_report(data_dir: Path, rows: List[Dict[str, str]]) -> Path:
    json_path = data_dir / "pt_export_summary.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(rows),
        "success": sum(r["status"] == "success" for r in rows),
        "failed": sum(r["status"] != "success" for r in rows),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert data/**/*.npz to unified Ricci flow .pt files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Root directory to scan npz files (default: ./data).",
    )
    parser.add_argument(
        "--flow-iterations",
        type=int,
        default=10,
        help="Ricci flow iterations when computing from raw edges (default: 10).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="OllivierRicci alpha parameter (default: 0.5).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="Sinkhorn",
        choices=["Sinkhorn"],
        help="Optimal transport solver (Sinkhorn).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Only process the first N npz files for quick checks (0 means all).",
    )
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--sinkhorn-reg", type=float, default=0.1)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        parser.error("Data directory does not exist")
    npz_files = discover_npz_files(data_dir)
    if args.max_files > 0:
        npz_files = npz_files[: args.max_files]
    if not npz_files:
        print(f"[INFO] No npz files found under: {data_dir}")
        empty_rows: List[Dict[str, str]] = []
        summary = write_summary(data_dir, empty_rows)
        details = write_json_report(data_dir, empty_rows)
        print(f"[INFO] Empty summary written: {summary}")
        print(f"[INFO] Empty details written: {details}")
        return 0

    print(f"[INFO] Found {len(npz_files)} npz files")
    rows: List[Dict[str, str]] = []
    for idx, npz_path in enumerate(npz_files, start=1):
        print(f"[{idx}/{len(npz_files)}] converting {npz_path}")
        row = convert_one(
            data_dir=data_dir,
            npz_path=npz_path,
            flow_iterations=args.flow_iterations,
            alpha=args.alpha,
            method=args.method,
            step=args.step,
            sinkhorn_reg=args.sinkhorn_reg,
        )
        rows.append(row)
        if row["status"] == "success":
            print(
                f"  -> OK dataset={row['dataset_name']} edges={row['num_edges']} "
                f"T={row['iterations']} source={row['source_mode']} output={row['output_pt']}"
            )
            if row["warning_msg"]:
                print(f"  -> WARN {row['warning_msg']}")
        else:
            print(f"  -> FAIL {row['error_msg']}")

    summary = write_summary(data_dir, rows)
    details = write_json_report(data_dir, rows)
    success = sum(r["status"] == "success" for r in rows)
    failed = len(rows) - success
    print(f"[DONE] success={success}, failed={failed}")
    print(f"[DONE] summary={summary}")
    print(f"[DONE] details={details}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
