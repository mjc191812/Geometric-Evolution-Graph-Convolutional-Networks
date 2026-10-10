from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import numpy as np
import ot
from functools import lru_cache

FLOW_VERSION = "linear-rho-sinkhorn-v1"


def discover_graph_files(root: Path) -> List[Path]:
    files = list(root.rglob("*.gpkl")) + list(root.rglob("*.gbkl"))
    return sorted(set(files))


def load_graph(path: Path) -> nx.Graph:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, nx.Graph):
        raise TypeError(f"{path} 不是 networkx 图，实际类型: {type(obj)}")
    return obj


def graph_components_nodes(G: nx.Graph) -> Iterable[set]:
    if G.is_directed():
        return nx.weakly_connected_components(G)
    return nx.connected_components(G)


def edge_key(u: int, v: int, directed: bool) -> Tuple[int, int]:
    if directed:
        return (int(u), int(v))
    u_i, v_i = int(u), int(v)
    return (u_i, v_i) if u_i <= v_i else (v_i, u_i)


def ensure_edge_weight(G: nx.Graph, weight_name: str = "weight") -> None:
    for u, v in G.edges():
        weight = float(G[u][v].get(weight_name, 1.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Edge weights must be finite and positive")
        G[u][v][weight_name] = weight


def neighbor_measure(G, node, alpha=0.5, weight_name="weight"):
    neighbors = list(G.neighbors(node))
    if not neighbors:
        return [node], np.array([1.0])
    weights = np.array([G[node][v][weight_name] for v in neighbors], dtype=float)
    probabilities = (1.0 - alpha) * weights / weights.sum()
    support = neighbors + [node]
    probabilities = np.append(probabilities, alpha)
    positive = probabilities > 0
    return [v for v, keep in zip(support, positive) if keep], probabilities[positive]


def compute_curvature_without_merging(G, method="ollivier", weight_name="weight", alpha=0.5, sinkhorn_reg=0.1):
    if method != "ollivier" or sinkhorn_reg <= 0:
        raise ValueError("Expected Ollivier-Ricci/Sinkhorn with positive regularization")
    shortest_path_sec = 0.0
    for nodes in graph_components_nodes(G):
        component = G.subgraph(nodes)
        measures = {u: neighbor_measure(component, u, alpha, weight_name) for u in nodes}

        @lru_cache(maxsize=32)
        def distances(source):
            nonlocal shortest_path_sec
            started = time.perf_counter()
            result = nx.single_source_dijkstra_path_length(component, source, weight=weight_name)
            shortest_path_sec += time.perf_counter() - started
            return result

        for u, v in component.edges():
            source, p = measures[u]
            target, q = measures[v]
            rho = distances(u)[v]
            costs = np.array([[distances(a)[b] for b in target] for a in source], dtype=float)
            plan = ot.bregman.sinkhorn_log(p, q, costs, sinkhorn_reg, numItermax=10000, stopThr=1e-9, warn=False)
            if (not np.isfinite(plan).all() or
                    not np.allclose(plan.sum(axis=1), p, atol=1e-6, rtol=0) or
                    not np.allclose(plan.sum(axis=0), q, atol=1e-6, rtol=0)):
                raise RuntimeError(f"Sinkhorn did not converge on edge ({u}, {v})")
            transport_cost = float(np.sum(plan * costs))
            curvature = 1.0 - transport_cost / rho
            if not math.isfinite(curvature):
                raise ValueError("Non-finite curvature")
            G[u][v]["ricciCurvature"] = curvature
            G[u][v]["rho"] = rho
        distances.cache_clear()
    return shortest_path_sec


def record_histories(
    G: nx.Graph,
    edge_list: Sequence[Tuple[int, int]],
    edge_index_map: Dict[Tuple[int, int], int],
    curvature_attr: str,
    directed: bool,
    weight_name: str = "weight",
) -> Tuple[List[float], List[float]]:
    curv = [0.0] * len(edge_list)
    wgt = [0.0] * len(edge_list)

    for u, v in edge_list:
        k = edge_key(u, v, directed)
        idx = edge_index_map[k]
        data = G[u][v]
        curv[idx] = float(data.get(curvature_attr, 0.0))
        wgt[idx] = float(data.get(weight_name, 1.0))
    return curv, wgt


def run_flow_no_merge(
    G: nx.Graph,
    method: str,
    iterations: int,
    step: float = 0.1,
    weight_name: str = "weight",
    alpha: float = 0.5,
    sinkhorn_reg: float = 0.1,
) -> Dict[str, object]:
    if method != "ollivier" or iterations < 0 or not 0 < step < 1 or not 0 <= alpha <= 1:
        raise ValueError("Invalid flow parameters")
    if G.is_multigraph() or G.is_directed() or nx.number_of_selfloops(G):
        raise ValueError("Expected a simple undirected graph without self-loops")
    n0, e0 = G.number_of_nodes(), G.number_of_edges()
    directed = G.is_directed()
    ensure_edge_weight(G, weight_name=weight_name)
    shortest_path_sec = 0.0
    flow_sec = 0.0

    edge_list = [(int(u), int(v)) for u, v in G.edges()]
    edge_index_map: Dict[Tuple[int, int], int] = {}
    for i, (u, v) in enumerate(edge_list):
        edge_index_map[edge_key(u, v, directed)] = i

    curvatures_history: List[List[float]] = []
    weights_history: List[List[float]] = []
    rho_history = []
    shortest_path_sec += compute_curvature_without_merging(G, alpha=alpha, weight_name=weight_name, sinkhorn_reg=sinkhorn_reg)
    curvature_attr = "ricciCurvature"
    for state in range(iterations + 1):
        c, w = record_histories(G, edge_list, edge_index_map, curvature_attr, directed, weight_name)
        curvatures_history.append(c)
        weights_history.append(w)
        rho_history.append([G[u][v]["rho"] for u, v in edge_list])
        if state == iterations:
            break
        started = time.perf_counter()
        new_weights = {}
        for u, v in edge_list:
            edge = G[u][v]
            value = edge[weight_name] - step * edge[curvature_attr] * edge["rho"]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Non-positive flow weight on edge ({u}, {v}); decrease step")
            new_weights[(u, v)] = value
        nx.set_edge_attributes(G, new_weights, weight_name)
        shortest_path_sec += compute_curvature_without_merging(G, alpha=alpha, weight_name=weight_name, sinkhorn_reg=sinkhorn_reg)
        flow_sec += time.perf_counter() - started

    n1, e1 = G.number_of_nodes(), G.number_of_edges()
    if n0 != n1 or e0 != e1:
        raise RuntimeError(f"flow 前后规模改变: ({n0},{e0}) -> ({n1},{e1})")

    src = [u for u, _ in edge_list]
    dst = [v for _, v in edge_list]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return {
        "flow_version": FLOW_VERSION,
        "measure": "linear_weight",
        "curvature_denominator": "shortest_path",
        "sinkhorn_reg": sinkhorn_reg,
        "rho": torch.tensor(rho_history, dtype=torch.float32),
        "iterations": iterations,
        "num_states": len(curvatures_history),
        "ot_method": "Sinkhorn",
        "alpha": alpha,
        "step": step,
        "method": method,
        "curvature_attr": curvature_attr,
        "edge_index": edge_index,
        "edge_list": edge_list,
        "curvatures": torch.tensor(curvatures_history, dtype=torch.float32),
        "weights": torch.tensor(weights_history, dtype=torch.float32),
        "num_nodes": n1,
        "num_edges": e1,
        "is_directed": directed,
        "shortest_path_sec": shortest_path_sec,
        "flow_sec": flow_sec,
    }


def infer_dataset_name(rel_path: Path) -> str:
    if len(rel_path.parts) > 1:
        return rel_path.parts[0].lower()
    return rel_path.stem.lower()


def save_outputs(
    rel: Path,
    out_graph_dir: Path,
    out_pt_dir: Path,
    graph: nx.Graph,
    flow_data: Dict[str, object],
    dataset_name: str,
) -> Tuple[Path, Path]:
    graph_path = out_graph_dir / rel
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with graph_path.open("wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    pt_path = out_pt_dir / rel.with_suffix(".pt")
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(flow_data)
    payload["dataset_name"] = dataset_name
    torch.save(payload, pt_path)
    return graph_path, pt_path


def output_paths_for(rel: Path, out_graph_dir: Path, out_pt_dir: Path) -> Tuple[Path, Path]:
    return out_graph_dir / rel, out_pt_dir / rel.with_suffix(".pt")


def run_one(
    graph_file: Path,
    data_root: Path,
    out_graph_dir: Path,
    out_pt_dir: Path,
    force_iterations: int | None,
    step: float,
    overwrite: bool,
    alpha: float = 0.5,
    sinkhorn_reg: float = 0.1,
) -> Dict[str, str]:
    rel = graph_file.relative_to(data_root)
    dataset_name = infer_dataset_name(rel)
    method = "ollivier"
    iterations = force_iterations if force_iterations is not None else 10

    rec = {
        "dataset": dataset_name,
        "input_graph": str(graph_file),
        "method": method,
        "iterations": str(iterations),
        "status": "failed",
        "elapsed_sec": "",
        "total_sec": "",
        "shortest_path_sec": "",
        "flow_sec": "",
        "num_nodes": "",
        "num_edges": "",
        "output_graph": "",
        "output_pt": "",
        "note": "",
        "error": "",
    }

    t0 = time.time()
    try:
        out_graph_path, out_pt_path = output_paths_for(rel, out_graph_dir, out_pt_dir)
        if (not overwrite) and out_graph_path.exists() and out_pt_path.exists():
            cached = torch.load(out_pt_path, map_location="cpu", weights_only=False)
            expected = {"flow_version": FLOW_VERSION, "sinkhorn_reg": sinkhorn_reg, "method": "ollivier", "ot_method": "Sinkhorn", "iterations": iterations, "alpha": alpha, "step": step}
            if any(cached.get(k) != v for k, v in expected.items()):
                raise ValueError("Existing cache has incompatible parameters; use --overwrite to regenerate")
            rec["status"] = "skipped"
            rec["output_graph"] = str(out_graph_path)
            rec["output_pt"] = str(out_pt_path)
            rec["note"] += "；检测到已有输出结果，已跳过"
            rec["error"] = "already_exists"
            return rec

        G = load_graph(graph_file)
        flow_data = run_flow_no_merge(G, method=method, iterations=iterations, step=step, weight_name="weight", alpha=alpha, sinkhorn_reg=sinkhorn_reg)
        out_g, out_pt = save_outputs(
            rel=rel,
            out_graph_dir=out_graph_dir,
            out_pt_dir=out_pt_dir,
            graph=G,
            flow_data=flow_data,
            dataset_name=dataset_name,
        )
        rec["status"] = "success"
        rec["num_nodes"] = str(flow_data["num_nodes"])
        rec["num_edges"] = str(flow_data["num_edges"])
        rec["shortest_path_sec"] = f"{float(flow_data.get('shortest_path_sec', 0.0)):.4f}"
        rec["flow_sec"] = f"{float(flow_data.get('flow_sec', 0.0)):.4f}"
        rec["output_graph"] = str(out_g)
        rec["output_pt"] = str(out_pt)
    except Exception as exc:
        rec["error"] = str(exc)
    finally:
        total = time.time() - t0
        rec["elapsed_sec"] = f"{total:.4f}"
        rec["total_sec"] = f"{total:.4f}"
    return rec


def write_reports(out_dir: Path, rows: List[Dict[str, str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ricci_flow_summary.csv"
    json_path = out_dir / "ricci_flow_summary.json"
    fields = [
        "dataset",
        "input_graph",
        "method",
        "iterations",
        "status",
        "elapsed_sec",
        "total_sec",
        "shortest_path_sec",
        "flow_sec",
        "num_nodes",
        "num_edges",
        "output_graph",
        "output_pt",
        "note",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    payload = {
        "total": len(rows),
        "success": sum(r["status"] == "success" for r in rows),
        "skipped": sum(r["status"] == "skipped" for r in rows),
        "failed": sum(r["status"] == "failed" for r in rows),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="批量计算 Ricci flow 并保存（不合并边，不改变图规模）")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/networkx"),
        help="输入目录（包含 .gpkl/.gbkl）",
    )
    parser.add_argument(
        "--out-graph-dir",
        type=Path,
        default=Path("data/ricci_flow_graphs"),
        help="输出图目录（保存 flow 后图）",
    )
    parser.add_argument(
        "--out-pt-dir",
        type=Path,
        default=Path("data/ricci_flow_pt"),
        help="输出 pt 目录（保存 edge_index/curvatures/weights 历史）",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("data"),
        help="summary(csv/json) 输出目录",
    )
    parser.add_argument("--method", choices=["Sinkhorn"], default="Sinkhorn")
    parser.add_argument("--iterations", type=int, default=10, help="Number of flow updates")
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--max-files", type=int, default=0, help="只跑前 N 个文件，0 表示全量")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出结果（默认跳过）")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sinkhorn-reg", type=float, default=0.1)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    out_graph_dir = args.out_graph_dir.resolve()
    out_pt_dir = args.out_pt_dir.resolve()
    summary_dir = args.summary_dir.resolve()

    files = discover_graph_files(input_dir)
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        print(f"[INFO] 未找到图文件: {input_dir}")
        write_reports(summary_dir, [])
        return 0

    force_iterations = args.iterations
    rows: List[Dict[str, str]] = []

    for idx, fp in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {fp}")
        rec = run_one(
            graph_file=fp,
            data_root=input_dir,
            out_graph_dir=out_graph_dir,
            out_pt_dir=out_pt_dir,
            force_iterations=force_iterations,
            step=args.step,
            overwrite=args.overwrite,
            alpha=args.alpha,
            sinkhorn_reg=args.sinkhorn_reg,
        )
        rows.append(rec)
        if rec["status"] == "success":
            print(
                f"  -> OK method={rec['method']} iter={rec['iterations']} "
                f"|V|={rec['num_nodes']} |E|={rec['num_edges']} "
                f"sp={rec['shortest_path_sec']}s flow={rec['flow_sec']}s total={rec['total_sec']}s"
            )
        elif rec["status"] == "skipped":
            print("  -> SKIP 已有结果，未重复计算")
        else:
            print(f"  -> FAIL {rec['error']}")

    write_reports(summary_dir, rows)
    success = sum(r["status"] == "success" for r in rows)
    skipped = sum(r["status"] == "skipped" for r in rows)
    failed = len(rows) - success - skipped
    print(f"[DONE] success={success}, skipped={skipped}, failed={failed}")
    print(f"[DONE] summary={summary_dir / 'ricci_flow_summary.csv'}")
    print(f"[DONE] details={summary_dir / 'ricci_flow_summary.json'}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
