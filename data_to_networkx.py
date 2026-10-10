"""
将 data 目录下的图数据批量转为 NetworkX 图。
支持：
  - *.npz：edges / edge_index，可选 edge_weight / edge_weights / weights 等
  - PyTorch Geometric 保存的 *.pt（Data 或 (dict, ...) 元组）
  - Ricci flow 导出的 dict（含 edge_index、weights、curvatures）
边权重：若存在 edge_attr（PyG）或 npz 中的权重数组，则写入边的属性名 weight。
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch

try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import (
        BaseStorage,
        EdgeStorage,
        GlobalStorage,
        NodeStorage,
    )

    torch.serialization.add_safe_globals(
        [
            torch_geometric.data.Data,
            DataEdgeAttr,
            DataTensorAttr,
            GlobalStorage,
            NodeStorage,
            EdgeStorage,
            BaseStorage,
        ]
    )
except ImportError:  # pragma: no cover
    torch_geometric = None
    Data = None  # type: ignore[misc, assignment]


def normalize_edge_index(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got shape={a.shape}")
    if a.shape[0] == 2:
        return a.astype(np.int64, copy=False)
    if a.shape[1] == 2:
        return a.T.astype(np.int64, copy=False)
    raise ValueError(f"edge_index must be [2, E] or [E, 2], got shape={a.shape}")


def infer_num_nodes(
    payload: Dict[str, np.ndarray],
    edge_index: np.ndarray,
) -> int:
    if "num_nodes" in payload:
        return int(np.asarray(payload["num_nodes"]).reshape(-1)[0])
    for key in ("node_features", "node_feat", "x"):
        if key in payload:
            return int(np.asarray(payload[key]).shape[0])
    if edge_index.size == 0:
        return 0
    return int(edge_index.max()) + 1


def _find_edge_weights(
    payload: Dict[str, np.ndarray],
    num_edges: int,
) -> Optional[np.ndarray]:
    for name in (
        "edge_weight",
        "edge_weights",
        "weights",
        "weight",
        "edge_attr",
    ):
        if name not in payload:
            continue
        w = np.asarray(payload[name]).reshape(-1)
        if w.size == num_edges:
            return w.astype(np.float64, copy=False)
    return None


def _mask_keys(payload: Dict[str, np.ndarray]) -> List[str]:
    return [k for k in payload if re.match(r"^(train|val|valid|test)_mask", k, re.I)]


def npz_to_networkx(npz_path: Path) -> nx.Graph:
    with np.load(npz_path, allow_pickle=True) as z:
        payload = {k: z[k] for k in z.files}

    if "edge_index" in payload:
        edge_index = normalize_edge_index(payload["edge_index"])
    elif "edges" in payload:
        edge_index = normalize_edge_index(payload["edges"])
    else:
        raise KeyError(f"{npz_path}: 需要 edge_index 或 edges")

    num_edges = edge_index.shape[1]
    num_nodes = infer_num_nodes(payload, edge_index)
    weights = _find_edge_weights(payload, num_edges)

    directed = False
    G = nx.Graph()
    G.graph["source_path"] = str(npz_path)
    G.graph["directed"] = directed

    for i in range(num_nodes):
        G.add_node(i)

    node_features = None
    for key in ("node_features", "node_feat", "x"):
        if key in payload:
            node_features = np.asarray(payload[key])
            break

    node_labels = None
    for key in ("node_labels", "y", "label"):
        if key in payload:
            node_labels = np.asarray(payload[key]).reshape(-1)
            break

    if node_features is not None:
        for i in range(min(num_nodes, node_features.shape[0])):
            G.nodes[i]["feature"] = node_features[i]

    if node_labels is not None:
        for i in range(min(num_nodes, node_labels.shape[0])):
            G.nodes[i]["label"] = int(node_labels[i])

    for e in range(num_edges):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        ed: Dict[str, Any] = {}
        if weights is not None:
            ed["weight"] = float(weights[e])
        if ed:
            G.add_edge(u, v, **ed)
        else:
            G.add_edge(u, v)

    for mk in _mask_keys(payload):
        G.graph[mk] = np.asarray(payload[mk])

    return G


def load_torch_pyg_data(path: Path) -> "Data":
    if Data is None:
        raise RuntimeError("需要安装 torch_geometric 才能加载 PyG 的 .pt 文件")
    obj: Any = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, Data):
        return obj
    if isinstance(obj, tuple) and len(obj) >= 1:
        first = obj[0]
        if isinstance(first, Data):
            return first
        if isinstance(first, dict):
            return Data.from_dict(first)
    raise TypeError(f"{path}: 无法解析为 torch_geometric.data.Data")


def _edge_weights_from_edge_attr(edge_attr: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if edge_attr is None:
        return None
    ea = edge_attr.detach().cpu().numpy()
    if ea.ndim == 1:
        return ea.astype(np.float64, copy=False)
    if ea.ndim == 2 and ea.shape[1] == 1:
        return ea.reshape(-1).astype(np.float64, copy=False)
    # 多维 edge_attr：第一列作为 weight，其余保留在边上
    return ea.astype(np.float64, copy=False)


def pyg_data_to_networkx(data: "Data", source_path: Path) -> nx.Graph:
    """与 notebook 中 pyg_to_nx 一致，并支持 edge_attr -> weight。"""
    edge_index = data.edge_index.cpu().numpy()
    num_nodes = int(data.num_nodes)
    node_features = data.x.cpu().numpy() if data.x is not None else None
    node_labels = data.y.cpu().numpy().reshape(-1) if data.y is not None else None

    ea = _edge_weights_from_edge_attr(
        data.edge_attr if hasattr(data, "edge_attr") else None
    )

    G = nx.Graph()
    G.graph["source_path"] = str(source_path)
    G.graph["directed"] = False

    for node_id in range(num_nodes):
        attrs: Dict[str, Any] = {}
        if node_features is not None:
            attrs["feature"] = node_features[node_id]
        if node_labels is not None:
            attrs["label"] = int(node_labels[node_id])
        G.add_node(node_id, **attrs)

    for e in range(edge_index.shape[1]):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        ed: Dict[str, Any] = {}
        if ea is not None:
            if ea.ndim == 1:
                ed["weight"] = float(ea[e])
            else:
                ed["weight"] = float(ea[e, 0])
                ed["edge_attr"] = ea[e].tolist()
        if ed:
            G.add_edge(u, v, **ed)
        else:
            G.add_edge(u, v)

    return G


def ricci_flow_dict_to_networkx(d: Dict[str, Any], source_path: Path) -> nx.Graph:
    edge_index = d["edge_index"]
    if isinstance(edge_index, torch.Tensor):
        ei = edge_index.cpu().numpy()
    else:
        ei = np.asarray(edge_index)
    ei = normalize_edge_index(ei)

    weights_t = d.get("weights")
    curv_t = d.get("curvatures")
    w_last: Optional[np.ndarray] = None
    c_last: Optional[np.ndarray] = None
    if isinstance(weights_t, torch.Tensor) and weights_t.dim() >= 2:
        w_last = weights_t[-1].detach().cpu().numpy().reshape(-1)
    elif isinstance(weights_t, torch.Tensor):
        w_last = weights_t.detach().cpu().numpy().reshape(-1)

    if isinstance(curv_t, torch.Tensor) and curv_t.dim() >= 2:
        c_last = curv_t[-1].detach().cpu().numpy().reshape(-1)
    elif isinstance(curv_t, torch.Tensor):
        c_last = curv_t.detach().cpu().numpy().reshape(-1)

    G = nx.DiGraph() if d.get("is_directed", False) else nx.Graph()
    G.graph["source_path"] = str(source_path)
    G.graph["dataset_name"] = d.get("dataset_name", "")
    G.graph["ricci_iterations"] = d.get("iterations", None)

    n_nodes = int(d.get("num_nodes", int(ei.max()) + 1 if ei.size else 0))
    for i in range(n_nodes):
        G.add_node(i)

    for e in range(ei.shape[1]):
        u, v = int(ei[0, e]), int(ei[1, e])
        ed: Dict[str, Any] = {}
        if w_last is not None and e < len(w_last):
            ed["weight"] = float(w_last[e])
        if c_last is not None and e < len(c_last):
            ed["ricci_curvature"] = float(c_last[e])
        if ed:
            G.add_edge(u, v, **ed)
        else:
            G.add_edge(u, v)

    return G


def is_ricci_flow_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    need = {"edge_index", "weights", "curvatures"}
    return need.issubset(obj.keys())


def pt_to_networkx(pt_path: Path) -> nx.Graph:
    obj: Any = torch.load(pt_path, map_location="cpu", weights_only=False)
    if is_ricci_flow_dict(obj):
        return ricci_flow_dict_to_networkx(obj, pt_path)
    if Data is not None:
        try:
            data = load_torch_pyg_data(pt_path)
            return pyg_data_to_networkx(data, pt_path)
        except TypeError:
            pass
    raise TypeError(f"{pt_path}: 不支持的 .pt 内容类型 {type(obj)}")


def iter_graph_files(
    data_dir: Path,
    include_subdir_names: Optional[Iterable[str]] = None,
) -> List[Path]:
    """收集 *.npz 与 *.pt；默认跳过明显非图数据的 pt。"""
    skip_pt_names = {"pre_filter.pt", "pre_transform.pt"}
    files: List[Path] = []
    for p in sorted(data_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(data_dir)
        if include_subdir_names is not None:
            if rel.parts and rel.parts[0] not in include_subdir_names:
                continue
        if any(part in {"networkx", "ricci_flow_graphs", "ricci_flow_pt"} for part in rel.parts):
            continue
        if p.suffix.lower() == ".npz":
            with np.load(p, allow_pickle=True) as payload:
                if "edges" in payload or "edge_index" in payload:
                    files.append(p)
        elif p.suffix.lower() == ".pt":
            if p.name in skip_pt_names:
                continue
            files.append(p)
    return files


def convert_all(
    data_dir: Path,
    out_dir: Optional[Path] = None,
) -> Dict[str, nx.Graph]:
    data_dir = data_dir.resolve()
    results: Dict[str, nx.Graph] = {}
    paths = iter_graph_files(data_dir)
    for path in paths:
        rel = path.relative_to(data_dir)
        key = str(rel).replace("\\", "/")
        try:
            if path.suffix.lower() == ".npz":
                G = npz_to_networkx(path)
            else:
                G = pt_to_networkx(path)
            G.remove_edges_from(nx.selfloop_edges(G))
            results[key] = G
            if out_dir is not None:
                out_path = out_dir / rel.with_suffix(".gpkl")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("wb") as f:
                    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:  # noqa: BLE001
            print(f"[SKIP] {key}: {exc}", file=sys.stderr)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="将 data 下 npz/pt 转为 NetworkX 图（含边权重）。")
    parser.add_argument("--datasets", nargs="+", help="Load named datasets through PyG/OGB")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="数据根目录（默认 ./data）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/networkx"),
        help="Output directory for NetworkX graphs",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else None

    if not data_dir.is_dir():
        print(f"目录不存在: {data_dir}", file=sys.stderr)
        return 1

    if args.datasets:
        from ricci_curvature import load_graph, canonical
        graphs = {}
        for name in args.datasets:
            graph = load_graph(name, data_dir)
            destination = out_dir / (canonical(name).replace(" ", "_") + ".gpkl")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as fp:
                pickle.dump(graph, fp, protocol=pickle.HIGHEST_PROTOCOL)
            graphs[name] = graph
    else:
        graphs = convert_all(data_dir, out_dir=out_dir)
    print(f"成功转换 {len(graphs)} 个图（失败项已在 stderr 中标注）。")
    for k, g in sorted(graphs.items()):
        print(f"  {k}: |V|={g.number_of_nodes()} |E|={g.number_of_edges()}")
    if out_dir:
        print(f"已写入: {out_dir}")
    return int(not args.datasets and len(graphs) != len(iter_graph_files(data_dir)))


if __name__ == "__main__":
    raise SystemExit(main())
