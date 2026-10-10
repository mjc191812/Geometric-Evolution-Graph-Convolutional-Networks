"""Benchmark Ollivier-Ricci flow preprocessing across the 17 manuscript datasets.

This script measures computation time. It does not write the cached .pt files;
use run_ricci_flow_and_save.py for serialized flow histories.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import networkx as nx
import torch
from torch_geometric.datasets import Amazon, Actor, Coauthor, Planetoid, WebKB, WikipediaNetwork

FILTERED = {
    "chameleon_filtered": "chameleon_filtered.npz", "squirrel_filtered": "squirrel_filtered.npz",
    "roman_empire": "roman_empire.npz", "amazon_ratings": "amazon_ratings.npz",
    "minesweeper": "minesweeper.npz", "questions": "questions.npz",
}
ALL_DATASETS = ["Cora", "Citeseer", "PubMed", "Coauthor CS", "Amazon Photos", "Cornell", "Texas", "Wisconsin", "Chameleon", "Actor", *FILTERED, "ogbn_arxiv"]
ALIASES = {"pubmed":"PubMed", "cs":"Coauthor CS", "coauthorcs":"Coauthor CS", "photo":"Amazon Photos", "chameleon-f":"chameleon_filtered", "squirrel-f":"squirrel_filtered", "ogbn-arxiv":"ogbn_arxiv", "coauthor cs":"Coauthor CS", "amazon photos":"Amazon Photos", "roman-empire":"roman_empire", "amazon-ratings":"amazon_ratings"}

def canonical(name):
    key=name.strip().lower()
    return ALIASES.get(key, {"cora":"Cora","citeseer":"Citeseer","cornell":"Cornell","texas":"Texas","wisconsin":"Wisconsin","chameleon":"Chameleon","actor":"Actor"}.get(key, name))

def from_npz(path: Path):
    z=np.load(path,allow_pickle=True); g=nx.Graph(); edges=np.asarray(z["edges"]); g.add_nodes_from(range(len(z["node_labels"])))
    g.add_edges_from((int(u),int(v)) for u,v in edges); g.remove_edges_from(nx.selfloop_edges(g)); return g

def load_graph(name, root: Path):
    name=canonical(name)
    if name in FILTERED: return from_npz(root/FILTERED[name])
    if name in {"Cora","Citeseer","PubMed"}: data=Planetoid(str(root),name)[0]
    elif name=="Coauthor CS": data=Coauthor(str(root),name="CS")[0]
    elif name=="Amazon Photos": data=Amazon(str(root),name="Photo")[0]
    elif name in {"Cornell","Texas","Wisconsin"}: data=WebKB(str(root),name=name)[0]
    elif name=="Chameleon": data=WikipediaNetwork(str(root),name="chameleon")[0]
    elif name=="Actor": data=Actor(str(root))[0]
    elif name=="ogbn_arxiv":
        from ogb.nodeproppred import PygNodePropPredDataset
        data=PygNodePropPredDataset(name="ogbn-arxiv",root=str(root))[0]
    else: raise ValueError(f"Unsupported dataset: {name}")
    g=nx.Graph(); g.add_nodes_from(range(data.num_nodes)); g.add_edges_from((int(u),int(v)) for u,v in data.edge_index.t().tolist()); g.remove_edges_from(nx.selfloop_edges(g)); return g

def compute_time(g, iterations: int, alpha: float, step: float = 0.1, sinkhorn_reg: float = 0.1):
    from run_ricci_flow_and_save import run_flow_no_merge
    t0 = time.perf_counter()
    run_flow_no_merge(g, "ollivier", iterations, alpha=alpha, step=step, sinkhorn_reg=sinkhorn_reg)
    return time.perf_counter() - t0


def main():
    p=argparse.ArgumentParser(); p.add_argument("--datasets",nargs="+",default=ALL_DATASETS); p.add_argument("--root",default="./data"); p.add_argument("--iterations",type=int,default=10); p.add_argument("--alpha",type=float,default=.5); p.add_argument("--step",type=float,default=.1); p.add_argument("--sinkhorn-reg",type=float,default=.1); args=p.parse_args()
    failed = False
    for raw in args.datasets:
        name=canonical(raw); print(f"Processing {name} ...",flush=True)
        try:
            g=load_graph(name,Path(args.root)); sec=compute_time(g,args.iterations,args.alpha,args.step,args.sinkhorn_reg); print(f"{name}: nodes={g.number_of_nodes()} edges={g.number_of_edges()} seconds={sec:.2f}")
        except Exception as exc:
            failed = True
            print(f"{name}: FAILED: {exc}")
    return int(failed)

if __name__ == "__main__": raise SystemExit(main())
