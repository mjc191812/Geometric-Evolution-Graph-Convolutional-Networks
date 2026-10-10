"""Train GEGCN on any of the 17 datasets used in the manuscript.

Examples:
  python main.py --dataset chameleon_filtered --ricci-pt data/ricci_flow_pt/chameleon_filtered.pt --split-file path/to/masks.npz
  python main.py --dataset Cora --ricci-pt data/ricci_flow_pt/Cora.pt --split-file path/to/masks.npz
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional
from sklearn.metrics import roc_auc_score

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Actor, Coauthor, Planetoid, WebKB, WikipediaNetwork

NPZ_FILES = {
    "chameleon_filtered": "chameleon_filtered.npz", "squirrel_filtered": "squirrel_filtered.npz",
    "roman_empire": "roman_empire.npz", "amazon_ratings": "amazon_ratings.npz",
    "minesweeper": "minesweeper.npz", "questions": "questions.npz",
}
ALIASES = {
    "pubmed": "PubMed", "coauthorcs": "Coauthor CS", "cs": "Coauthor CS",
    "photo": "Amazon Photos", "amazonphoto": "Amazon Photos", "amazonphotos": "Amazon Photos", "chameleon-f": "chameleon_filtered",
    "squirrel-f": "squirrel_filtered", "roman-empire": "roman_empire", "amazon-ratings": "amazon_ratings",
    "ogbn-arxiv": "ogbn_arxiv", "cora": "Cora", "citeseer": "Citeseer", "cornell": "Cornell", "texas": "Texas", "wisconsin": "Wisconsin", "chameleon": "Chameleon", "actor": "Actor",
}

def canonical(name: str) -> str:
    key = name.strip().lower().replace(" ", "").replace("_", "_")
    return ALIASES.get(key, key if key in NPZ_FILES else name)

def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_npz(name: str, root: Path, split_id: int) -> Data:
    z = np.load(root / NPZ_FILES[name], allow_pickle=True)
    x = torch.as_tensor(z["node_features"], dtype=torch.float32)
    y = torch.as_tensor(z["node_labels"], dtype=torch.long).view(-1)
    e = torch.as_tensor(z["edges"], dtype=torch.long).t().contiguous()
    data = Data(x=x, y=y, edge_index=e)
    return data

def load_data(name: str, root: Path, split_id: int) -> Data:
    name = canonical(name)
    if name in NPZ_FILES:
        return load_npz(name, root, split_id)
    if name == "Cora" or name == "Citeseer" or name == "PubMed":
        return Planetoid(str(root), name, transform=None)[0]
    if name == "Coauthor CS":
        return Coauthor(str(root), name="CS", transform=None)[0]
    if name == "Amazon Photos":
        return Amazon(str(root), name="Photo", transform=None)[0]
    if name in {"Cornell", "Texas", "Wisconsin"}:
        return WebKB(str(root), name=name, transform=None)[0]
    if name == "Chameleon":
        return WikipediaNetwork(str(root), name="chameleon", transform=None)[0]
    if name == "Actor":
        return Actor(str(root), transform=None)[0]
    if name == "ogbn_arxiv":
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
            ds = PygNodePropPredDataset(name="ogbn-arxiv", root=str(root))
            data = ds[0]
            data.y = data.y.view(-1)
            return data
        except ImportError as exc:
            raise RuntimeError("OGBN-arXiv requires the optional 'ogb' package") from exc
    raise ValueError(f"Unsupported dataset: {name}")

def resolve_pt(name: str, path: Optional[str], root: Path) -> Path:
    if path:
        p = Path(path)
        if p.exists(): return p
        raise FileNotFoundError(p)
    key = canonical(name)
    candidates = {
        "Cora": [root/"ricci_flow_pt/Cora/processed/data.pt", root/"ricci_flow_pt/data.pt", root/"Cora_ricci_flow_new.pt"],
        "Citeseer": [root/"Citeseer_ricci_flow_new.pt", root/"ricci_flow_pt/Citeseer.pt"],
        "PubMed": [root/"PubMed_ricci_flow.pt", root/"Pubmed_ricci_flow.pt", root/"ricci_flow_pt/PubMed.pt"],
        "Coauthor CS": [root/"CS_ricci_flow.pt", root/"ricci_flow_pt/CS.pt"],
        "Amazon Photos": [root/"Photo_ricci_flow.pt", root/"ricci_flow_pt/Photo.pt"],
        "Cornell": [root/"Cornell_ricci_flow.pt", root/"ricci_flow_pt/cornell.pt"],
        "Texas": [root/"Texas_ricci_flow.pt", root/"ricci_flow_pt/texas.pt"],
        "Wisconsin": [root/"Wisconsin_ricci_flow.pt", root/"ricci_flow_pt/wisconsin.pt"],
        "Chameleon": [root/"Chameleon_ricci_flow.pt", root/"ricci_flow_pt/chameleon.pt"],
        "Actor": [root/"Actor_ricci_flow.pt", root/"ricci_flow_pt/actor.pt"],
        "ogbn_arxiv": [root/"ricci_flow_pt/ogbn_arxiv/processed/geometric_data_processed.pt"],
    }
    if key in NPZ_FILES: candidates[key] = [root/f"ricci_flow_pt/{key}.pt", root/f"{key}_ricci_flow.pt"]
    for p in [root / "ricci_flow_pt" / (key.replace(" ", "_") + ".pt"), *candidates.get(key, [])]:
        if p.exists(): return p
    raise FileNotFoundError(f"No Ricci-flow cache found for {name}; pass --ricci-pt explicitly")

def load_flow(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    required = {"edge_index", "curvatures", "weights"}
    if not isinstance(obj, dict) or not required.issubset(obj):
        raise ValueError(f"{path} must contain {sorted(required)}")
    if obj.get("flow_version") != "linear-rho-sinkhorn-v1":
        raise ValueError("Flow cache must use the linear-measure shortest-path Sinkhorn definition")
    edge_index = torch.as_tensor(obj["edge_index"], dtype=torch.long)
    curv = torch.as_tensor(obj["curvatures"], dtype=torch.float32)
    weights = torch.as_tensor(obj["weights"], dtype=torch.float32)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or curv.ndim != 2 or curv.shape[0] == 0 or weights.shape != curv.shape or edge_index.shape[1] != curv.shape[1]:
        raise ValueError(f"Incompatible flow tensors in {path}")
    if not torch.isfinite(curv).all() or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Flow values must be finite and weights positive")
    if obj.get("num_nodes") is not None and edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= obj["num_nodes"]):
        raise ValueError("Flow endpoint outside the node range")
    return edge_index, curv, weights

class EdgeRNN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__(); self.lstm = nn.LSTM(2, hidden, batch_first=True); self.head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
    def forward(self, seq): return self.head(self.lstm(seq)[0][:, -1]).squeeze(-1)

class GEGCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__(); self.rnn=EdgeRNN(hidden); self.lin1=nn.Linear(in_dim,hidden); self.lin2=nn.Linear(hidden,out_dim); self.drop=nn.Dropout(.5)
    def propagate(self, x, edge_index, score):
        src,dst=edge_index; d=torch.zeros(x.size(0),device=x.device).index_add_(0,dst,score).clamp_min(1e-12)
        norm=score*d[src].pow(-.5)*d[dst].pow(-.5); msg=self.lin1(x)[src]*norm[:,None]
        out=torch.zeros((x.size(0),msg.size(1)),device=x.device); return out.index_add_(0,dst,msg)
    def forward(self,x,edge_index,seq):
        score=self.rnn(seq); h=F.relu(self.propagate(self.drop(x),edge_index,score)); h=self.drop(h)
        src,dst=edge_index; d=torch.zeros(h.size(0),device=h.device).index_add_(0,dst,score).clamp_min(1e-12)
        norm=score*d[src].pow(-.5)*d[dst].pow(-.5); msg=self.lin2(h)[src]*norm[:,None]
        out=torch.zeros((h.size(0),msg.size(1)),device=h.device).index_add_(0,dst,msg)
        return F.log_softmax(out,dim=1)

def make_edges(flow_edge, curv, weights, n, device):
    if flow_edge.numel() and (flow_edge.min() < 0 or flow_edge.max() >= n):
        raise ValueError("Flow cache and dataset have incompatible node indices")
    if torch.any(flow_edge[0] == flow_edge[1]):
        raise ValueError("Flow cache must not include self-loops")
    keys = torch.sort(flow_edge, dim=0).values.t()
    if torch.unique(keys, dim=0).shape[0] != flow_edge.shape[1]:
        raise ValueError("Flow cache must contain each undirected edge once")
    e=torch.cat([flow_edge,flow_edge.flip(0)],1)
    seq=torch.stack([curv.T,weights.T],-1).repeat((2,1,1))
    loops=torch.arange(n); le=torch.stack([loops,loops]); ls=torch.zeros((n,curv.size(0),2)); ls[:,:,1]=1
    return torch.cat([e,le],1).to(device), torch.cat([seq,ls]).to(device)

def load_masks(data: Data, path: str, split_id: int = 0) -> None:
    with np.load(path, allow_pickle=False) as source:
        for attr in ("train_mask", "val_mask", "test_mask"):
            if attr not in source:
                raise ValueError(f"Split file is missing {attr}")
            mask = torch.as_tensor(source[attr])
            if mask.dtype != torch.bool and not torch.all((mask == 0) | (mask == 1)):
                raise ValueError(f"{attr} must be a boolean mask, not node indices")
            mask = mask.bool()
            if mask.ndim == 2:
                if mask.shape[0] == data.num_nodes:
                    if not 0 <= split_id < mask.shape[1]:
                        raise ValueError("split_id outside mask columns")
                    mask = mask[:, split_id]
                else:
                    if not 0 <= split_id < mask.shape[0]:
                        raise ValueError("split_id outside mask rows")
                    mask = mask[split_id]
            if mask.ndim != 1 or mask.numel() != data.num_nodes or not mask.any():
                raise ValueError(f"{attr} must select nodes from this dataset")
            setattr(data, attr, mask)
    if (data.train_mask & data.val_mask).any() or (data.train_mask & data.test_mask).any() or (data.val_mask & data.test_mask).any():
        raise ValueError("Training, validation and test masks must be disjoint")


def metric_for_dataset(name: str) -> str:
    return "roc_auc" if canonical(name).lower() in {"minesweeper", "questions"} else "accuracy"


def evaluate_predictions(log_probs, labels, mask, metric):
    selected_labels = labels[mask].detach().cpu().numpy()
    selected = log_probs[mask]
    if selected_labels.size == 0:
        raise ValueError("Evaluation mask is empty")
    if metric == "roc_auc":
        if selected.shape[1] != 2 or not np.isin(selected_labels, [0, 1]).all():
            raise ValueError("ROC-AUC expects binary labels 0/1 and two output classes")
        if np.unique(selected_labels).size != 2:
            raise ValueError("ROC-AUC requires both classes in the evaluation split")
        probabilities = selected.exp()[:, 1].detach().cpu().numpy()
        return float(roc_auc_score(selected_labels, probabilities))
    if metric != "accuracy":
        raise ValueError(f"Unsupported metric: {metric}")
    return float((selected.argmax(1) == labels[mask]).float().mean())


def run(args):
    seed_all(args.seed); root=Path(args.root); name=canonical(args.dataset); data=load_data(name,root,args.split_id)
    load_masks(data, args.split_file, args.split_id)
    metric = metric_for_dataset(name)
    flow=resolve_pt(name,args.ricci_pt,root); edge,curv,w=load_flow(flow); device=torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    edge,seq=make_edges(edge,curv,w,data.num_nodes,device); data=data.to(device); scores=[]
    for r in range(args.runs):
        seed_all(args.seed+r); model=GEGCN(data.num_features,args.hidden_dim,int(data.y.max())+1).to(device); opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); best=-1; state=None; wait=0
        for epoch in range(args.epochs):
            model.train(); opt.zero_grad(); out=model(data.x,edge,seq); loss=F.nll_loss(out[data.train_mask],data.y[data.train_mask]); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                val=evaluate_predictions(model(data.x,edge,seq), data.y, data.val_mask, metric)
            if val>best: best=val; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; wait=0
            else: wait+=1
            if wait>=args.patience: break
        model.load_state_dict(state); model.eval()
        with torch.no_grad():
            score=evaluate_predictions(model(data.x,edge,seq), data.y, data.test_mask, metric)
        scores.append(score); print(f"run={r} metric={metric} val={best:.4f} test={score:.4f}")
    print(f"dataset={name} metric={metric} cache={flow} mean={np.mean(scores):.4f} std={np.std(scores,ddof=1) if len(scores)>1 else 0:.4f}")

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True); p.add_argument("--root",default="./data"); p.add_argument("--ricci-pt",default=None); p.add_argument("--split-file",required=True); p.add_argument("--split-id",type=int,default=0); p.add_argument("--seed",type=int,default=42); p.add_argument("--runs",type=int,default=10); p.add_argument("--epochs",type=int,default=2000); p.add_argument("--patience",type=int,default=200); p.add_argument("--hidden-dim",type=int,default=64); p.add_argument("--lr",type=float,default=.005); p.add_argument("--weight-decay",type=float,default=5e-4); p.add_argument("--cpu",action="store_true"); run(p.parse_args())
