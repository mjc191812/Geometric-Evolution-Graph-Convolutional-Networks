# Geometric Evolution Graph Convolutional Networks (GEGCN)

This repository contains the supplementary data and preprocessing code for GEGCN, a graph representation learning method that encodes discrete Ricci-flow states with an LSTM and uses the resulting edge scores for graph convolution.

## Repository contents

- `data/`: raw graph data, node features, and labels.
- `main.py`: training entry point for GEGCN on the supported datasets.
- `ricci_curvature.py`: Ollivier-Ricci preprocessing-time benchmark across the supported datasets.
- `run_ricci_flow_and_save.py`: computes and saves Ricci-flow histories.
- `convert_npz_to_pt.py`: converts NPZ inputs and flow histories into the `.pt` cache format.
- `data_to_networkx.py`: converts dataset or flow files to NetworkX graphs.

The preprocessing scripts generate the graph representations and flow features used by the model. Large `.pt` and `.gz` files are managed with Git LFS.

## Installation

```bash
pip install -r requirements.txt
```

For a CUDA-enabled PyTorch installation, install the matching PyTorch build first, then run the command above.

## Quick start

```bash
python data_to_networkx.py --datasets Cora --out-dir data/networkx
python run_ricci_flow_and_save.py --input-dir data/networkx --method Sinkhorn --iterations 10
python main.py --dataset Cora --ricci-pt data/ricci_flow_pt/Cora.pt --split-file path/to/masks.npz
```


## Preprocessing and evaluation

Ricci flow follows our [one-step random-walk formulation](https://github.com/mjc191812/Evolution-of-weights-on-a-connected-finite-graph/blob/bd2c16d5404e8c9112c9aa78d1465fc5f1b35527/evolution%20of%20graph%20weight/one_step/One-step%20alpha-lazy%20random%20walk.py#L258), using Sinkhorn and retaining all connected components.

Minesweeper and Questions use ROC-AUC; all other datasets use accuracy. Model selection uses the corresponding validation metric.

Generate the train/validation/test splits according to the protocols in the paper, then pass an NPZ file containing `train_mask`, `val_mask`, and `test_mask` through `--split-file`.
