# QUEEN

QUEEN is a quick incremtnal memory-efficent graph augmentation framework with lower augmentation complexity and peak GPU memory usage.

## To prepare the environment for running QUEEN.

```bash
pip install -r requirements.txt
```

## Quick Strat

### How to run QUEEN without mutal information based ealy-stop strategy?

Runing GCN with QUEEN, the memory limitation is allowed to change according to your need.

```bash
python QUEEN_GCN.py --data='reddit' --prob='gradient' --memory=8500  
python QUEEN_GCN.py --data='reddit' --prob='gradient' --memory=8000
```

Runing GraphSAGE with QUEEN

```bash
python QUEEN_GraphSAGE.py --data='reddit' --prob='gradient' --memory=9500
python QUEEN_GraphSAGE.py --data='reddit' --prob='gradient' --memory=9000
```

To change edge sampling strategy, set prob='feature' or prob='gradient'. 

```bash
python QUEEN_GCN.py --data='reddit' --prob='feature' --memory=8500  
```

To change dataset, setting data='xxx' and we support 'cora', 'citesser', 'pubmed', 'reddit', 'products', 'proteins' and 'amazon'. Amazon dataset can be obtained from graphsaint, while other directly obtained by dgl.data.

### How to run QUEEN with mutual information based early-stop strategy?

Runing GCN with QUEEN and early-stop:

```bash
python QUEEN_GCN.py --data='reddit' --prob='gradient' --memory=8500 --explain='Y' --th=0.11015
```

Runing GraphSAGE with QUEEN and early-stop:

```bash
python QUEEN_GraphSAGE.py --data='reddit' --prob='gradient' --memory=9500 --explain='Y' --th=0.11
```
The treshold for early-stop can fluctuate around the above setting value to stop earlier or later.

## Logs

The training results, including GNN model, total trainig time, total augmentation time and val precision will be recorded in 'results.txt'.

The mutual information based score for mearsure the influential information contained in each partial structure will be recorded in 'record.txt'.

## All figures of experimental results can be obtained by draw.py.

```bash
python draw.py
```
