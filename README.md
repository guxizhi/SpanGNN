# SpanGNN

SpanGNN: Towards Memory-Efficient Graph Neural Networks via Spanning Subgraph Training.

## To prepare the environment for running QUEEN.

```bash
pip install -r requirements.txt
```

## Quick Strat

### How to run SpanGNN?

Runing GCN with SpanGNN, the memory limitation is allowed to change according to your need.

```bash
python SpanGNN.py --data='reddit' --prob='gradient' --edge_ratio=0.3 --model='gcn'
python SpanGNN.py --data='reddit' --prob='gradient' --edge_ratio=0.4 --model='gcn'
```

Runing GraphSAGE with SpanGNN

```bash
python SpanGNN.py --data='reddit' --prob='gradient' --edge_ratio=0.3 --model='sage'
python SpanGNN.py --data='reddit' --prob='gradient' --edge_ratio=0.4 --model='sage'
```

To change edge sampling strategy, set prob='feature' or prob='gradient'. 

```bash
python SpanGNN.py --data='reddit' --prob='feature' --edge_ratio=0.3 --model='gcn'
```

To change dataset, setting data='xxx' and we support 'cora', 'citesser', 'pubmed', 'reddit', 'products', 'proteins' and 'amazon'. Amazon dataset can be obtained from graphsaint, while other directly obtained by dgl.data.

## Logs

The training results, including GNN model, total trainig time, total augmentation time and val precision will be recorded in 'results.txt'.

The mutual information based score for mearsure the influential information contained in each partial structure will be recorded in 'record.txt'.

## All figures of experimental results can be obtained by draw.py.

```bash
python draw.py
```
