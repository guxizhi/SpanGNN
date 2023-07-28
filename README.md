# QUEEN

QUEEN is a quick incremtnal memory-efficent graph augmentation framework with lower augmentation complexity and peak GPU memory usage.

## How to run QUEEN without mutal information based ealy-stop strategy?

To run GCN with QUEEN

```bash
python QUEEN_GCN.py --data='reddit' --prob='gradient' --memory=8500
```

To run GraphSAGE with QUEEN

```bash
python QUEEN_GraphSAGE.py --data='reddit' --prob='gradient' --memory=9500
```

To change edge sampling strategy, just set prob='feature'.  
To set memory limitation advanced, just set memory=xxx.  
To change dataset， just set data='xxx' and we support Cora, Citesser, Pubmed, Reddit, ogbn-products, ogbn-proteins and Amazon now (should be in lower case).

## How to run QUEEN with mutual information based early-stop strategy?

To run GCN with QUEEN and early-stop:

```bash
python QUEEN_GCN.py --data='reddit' --prob='gradient' --explain='Y' --memory=8500 --explain='Y' --th=0.11015
```

To run GraphSAGE with QUEEN and early-stop:

```bash
python QUEEN_GraphSAGE.py --data='reddit' --prob='gradient' --memory=9500 --explain='Y' --th=0.11
```

The treshold for early-stop can fluctuate around the above setting value to stop earlier or later.  
And the mutual information releated result at each iteration will be recored in 'record.txt'.

## All figures of experimenteal results can be obtained by draw.py.

```bash
python draw.py
```
