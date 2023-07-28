# QUEEN

QUEEN is a quick incremtnal memory-efficent graph augmentation framework

## How to run QUEEN without mutal information based ealy-stop strategy?

To run GCN with QUEEN

```bash
python QUEEN_GCN.py --data='reddit' --prob='gradient' --memory=8500
```

To run GraphSAGE with QUEEN

```bash
python QUEEN_GraphSAGE.py --data='reddit' --prob='gradient' --memory=9500
```

To change edge sampling strategy, just set prob='feature'

## How to run QUEEN with mutual information based early-stop strategy?

To run GCN with QUEEN and early-stop

```bash
python QUEEN_GCN.py --data='reddit' --prob='gradient' --explain='Y' --th=
```

To run GraphSAGE with QUEEN and early-stop

```bash
python QUEEN_GraphSAGE.py --data='reddit' --prob='gradient' --explain='Y'
```
