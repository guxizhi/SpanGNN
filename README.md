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
python QUEEN_GCN.py --data='reddit' --prob='gradient' --explain='Y' --memory=8500 --explain='Y' --th=0.11015
```

To run GraphSAGE with QUEEN and early-stop

```bash
python QUEEN_GraphSAGE.py --data='reddit' --prob='gradient' --memory=9500 --explain='Y' --th=0.11
```

The treshold for early-stop can fluctuate about the above ssetting value to stop earlier or later

## All figures of experimenteal results can be obtained by draw.py

```bash
python draw.py
```
