# memory experiments
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from numpy.random import randn
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time

# data = {
#     "Reddit": {
#         "x": [30, 40, 50, 60, 70],
#         "y_values": [
#             [10919,10919,10919,10919,10919],
#             [8107,7659,7215,6769,6321],
#             [8107,7659,7215,6769,6321],
#             [8107,7659,7215,6769,6321]
#         ],
#         "ylabel": "Peak Memory"
#     },
#     "Ogbn-products": {
#         "x": [30, 40, 50, 60, 70],
#         "y_values": [
#             [23022,23022,23022,23022,23022],
#             [20555,19729,18905,18081,17257],
#             [20555,19729,18905,18081,17257],
#             [20555,19729,18905,18081,17257]
#         ],
#         "markers": ['o', 'x', 's', '+'],
#         "ylabel": "Peak Memory"
#     },
#     "Ogbn-proteins": {
#         "x": [30, 40, 50, 60, 70],
#         "y_values": [
#             [7754,7754,7754,7754,7754],
#             [6213,5687,5173,4657,4139],
#             [6213,5687,5173,4657,4139],
#             [6213,5687,5173,4657,4139]
#         ],
#         "markers": ['o', 'x', 's', '+'],
#         "ylabel": "Peak Memory"
#     },
#     "Amazon": {
#         "x": [30, 40, 50, 60, 70],
#         "y_values": [
#             [24166,24166,24166,24166,24166],
#             [16039,16039,16039,16039,16039],
#             [19985,19985,19047,17989,16993],
#             [21078,20065,19047,17989,16993]
#         ],
#         "markers": ['o', 'x', 's', '+'],
#         "ylabel": "Peak Memory"
#     }
# }

data = {
    "Reddit": {
        "x": [30, 40, 50, 60, 70],
        "y_values": [
            [11002,11002,11002,11002,11002],
            [9403,8867,8335,7803,7267],
            [9403,8867,8335,7803,7267],
            [9403,8867,8335,7803,7267]
        ],
        "markers": ['o', 'x', 's', '+'],
        "ylabel": "Acc"
    },
    "Ogbn-products": {
        "x": [30, 40, 50, 60, 70],
        "y_values": [
            [27750,27750,27750,27750,27750],
            [24839,23867,22899,21931,20963],
            [24839,23867,22899,21931,20963],
            [24839,23867,22899,21931,20963]
        ],
        "markers": ['o', 'x', 's', '+'],
        "ylabel": "Acc"
    },
    "Ogbn-proteins": {
        "x": [30, 40, 50, 60, 70],
        "y_values": [
            [9000,9000,9000,9000,9000],
            [7187,6591,5967,5359,4751],
            [7187,6591,5967,5359,4751],
            [7187,6591,5967,5359,4751]
        ],
        "markers": ['o', 'x', 's', '+'],
        "ylabel": "AUC-ROC"
    },
    "Amazon": {
        "x": [30, 40, 50, 60, 70],
        "y_values": [
            [27932,27932,27932,27932,27932],
            [18209,18209,18209,18209,18209],
            [22929,22929,21895,20545,19337],
            [24244,23572,21895,20545,19337]
        ],
        "markers": ['o', 'x', 's', '+'],
        "ylabel": "F1-score"
    }
}

# Adjusting the plot layout to have four subplots in one line and using a prettier color scheme

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

# Titles and legends
titles = ["Reddit", "Ogbn-products", "Ogbn-proteins", "Amazon"]
legends = ['SAGE+Full-graph', 'SAGE+SpanGNN-F', 'SAGE+SpanGNN-G', 'SAGE+DropEdge']

# Prettier colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

i = 0
for ax, (title, info) in zip(axes, data.items()):
    for y, color in zip(info["y_values"], colors):
        if i == 0:
            ax.plot(info["x"], y, markersize=10, color=color, linestyle='--')
        elif i == 1: 
            ax.bar([x-2 for x in info["x"]], y, color=color, width=2)
        elif i == 2: 
            ax.bar(info["x"], y, color=color, width=2)
        elif i == 3: 
            ax.bar([x+2 for x in info["x"]], y, color=color, width=2)
        i += 1
    i = 0
    ax.set_title(title, fontsize=25)
    ax.set_xlabel(r'Edge ratio $ \alpha_{up} $', fontsize=25)
    ax.set_ylabel(info["ylabel"], fontsize=25)
    ax.set_xticks([30, 40, 50, 60, 70])
    ax.tick_params(axis='both', which='major', labelsize=16)

# Adjust legends to appear once and prettier
fig.legend(legends, loc='upper center', ncol=5, fontsize=25, bbox_to_anchor=(0.5, 1))

# Adjust layout for better appearance
plt.tight_layout(rect=[0, 0, 1, 0.85])

# Show the updated plot
plt.show()
plt.savefig('../fig_new/merged_sage_memory.pdf', )
