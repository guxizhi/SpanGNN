# ablation studies
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from numpy.random import randn
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time

# Data for the plots
# data = {
#     "Reddit": {
#         "x": [30, 40, 50, 60, 70],
#         "y_values": [
#             [95.04, 95.01, 95.02, 95.09, 95.03],
#             [94.42, 94.70, 94.81, 94.81, 94.94],
#             [93.35, 93.83, 94.03, 94.35, 94.65],
#             [94.90, 94.92, 95.04, 95.04, 95.03],
#             [94.45, 94.66, 94.71, 94.87, 94.88]
#         ],
#         "markers": ['o', 'x', 's', '+', 'D'],
#         "ylabel": "Acc"
#     },
#     "Ogbn-products": {
#         "x": [30, 40, 50, 60, 70],
#         "y_values": [
#             [91.81, 91.98, 91.93, 91.97, 91.98],
#             [91.09, 91.45, 91.68, 91.80, 91.85],
#             [90.54, 91.07, 91.50, 91.62, 91.67],
#             [91.61, 91.83, 91.86, 91.85, 91.91],
#             [91.10, 91.38, 91.65, 91.77, 91.78]
#         ],
#         "markers": ['o', 'x', 's', '+', 'D'],
#         "ylabel": "Acc"
#     },
#     "Ogbn-proteins": {
#         "x": [30, 40, 50, 60, 70],
#         "y_values": [
#             [86.72, 86.86, 87.04, 87.12, 87.21],
#             [86.42, 86.66, 86.88, 86.88, 86.96],
#             [86.49, 86.60, 86.72, 86.88, 86.97],
#             [86.57, 86.72, 86.78, 86.77, 86.91],
#             [86.25, 86.47, 86.64, 86.65, 86.72]
#         ],
#         "markers": ['o', 'x', 's', '+', 'D'],
#         "ylabel": "AUC-ROC"
#     },
#     "Amazon": {
#         "x": [30, 40, 50, 60, 70],
#         "y_values": [
#             [46.81, 46.81, 46.81, 46.81, 46.81],
#             [42.30, 45.68, 47.07, 47.68, 47.68],
#             [19.52, 24.54, 28.45, 32.35, 35.36],
#             [45.51, 46.33, 47.44, 48.05, 47.51],
#             [36.93, 40.11, 42.39, 44.20, 45.25]
#         ],
#         "markers": ['o', 'x', 's', '+', 'D'],
#         "ylabel": "F1-score"
#     }
# }

data = {
    "Reddit": {
        "x": [30, 40, 50, 60, 70],
        "y_values": [
            [96.28, 96.29, 96.33, 96.34, 96.34],
            [95.78, 96.04, 96.08, 96.22, 96.17],
            [95.27, 95.47, 95.51, 95.67, 95.86],
            [96.20, 96.19, 96.19, 96.20, 96.35],
            [95.83, 95.90, 95.97, 96.10, 96.04]
        ],
        "markers": ['o', 'x', 's', '+', 'D'],
        "ylabel": "Acc"
    },
    "Ogbn-products": {
        "x": [30, 40, 50, 60, 70],
        "y_values": [
            [91.91, 92.02, 92.06, 92.16, 92.16],
            [91.40, 91.72, 91.91, 92.01, 91.91],
            [91.31, 91.58, 91.78, 91.84, 91.83],
            [91.86, 92.03, 92.03, 92.04, 92.21],
            [91.48, 91.70, 91.88, 91.94, 92.04]
        ],
        "markers": ['o', 'x', 's', '+', 'D'],
        "ylabel": "Acc"
    },
    "Ogbn-proteins": {
        "x": [30, 40, 50, 60, 70],
        "y_values": [
            [89.85, 90.15, 90.19, 90.31, 90.44],
            [89.76, 89.97, 90.10, 90.25, 90.31],
            [89.55, 89.82, 90.01, 90.21, 90.28],
            [89.57, 89.81, 89.98, 89.99, 90.11],
            [89.47, 89.61, 89.78, 89.93, 90.13]
        ],
        "markers": ['o', 'x', 's', '+', 'D'],
        "ylabel": "AUC-ROC"
    },
    "Amazon": {
        "x": [30, 40, 50, 60, 70],
        "y_values": [
            [76.24, 76.24, 76.24, 76.24, 76.24],
            [76.25, 76.22, 76.25, 76.25, 76.25],
            [76.17, 76.26, 76.19, 76.28, 76.22],
            [76.04, 76.06, 76.08, 76.08, 76.11],
            [76.03, 76.03, 76.04, 76.05, 76.08]
        ],
        "markers": ['o', 'x', 's', '+', 'D'],
        "ylabel": "F1-score"
    }
}

# Adjusting the plot layout to have four subplots in one line and using a prettier color scheme

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

# Titles and legends
titles = ["Reddit", "Ogbn-products", "Ogbn-proteins", "Amazon"]
legends = ['SpanGNN-F', 'SpanGNN-G', 'SpanGNN w/o QA', 'SpanGNN-F w/o EE', 'SpanGNN-G w/o EE']

# Prettier colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for ax, (title, info) in zip(axes, data.items()):
    for y, color, marker in zip(info["y_values"], colors, info["markers"]):
        ax.plot(info["x"], y, marker=marker, markersize=10, color=color)
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
# plt.savefig('../fig_new/merged_gcn.pdf', )
plt.savefig('../fig_new/merged_sage.pdf', )
"""=======================LHZ==================================="""


"""=======================LHZ==================================="""
import matplotlib.pyplot as plt
import numpy as np

# Data
# datasets = ['Reddit', 'Ogbn-products', 'Ogbn-proteins', 'Amazon']
# num_list_scene1 = [0.8713, 1.5217, 1.1920, 1.2643]
# num_list_scene2 = [0.8714, 1.9974, 1.1560, 1.1714]
# num_list_scene3 = [2.7955, 3.3873, 2.6457, 7.1480]
# num_list_scene4 = [55.3957, 53.4607, 32.1757, 126.5801]
# # num_list_scene1 = [0.8750, 1.5670, 1.1617, 1.0463]
# # num_list_scene2 =[0.8752, 1.9769, 1.1506, 1.1714]
# # num_list_scene3 = [2.8193, 3.4410, 2.2677, 5.8413] # 3.21, 2.20, 1.95, 5.58
# # num_list_scene4 = [51.3030, 58.4994, 34.2332, 138.1929] # 37.33, 29.47, 132.08

# # Setup for plotting
# x = np.arange(len(datasets))  # the label locations
# width = 0.2  # the width of the bars

# fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.05})

# # Plot lower range data
# axs[0].bar(x - 1.5*width, num_list_scene1, width, label='SpanGNN', color='c')
# axs[0].bar(x - 0.5*width, num_list_scene2, width, label='SpanGNN w/o EE', color='#A52A2A')
# axs[0].bar(x + 0.5*width, num_list_scene3, width, label='DropEdge', color='b')
# axs[0].bar(x + 1.5*width, num_list_scene4, width, label='DS', color='#D2691E')

# # Plot higher range data
# axs[1].bar(x - 1.5*width, num_list_scene1, width, color='c')
# axs[1].bar(x - 0.5*width, num_list_scene2, width, color='#A52A2A')
# axs[1].bar(x + 0.5*width, num_list_scene3, width, color='b')
# axs[1].bar(x + 1.5*width, num_list_scene4, width, color='#D2691E')

# # Adding grid lines
# for ax in axs:
#     ax.grid(which='both', axis='y', linestyle='--', linewidth=0.5, color='grey')
#     ax.grid(which='both', axis='x', linestyle='--', linewidth=0.5, color='grey')

# # Adjustments to match previous customization
# axs[0].spines['bottom'].set_visible(False)
# axs[1].spines['top'].set_visible(False)
# axs[0].xaxis.tick_top()
# axs[0].tick_params(labeltop=False, labelsize=16)  # don't put tick labels at the top
# axs[1].tick_params(labelsize=16)
# axs[1].xaxis.tick_bottom()

# # Setting the y-axis tick interval
# axs[0].set_ylim(20, 140)
# axs[0].set_yticks(np.arange(20, 150, 60)) # Set a tick every 12 units to match the density of lower plot
# axs[1].set_ylim(0, 8)
# axs[1].set_yticks(np.arange(0, 9, 2)) # Set a tick every 2 units as per user request

# # Labels, titles, and custom adjustments
# axs[1].set_xticks(x)
# axs[1].set_xticklabels(datasets, fontsize=16)
# axs[0].legend(loc='upper left', fontsize=16, framealpha=0.5)
# axs[1].set_xlabel('Datasets', fontsize=20)
# axs[1].set_ylabel('Time', fontsize=20, y=0.7)
# # axs[0].set_title('Comparison of Computation Time Across Different Methods and Datasets')

# plt.tight_layout()
# plt.savefig('../fig/time_consume_sage.pdf', dpi=600)
# plt.show()
