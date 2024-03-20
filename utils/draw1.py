# full graph and DropEdge experiments
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
#             [95.02,95.02,95.02,95.02,95.02],
#             [93.49,93.72,94.35,94.54,94.73],
#             [95.04,95.01,95.02,95.09,95.03],
#             [94.42,94.70,94.81,94.81,94.94]
#         ],
#         "markers": ['o', 'x', 's', '+'],
#         "ylabel": "Acc"
#     },
#     "Ogbn-products": {
#         "x": [30, 40, 50, 60, 70],
#         "y_values": [
#             [91.98,91.98,91.98,91.98,91.98],
#             [90.58,91.19,91.48,91.67,91.74],
#             [91.81,91.98,91.93,91.97,91.98],
#             [91.09,91.45,91.68,91.80,91.85]
#         ],
#         "markers": ['o', 'x', 's', '+'],
#         "ylabel": "Acc"
#     },
#     "Ogbn-proteins": {
#         "x": [30, 40, 50, 60, 70],
#         "y_values": [
#             [86.90,86.90,86.90,86.90,86.90],
#             [86.18,86.41,86.56,86.66,86.72],
#             [86.72,86.86,87.04,87.12,87.21],
#             [86.42,86.66,86.88,86.88,86.92]
#         ],
#         "markers": ['o', 'x', 's', '+'],
#         "ylabel": "AUC-ROC"
#     },
#     "Amazon": {
#         "x": [30, 40, 50, 60, 70],
#         "y_values": [
#             [47.88,47.88,47.88,47.88,47.88],
#             [19.70,24.35,28.54,32.63,36.54],
#             [46.81,46.81,46.81,46.81,46.81],
#             [42.30,45.68,47.07,47.68,47.68]
#         ],
#         "markers": ['o', 'x', 's', '+'],
#         "ylabel": "F1-score"
#     }
# }

# data = {
#     "Reddit": {
#         "x": [30, 40, 50, 60, 70],
#         "y_values": [
#             [96.19,96.19,96.19,96.19,96.19],
#             [95.50,95.59,95.82,95.88,96.01],
#             [96.28,96.29,96.33,96.34,96.34],
#             [95.78,96.04,96.08,96.22,96.27]
#         ],
#         "markers": ['o', 'x', 's', '+'],
#         "ylabel": "Acc"
#     },
#     "Ogbn-products": {
#         "x": [30, 40, 50, 60, 70],
#         "y_values": [
#             [92.14,92.14,92.14,92.14,92.14],
#             [91.58,91.80,91.88,92.03,92.08],
#             [91.91,92.02,92.06,92.16,92.16],
#             [91.40,91.72,91.91,92.01,91.91]
#         ],
#         "markers": ['o', 'x', 's', '+'],
#         "ylabel": "Acc"
#     },
#     "Ogbn-proteins": {
#         "x": [30, 40, 50, 60, 70],
#         "y_values": [
#             [90.22,90.22,90.22,90.22,90.22],
#             [89.30,89.63,89.80,89.85,90.05],
#             [89.95,90.15,90.19,90.31,90.44],
#             [89.76,89.97,90.10,90.25,90.31]
#         ],
#         "markers": ['o', 'x', 's', '+'],
#         "ylabel": "AUC-ROC"
#     },
#     "Amazon": {
#         "x": [30, 40, 50, 60, 70],
#         "y_values": [
#             [76.05,76.05,76.05,76.05,76.05],
#             [75.98,75.96,75.98,76.02,76.00],
#             [76.24,76.24,76.24,76.24,76.24],
#             [76.25,76.22,76.25,76.25,76.25]
#         ],
#         "markers": ['o', 'x', 's', '+'],
#         "ylabel": "F1-score"
#     }
# }

# # Adjusting the plot layout to have four subplots in one line and using a prettier color scheme

# fig, axes = plt.subplots(1, 4, figsize=(24, 6))

# # Titles and legends
# titles = ["Reddit", "Ogbn-products", "Ogbn-proteins", "Amazon"]
# legends = ['SAGE+Full-graph', 'SAGE+DropEdge', 'SAGE+SpanGNN-F', 'SAGE+SpanGNN-G']

# # Prettier colors
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# i = 0
# for ax, (title, info) in zip(axes, data.items()):
#     for y, color, marker in zip(info["y_values"], colors, info["markers"]):
#         if i == 0:
#             ax.plot(info["x"], y, marker=marker, markersize=10, color=color, linestyle='--')
#         else: 
#             ax.plot(info["x"], y, marker=marker, markersize=10, color=color)
#         i += 1
#     i = 0
#     ax.set_title(title, fontsize=25)
#     ax.set_xlabel(r'Edge ratio $ \alpha_{up} $', fontsize=25)
#     ax.set_ylabel(info["ylabel"], fontsize=25)
#     ax.set_xticks([30, 40, 50, 60, 70])
#     ax.tick_params(axis='both', which='major', labelsize=16)

# # Adjust legends to appear once and prettier
# fig.legend(legends, loc='upper center', ncol=5, fontsize=25, bbox_to_anchor=(0.5, 1))

# # Adjust layout for better appearance
# plt.tight_layout(rect=[0, 0, 1, 0.85])

# # Show the updated plot
# plt.show()
# # plt.savefig('../fig_new/merged_gcn_fd.pdf', )
# plt.savefig('../fig_new/merged_sage_fd.pdf', )




# sensitive experiments
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()

# first-step = 50
x1 = [10,20,30]
y1 = [95.02,95.00,94.92]
# y1 = [96.33,96.19,96.23]

# first-step = 100
x2 = [10,20,30,40,50]
y2 = [95.06,95.08,95.08,94.99,94.93]
# y2 = [96.39,96.29,96.30,96.19,96.24]

plt.xticks([10,20,30,40,50])
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.plot(x1, y1, label='First-step Size = 50W', marker='o', markersize=10)
plt.plot(x2, y2, label='First-step Size = 100W', marker='x', markersize=10)
plt.xlabel("Second-step Size", fontsize=20)
plt.ylabel("Acc", fontsize=20)

plt.tight_layout()
plt.legend(fontsize=18)
# 显示图形
plt.show()

plt.savefig('../fig_new/gcn_sensitive.pdf', )
