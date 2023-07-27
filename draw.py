import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from numpy.random import randn
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time

# mutual information calculate api
# x = torch.randint(0, 2, [10000, 20])
# y = torch.randint(0, 2, [10000, 20])

# t1 = time.time()
# sum = 0
# for data1, data2 in zip(x, y):
#     sum += mutual_info_score(data1, data2)
# t2 = time.time()
# print("time1: ", t2 - t1)

# x = np.reshape(x, -1)
# y = np.reshape(y, -1)

# t3 = time.time()
# print(mutual_info_score(x, x))
# print(mutual_info_score(x, y))
# t4 = time.time()
# print("time2: ", t4 - t3)

# x = torch.tensor([0, 0, 0, 0, 0, 0, 1])
# y = torch.tensor([0, 0, 0, 0, 0, 1, 0])
# print(mutual_info_score(x, x))
# print(mutual_info_score(x, y))

def readTxtNumpy(filename, col):
    data = np.loadtxt(filename, dtype=np.float32, delimiter=' ', usecols=(1))
    return data[:]

# x = [i for i in range(3000)]
# # # # y0 = readTxtNumpy('./time_compare/sage_amazon.txt', 1)[:814]
# y = readTxtNumpy('record0.txt', 1)
# plt.title('early stop on products')
# plt.xlabel('epoch')
# plt.ylabel('MID')
# plt.plot(x, y, marker='o', markersize=1)
# print(np.percentile(y, [25, 50, 75]))
# y0_pre = []
# y1_pre = []
# sum0, sum1 = 0, 0
# # for t in y0:
# #     sum0 += t
# for (i, t) in enumerate(y1):
#     sum1 += t
#     if i > 2219: 
#         print(i)
#         break
# print(sum0, sum1)
# y0 = readTxtNumpy('record0.txt', 1)
# y1 = readTxtNumpy('record1.txt', 1)

# gcn reddit compared dropedge and QUEEN-GFR
x = [8500,8250,8000,7750,7500,7250,7000]
y1 = [95.15,95.12,95.06,94.93,94.80,94.71,94.60]
y2 = [95.26,95.29,95.22,95.07,95.07,95.04,95.04]
y3 = [95.39,95.39,95.31,95.30,95.46,95.35,95.33]
y4 = [94.94,94.95,94.71,94.70,94.64,94.35,94.13]
y5 = [95.41,95.23,95.26,95.22,95.13,95.15,95.09]
y6 = [95.33,95.33,95.39,95.35,95.44,95.37,95.42]

# gcn products compared dropedge and QUEEN-GFR
# x = [20000,19500,19000,18500,18000,17500,17000]
# y1 = [91.60,91.55,91.42,91.25,91.09,90.79,90.49]
# y2 = [91.50,91.42,91.47,91.40,91.36,91.21,90.88]
# y3 = [91.63,91.65,91.69,91.60,91.68,91.57,91.42]
# y4 = [91.34,91.44,91.19,91.12,90.81,90.61,90.13]
# y5 = [91.66,91.58,91.60,91.28,91.30,90.94,90.47]
# y6 = [91.77,91.81,91.84,91.80,91.72,91.61,91.57]

# gcn proteins compared dropedge and QUEEN-GFR
# x = [6500,6250,6000,5750,5500,5250,5000]
# y1 = [86.70,86.74,86.89,86.66,86.69,86.58,86.54]
# y2 = [87.11,87.03,86.98,86.92,86.91,86.82,86.80]
# y3 = [87.06,87.16,87.19,87.10,87.12,87.11,87.00]
# y4 = [86.98,86.94,86.98,86.88,86.84,86.76,86.75]
# y5 = [86.74,86.62,86.53,86.40,86.12,86.02,85.76]
# y6 = [86.94,86.98,86.91,86.84,86.82,86.68,86.52]

# gcn amazon compared dropedge and QUEEN-GFR
# x = [20000,19500,19000,18500,18000,17500,17000]
# y1 = [33.18,31.44,29.45,27.37,25.21,22.57,20.33]
# y2 = [47.79,47.57,47.39,47.18,46.51,45.70,44.26]
# y3 = [46.78,46.78,46.78,46.78,46.78,46.78,46.78]
# y4 = [33.58,31.46,29.26,27.54,25.34,22.86,20.44]
# y5 = [48.38,47.30,47.23,47.17,46.92,46.55,45.86]
# y6 = [46.90,46.90,46.90,46.90,46.90,46.90,46.90]

# sage reddit compared dropedge and QUEEN-GFR
# x = [9500,9250,9000,8750,8500,8250,8000]
# y1 = [96.20,96.11,96.13,96.09,95.95,96.00,95.81]
# y2 = [96.51,96.53,96.46,96.20,96.21,96.10,96.07]
# y3 = [96.51,96.63,96.62,96.56,96.62,96.57,96.62]
# y4 = [96.43,96.24,96.26,96.19,96.14,96.10,95.90]
# y5 = [96.27,96.22,96.12,96.05,96.02,95.95,95.61]
# y6 = [96.29,96.48,96.61,96.48,96.58,96.49,96.53]

# sage products compared dropedge and QUEEN-GFR
# x = [24000,23500,23000,22500,22000,21500,21000]
# y1 = [91.83,91.87,91.77,91.71,91.61,91.54,91.49]
# y2 = [91.33,91.28,91.26,91.00,91.08,90.78,90.63]
# y3 = [91.86,91.84,91.77,91.91,91.90,91.94,91.83]
# y4 = [91.72,91.62,91.59,91.59,91.63,91.40,91.21]
# y5 = [91.36,91.29,91.14,90.90,90.66,90.20,89.95]
# y6 = [91.92,91.86,91.82,91.84,91.83,91.67,91.52]

# sage proteins compared dropedge and QUEEN-GFR
# x = [7500,7250,7000,6750,6500,6250,6000]
# y1 = [90.02,90.03,89.89,89.90,89.79,89.80,89.75]
# y2 = [90.36,90.31,90.32,90.26,90.19,90.10,90.05]
# y3 = [90.48,90.41,90.48,90.44,90.49,90.27,90.34]
# y4 = [90.36,90.36,90.25,90.34,90.24,90.15,90.24]
# y5 = [90.10,89.93,89.94,89.92,89.76,89.74,89.70]
# y6 = [90.21,90.16,90.29,90.18,90.17,90.05,90.04]

# sage amazon compared dropedge and QUEEN-GFR
# x = [23000,22500,22000,21500,21000,20500,20000]
# y1 = [75.97,76.06,76.07,76.00,76.11,75.99,75.92]
# y2 = [76.28,76.30,76.24,76.24,76.20,76.21,76.29]
# y3 = [76.26,76.26,76.26,76.26,76.26,76.26,76.26]
# y4 = [76.26,76.23,76.23,76.22,76.20,76.27,76.26]
# y5 = [76.04,76.10,76.18,76.08,76.13,76.05,76.10]
# y6 = [75.97,75.97,75.97,75.97,75.97,75.97,75.97]

# plt.title('Acc as memory limitation decreasing',fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('memory',fontsize=15)
# plt.ylabel('acc',fontsize=15)
# plt.plot(x, y1, marker='o', markersize=5)
# plt.plot(x, y2, marker='x', markersize=5)
# plt.plot(x, y3, marker='^', markersize=5)
# plt.plot(x, y4, marker='s', markersize=5)
# plt.plot(x, y5, marker='+', markersize=5, linestyle='dotted')
# plt.plot(x, y6, marker='D', markersize=5, linestyle='dotted')
# plt.tight_layout()
# # , loc='center left', bbox_to_anchor=(0, 0.6)
# plt.legend(['DropEdge', 'QUEEN-G', 'QUEEN-F', 'QUEEN-R', 'Subgraph-G', 'Subgraph-F'])
# plt.gca().invert_xaxis() 
# plt.savefig('fig/gcn_proteins.png')

# sage amazon framework
# x = [146.3,135.6,124.8,114.1,103.3,92.6,81.8]
# y1 = [76.28,76.30,76.24,76.24,76.20,76.21,76.29]
# y2 = [76.04,76.10,76.18,76.08,76.13,76.05,76.10]
# y3 = [76.26,76.26,76.26,76.26,76.26,76.26,76.26]
# y4 = [75.97,75.97,75.97,75.97,75.97,75.97,75.97]

# sage products framework
# x = [79.5,73,66.6,60,53.5,46,40.5]
# y1 = [91.33,91.28,91.26,91.00,91.08,90.78,90.63]
# y2 = [91.36,91.29,91.14,90.90,90.66,90.20,89.95]
# y3 = [91.86,91.84,91.77,91.91,91.90,91.94,91.83]
# y4 = [91.92,91.86,91.82,91.84,91.83,91.67,91.52]

# sage reddit framework
# x = [81.3,75.5,70.3,64.8,59.4,54,48.8]
# y1 = [96.51,96.53,96.46,96.20,96.21,96.10,96.07]
# y2 = [96.27,96.22,96.12,96.05,96.02,95.95,95.61]
# y3 = [96.51,96.63,96.62,96.56,96.62,96.57,96.62]
# y4 = [96.29,96.48,96.61,96.48,96.58,96.49,96.53]

# sage proteins framework
# x = [58.8,55.6,52.4,49.0,45.9,42.5,39.3]
# y1 = [90.36,90.31,90.32,90.26,90.19,90.10,90.05]
# y2 = [90.10,89.93,89.94,89.92,89.76,89.74,89.70]
# y3 = [90.48,90.41,90.48,90.44,90.49,90.27,90.34]
# y4 = [90.21,90.16,90.29,90.18,90.17,90.05,90.04]

# gcn reddit framework
# x = [88.6,80.4,75.8,67.7,63.0,56.6,50.2]
# y1 = [95.26,95.29,95.22,95.07,95.07,95.04,95.04]
# y2 = [95.41,95.23,95.26,95.22,95.13,95.15,95.09]
# y3 = [95.39,95.39,95.31,95.30,95.46,95.35,95.33]
# y4 = [95.33,95.33,95.39,95.35,95.44,95.37,95.42]

# gcn products framework
# x = [82.3,74.7,67.1,59.3,51.7,44.1,36.5]
# y1 = [91.50,91.42,91.47,91.40,91.36,91.21,90.88]
# y2 = [91.66,91.58,91.60,91.28,91.30,90.94,90.47]
# y3 = [91.63,91.65,91.69,91.60,91.68,91.57,91.42]
# y4 = [91.77,91.81,91.84,91.80,91.72,91.61,91.57]

# gcn proteins framework
# x = [59.1,55.3,51.4,47.7,43.7,40.1,36.2]
# y1 = [87.11,87.03,86.98,86.92,86.91,86.82,86.80]
# y2 = [86.74,86.62,86.53,86.40,86.12,86.02,85.76]
# y3 = [87.06,87.16,87.19,87.10,87.12,87.11,87.00]
# y4 = [86.94,86.98,86.91,86.84,86.82,86.68,86.52]

# gcn amazon framework
# x = [156.7,149.9,137.4,124.3,111.5,98.6,85.5]
# y1 = [47.79,47.57,47.39,47.18,46.51,45.70,44.26]
# y2 = [48.38,47.30,47.23,47.17,46.92,46.55,45.86]
# y3 = [46.78,46.78,46.78,46.78,46.78,46.78,46.78]
# y4 = [46.90,46.90,46.90,46.90,46.90,46.90,46.90]

# x_ticks = np.arange(6000, 6750, 250)
# plt.xticks(x_ticks)
# plt.title('val acc as number of edges decreasing')
# plt.xlabel('the number of edges (Million)')
# plt.ylabel('acc')
# plt.plot(x, y1, marker='o', markersize=5, color='orange')
# plt.plot(x, y2, marker='o', markersize=5, color='orange', linestyle='dotted')
# plt.plot(x, y3, marker='x', markersize=5, color='blue')
# plt.plot(x, y4, marker='x', markersize=5, color='blue', linestyle='dotted')

# plt.legend(['QUEEN-G', 'Vanilla-G', 'QUEEN-F', 'Vanilla-F'])
# plt.gca().invert_xaxis() 
# plt.savefig('fig/gcn_proteins_framework.png')

# x = ['Amazon', 'ogbn-proteins', 'ogbn-products', 'Reddit']
# # graphsage
# y1 = [12277, 5296, 9484, 5054]
# y2 = [16197, 2852, 17905, 6015]
# # gcn
# y1 = [24763-16197, 7753-2483, 23029-14435, 9441-4965]
# y2 = [16197, 2483, 14435, 4965]

# for i in range(len(y1)):
#     y1[i] = y1[i] / (y1[i] + y2[i])
#     y2[i] = y2[i] / (y1[i] + y2[i])

# plt.bar(x,y1,width=0.4,label='edge-related',color='#f9766e',edgecolor='grey',zorder=5)
# plt.bar(x,y2,width=0.4,bottom=y1,label='edge-unrelated',color='#00bfc4',edgecolor='grey',zorder=5)
# plt.tick_params(axis='x',length=0)
# plt.xlabel('dataset',fontsize=12)
# plt.ylabel('percentage',fontsize=12)
# plt.ylim(0,1.01)
# plt.yticks(np.arange(0,1.2,0.2),[f'{i}%' for i in range(0,120,20)])
# plt.grid(axis='y',alpha=0.5,ls='--')
# # 添加图例，将图例移至图外
# plt.legend(ncols=2,loc='upper center',frameon=True,bbox_to_anchor=(0.5,1.16),prop={'size': 18})
# plt.tight_layout()
# plt.gcf().set_size_inches(9, 5)
# plt.savefig('gcn_memory.png', dpi=600)
# plt.show()

#-*- coding:utf-8 -*-

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mtick

# def main():
#     plt.rcdefaults()

#     info_list = [("QUEEN-G", 2650, 1391, 4344-2650), ("QUEEN-F", 5414, 2488, 6086-5414), 
#                  ("QUEEN-G", 5658, 1438, 8004-5658), ("QUEEN-F", 7528, 2331, 8473-7528), 
#                  ("QUEEN-G", 3039, 1169, 5258-3039), ("QUEEN-F", 4990, 1793, 6465-4990),
#                  ("QUEEN-G", 5160, 2277, 5870-5160), ("QUEEN-F", 6736, 2912, 6854-6736),
#                  ("QUEEN-G", 3951, 1562, 5918-3951), ("QUEEN-F", 5303, 1837, 6961-5303),
#                  ("QUEEN-G", 3070, 814, 9058-3070), ("QUEEN-F", 8648, 2568, 9885-8648)]
#     positions = np.arange(len(info_list))
#     names = [row[0] for row in info_list]
#     scores = [row[1] for row in info_list]
#     proges = [row[2] for row in info_list]
#     accumulate = [row[3] for row in info_list]

#     fig, ax1 = plt.subplots()

#     # 直方图
#     ax1.bar(positions, scores, width=0.5, color='lightgrey', align='center', label="train with aug", edgecolor='black')
#     ax1.bar(positions, accumulate, bottom=scores, color='dimgrey', width=0.5, align='center', label="train w/o aug", edgecolor='black')
#     ax1.set_xticks(positions)
#     ax1.set_xticks(positions)
#     ax1.set_xticklabels(names, rotation=90)
#     ax1.set_ylabel("time")
#     max_score = max(scores)
#     ax1.set_ylim(0, int(max_score * 1.2))

#     # 折线图
#     ax2 = ax1.twinx()
#     ax2.plot(positions[0: 2], proges[0: 2], 'x-', markersize=6, color='black', label="#epoch")
#     ax2.plot(positions[2: 4], proges[2: 4], 'x-', markersize=6, color='black')
#     ax2.plot(positions[4: 6], proges[4: 6], 'x-', markersize=6, color='black')
#     ax2.plot(positions[6: 8], proges[6: 8], 'x-', markersize=6, color='black')
#     ax2.plot(positions[8: 10], proges[8: 10], 'x-', markersize=6, color='black')
#     ax2.plot(positions[10: 12], proges[10: 12], 'x-', markersize=6, color='black')
#     # 设置纵轴格式
#     ax2.set_ylim(0, 3000)
#     ax2.set_ylabel("epoch")

#     # 图例
#     handles1, labels1 = ax1.get_legend_handles_labels()
#     handles2, labels2 = ax2.get_legend_handles_labels()
#     plt.legend(handles1+handles2, labels1+labels2, loc='upper center', ncol=3)
#     ax = plt.gca()
#     # ax.set_aspect(1)
#     plt.tight_layout()
#     plt.savefig('bar.png', dpi=600)
#     plt.show()


# if __name__ == '__main__':
#     main()

# draw figure for early stop sage
# reddit
# acc = [96.03,95.98,96.05,95.91,95.83,95.60]
# edges = [53.1,52.7,49.3,45.3,45.4,41.1]
# acc_base = [95.95,96.00,95.81]
# edges_base = [58.5,54.2,48.8]
# plt.subplot(2, 2, 1)
# plt.scatter(edges, acc, c='black')
# plt.scatter(edges_base, acc_base)
# plt.title("early stop on Reddit")
# x_b = plt.xticks()[0][0]
# y_b = plt.yticks()[0][-1]
# color = ['blue', 'green', 'yellow']
# for i, (x, y) in enumerate(zip(edges_base, acc_base)):
#     x_fill = [x_b, x_b, x, x]
#     y_fill = [y_b, y, y, y_b]
#     plt.fill(x_fill, y_fill, alpha=0.2, c=color[i])
# # proteins
# acc = [90.18,90.20,90.28,90.02,90.04,90.02]
# edges = [50.9,47.7,46.7,43.8,40.4,39.9]
# acc_base = [90.02,90.03,89.89]
# edges_base = [58.9,55.8,52.5]
# plt.subplot(2, 2, 2)
# plt.scatter(edges, acc, c='black')
# plt.scatter(edges_base, acc_base)
# plt.title("early stop on proteins")
# x_b = plt.xticks()[0][0]
# y_b = plt.yticks()[0][-1]
# color = ['blue', 'green', 'yellow']
# for i, (x, y) in enumerate(zip(edges_base, acc_base)):
#     x_fill = [x_b, x_b, x, x]
#     y_fill = [y_b, y, y, y_b]
#     plt.fill(x_fill, y_fill, alpha=0.2, c=color[i])
# # products
# acc = [91.76,91.72,91.67,91.57,91.64,91.56]
# edges = [36.2,34.3,33.1,31.7,30.1,28.4]
# acc_base = [91.64,91.54,91.49]
# edges_base = [53.6,47.1,40.6]
# plt.subplot(2, 2, 3)
# plt.scatter(edges, acc, c='black')
# plt.scatter(edges_base, acc_base)
# plt.title("early stop on products")
# x_b = plt.xticks()[0][0]
# y_b = plt.yticks()[0][-1]
# color = ['blue', 'green', 'yellow']
# for i, (x, y) in enumerate(zip(edges_base, acc_base)):
#     x_fill = [x_b, x_b, x, x]
#     y_fill = [y_b, y, y, y_b]
#     plt.fill(x_fill, y_fill, alpha=0.2, c=color[i])
# # amazon
# acc = [76.21,76.21,76.29,76.22,76.26,76.23]
# edges = [92.2,87.1,81.4,83.2,74.4,73.6]
# acc_base = [76.11,75.99,75.92]
# edges_base = [103.3,93.1,82.7]
# plt.subplot(2, 2, 4)
# plt.scatter(edges, acc, c='black')
# plt.scatter(edges_base, acc_base)
# plt.title("early stop on Amazon")
# x_b = plt.xticks()[0][0]
# y_b = plt.yticks()[0][-1]
# color = ['blue', 'green', 'yellow']
# for i, (x, y) in enumerate(zip(edges_base, acc_base)):
#     x_fill = [x_b, x_b, x, x]
#     y_fill = [y_b, y, y, y_b]
#     plt.fill(x_fill, y_fill, alpha=0.2, c=color[i])
# plt.tight_layout()
# plt.savefig('earlystop_sage.png', dpi=600)
# plt.show()

# early stop for gcn
# reddit
acc = [94.79,94.73,94.71,94.78,94.75,94.91]
edges = [38.4,39.3,38.1,35.8,36.8,38.6]
acc_base = [94.8,94.71,94.60]
edges_base = [63.2,56.7,50.3]
plt.subplot(2, 2, 1)
plt.scatter(edges, acc, c='black')
plt.scatter(edges_base, acc_base)
plt.title("early stop on Reddit")
x_b = plt.xticks()[0][0]
y_b = plt.yticks()[0][-1]
color = ['blue', 'green', 'yellow']
for i, (x, y) in enumerate(zip(edges_base, acc_base)):
    x_fill = [x_b, x_b, x, x]
    y_fill = [y_b, y, y, y_b]
    plt.fill(x_fill, y_fill, alpha=0.2, c=color[i])
# proteins
acc = [86.89,86.78,86.75,86.80,86.66,86.75]
edges = [39.1,37.6,34.7,34.2,32.2,36.3]
acc_base = [86.69,86.58,86.54]
edges_base = [43.9,40.4,36.4]
plt.subplot(2, 2, 2)
plt.scatter(edges, acc, c='black')
plt.scatter(edges_base, acc_base)
plt.title("early stop on proteins")
x_b = plt.xticks()[0][0]
y_b = plt.yticks()[0][-1]
color = ['blue', 'green', 'yellow']
for i, (x, y) in enumerate(zip(edges_base, acc_base)):
    x_fill = [x_b, x_b, x, x]
    y_fill = [y_b, y, y, y_b]
    plt.fill(x_fill, y_fill, alpha=0.2, c=color[i])
# amazon
acc = [42.52,42.54,42.73,42.51,42.47,42.35]
edges = [74.9,74.6,74.5,73.4,73.8,73.3]
acc_base = [25.21,22.57,20.33]
edges_base = [111.7,98.7,85.7]
plt.subplot(2, 2, 3)
plt.scatter(edges, acc, c='black')
plt.scatter(edges_base, acc_base)
plt.title("early stop on Amazon")
x_b = plt.xticks()[0][0]
y_b = plt.yticks()[0][-1]
color = ['blue', 'green', 'yellow']
for i, (x, y) in enumerate(zip(edges_base, acc_base)):
    x_fill = [x_b, x_b, x, x]
    y_fill = [y_b, y, y, y_b]
    plt.fill(x_fill, y_fill, alpha=0.2, c=color[i])
# products
x = [i for i in range(3000)]
# # # y0 = readTxtNumpy('./time_compare/sage_amazon.txt', 1)[:814]
y = readTxtNumpy('record0.txt', 1)
plt.subplot(2, 2, 4)
plt.title('early stop on products')
plt.xlabel('epoch')
plt.ylabel('MID')
plt.plot(x, y, marker='o', markersize=1)
plt.tight_layout()
plt.savefig('earlystop_gcn.png', dpi=600)
plt.show()

# def readTxtNumpy(filename, col):
#     data = np.loadtxt(filename, dtype=np.float32, delimiter=' ', usecols=(col))
#     return data[:]

# x = [i for i in range(3000)]
# y = readTxtNumpy('record0.txt', 1)
# y1 = readTxtNumpy('record0.txt', 2)
# color = ['b','c','g','k','m','r','y']
# for i in range(5):
#     y_temp = y[i*3000: (i+1)*3000]
#     y1_temp = y1[i*3000: (i+1)*3000]
#     print(np.where(y1_temp==np.max(y1_temp)))
#     plt.scatter(x, y_temp, c=color[i], s=1)
#     plt.scatter(x, y1_temp, c=color[i], s=1)

# plt.xlim((0, 3000))
# plt.ylim((0, 1))
# plt.savefig('smooth_compare.png')
# plt.show()   