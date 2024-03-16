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

# def readTxtNumpy(filename, col):
#     data = np.loadtxt(filename, dtype=np.float32, delimiter=' ', usecols=(1))
#     return data[:]

# # x = [i for i in range(3000)]
# # # # # y0 = readTxtNumpy('./time_compare/sage_amazon.txt', 1)[:814]
# y = readTxtNumpy('../record_sage.txt', 1)[33000:33350]
# # plt.title('early stop on products')
# # plt.xlabel('epoch')
# # plt.ylabel('MID')
# # plt.plot(x, y, marker='o', markersize=1)
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
# x = [70,60,50,40,30]
# y1 = [95.03,95.09,95.02,95.01,95.04]
# y2 = [94.94,94.81,94.81,94.70,94.42]
# y4 = [94.65,94.35,94.03,93.83,93.35]
# y5 = [95.03,95.04,95.04,94.92,94.90]
# y6 = [94.88,94.87,94.71,94.66,94.45]
# plt.title('Acc as edge ratio increasing',fontsize=15)
# my_x_ticks = np.arange(30, 71, 10)
# plt.xticks(my_x_ticks)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('edge ratio',fontsize=15)
# plt.ylabel('acc',fontsize=15)
# plt.plot(x[::-1], y1[::-1], marker='o', markersize=5)
# plt.plot(x[::-1], y2[::-1], marker='x', markersize=5)
# plt.plot(x[::-1], y4[::-1], marker='s', markersize=5)
# plt.plot(x[::-1], y5[::-1], marker='+', markersize=5)
# plt.plot(x[::-1], y6[::-1], marker='D', markersize=5)
# plt.tight_layout()
# plt.legend(['LPGNN-F', 'LPGNN-G', 'LPGNN w/o QA', 'LPGNN-F w/o CL', 'LPGNN-G w/o CL'],loc=4)
# plt.savefig('../fig/gcn_reddit.pdf', )


# gcn products compared dropedge and QUEEN-GFR
# x = [70,60,50,40,30]
# y1 = [91.98,91.97,91.93,91.98,91.81]
# y2 = [91.85,91.80,91.68,91.45,91.09]
# y4 = [91.67,91.62,91.50,91.07,90.54]
# y5 = [91.91,91.85,91.86,91.83,91.61]
# y6 = [91.78,91.77,91.65,91.38,91.10]
# plt.title('Acc as edge ratio increasing',fontsize=15)
# my_x_ticks = np.arange(30, 71, 10)
# plt.xticks(my_x_ticks)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('edge ratio',fontsize=15)
# plt.ylabel('acc',fontsize=15)
# plt.plot(x[::-1], y1[::-1], marker='o', markersize=5)
# plt.plot(x[::-1], y2[::-1], marker='x', markersize=5)
# plt.plot(x[::-1], y4[::-1], marker='s', markersize=5)
# plt.plot(x[::-1], y5[::-1], marker='+', markersize=5)
# plt.plot(x[::-1], y6[::-1], marker='D', markersize=5)
# plt.tight_layout()
# plt.legend(['LPGNN-F', 'LPGNN-G', 'LPGNN w/o QA', 'LPGNN-F w/o CL', 'LPGNN-G w/o CL'],loc=4)
# plt.savefig('../fig/gcn_products.pdf', )

# gcn proteins compared dropedge and QUEEN-GFR
# x = [70,60,50,40,30]
# y1 = [87.21,87.12,87.04,86.86,86.72]
# y2 = [86.96,86.88,86.88,86.66,86.42]
# y4 = [86.97,86.88,86.72,86.60,86.49]
# y5 = [86.91,86.77,86.78,86.72,86.57]
# y6 = [86.72,86.65,86.64,86.47,86.25]
# plt.title('Acc as edge ratio increasing',fontsize=15)
# my_x_ticks = np.arange(30, 71, 10)
# plt.xticks(my_x_ticks)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('edge ratio',fontsize=15)
# plt.ylabel('acc',fontsize=15)
# plt.plot(x[::-1], y1[::-1], marker='o', markersize=5)
# plt.plot(x[::-1], y2[::-1], marker='x', markersize=5)
# plt.plot(x[::-1], y4[::-1], marker='s', markersize=5)
# plt.plot(x[::-1], y5[::-1], marker='+', markersize=5)
# plt.plot(x[::-1], y6[::-1], marker='D', markersize=5)
# plt.tight_layout()
# plt.legend(['LPGNN-F', 'LPGNN-G', 'LPGNN w/o QA', 'LPGNN-F w/o CL', 'LPGNN-G w/o CL'],loc=4)
# plt.savefig('../fig/gcn_proteins.pdf', )

# gcn amazon compared dropedge and QUEEN-GFR
# x = [70,60,50,40,30]
# y1 = [46.81,46.81,46.81,46.81,46.81]
# y2 = [47.68,47.68,47.07,45.68,42.30]
# y4 = [35.36,32.35,28.45,24.54,19.52]
# y5 = [47.51,48.05,47.44,46.33,45.51]
# y6 = [47.61,46.27,43.45,40.11,36.93]
# plt.title('Acc as edge ratio increasing',fontsize=15)
# my_x_ticks = np.arange(30, 71, 10)
# plt.xticks(my_x_ticks)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('edge ratio',fontsize=15)
# plt.ylabel('acc',fontsize=15)
# plt.plot(x[::-1], y1[::-1], marker='o', markersize=5)
# plt.plot(x[::-1], y2[::-1], marker='x', markersize=5)
# plt.plot(x[::-1], y4[::-1], marker='s', markersize=5)
# plt.plot(x[::-1], y5[::-1], marker='+', markersize=5)
# plt.plot(x[::-1], y6[::-1], marker='D', markersize=5)
# plt.tight_layout()
# plt.legend(['LPGNN-F', 'LPGNN-G', 'LPGNN w/o QA', 'LPGNN-F w/o CL', 'LPGNN-G w/o CL'],loc=4)
# plt.savefig('../fig/gcn_amazon.pdf', )

# sage reddit compared dropedge and QUEEN-GFR
# x = [70,60,50,40,30]
# y1 = [96.34,96.34,96.33,96.29,96.28]
# y2 = [96.17,96.22,96.08,96.04,95.78]
# y4 = [95.86,95.67,95.51,95.47,95.27]
# y5 = [96.35,96.20,96.19,96.19,96.20]
# y6 = [96.04,96.10,95.97,95.90,95.83]
# plt.title('Acc as edge ratio increasing',fontsize=15)
# my_x_ticks = np.arange(30, 71, 10)
# plt.xticks(my_x_ticks)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('edge ratio',fontsize=15)
# plt.ylabel('acc',fontsize=15)
# plt.plot(x[::-1], y1[::-1], marker='o', markersize=5)
# plt.plot(x[::-1], y2[::-1], marker='x', markersize=5)
# plt.plot(x[::-1], y4[::-1], marker='s', markersize=5)
# plt.plot(x[::-1], y5[::-1], marker='+', markersize=5)
# plt.plot(x[::-1], y6[::-1], marker='D', markersize=5)
# plt.tight_layout()
# plt.legend(['LPGNN-F', 'LPGNN-G', 'LPGNN w/o QA', 'LPGNN-F w/o CL', 'LPGNN-G w/o CL'],loc=4)
# plt.savefig('../fig/sage_reddit.pdf', )

# sage products compared dropedge and QUEEN-GFR
# x = [70,60,50,40,30]
# y1 = [92.16,92.16,92.06,92.02,91.91]
# y2 = [91.91,92.01,91.91,91.72,91.40]
# y4 = [91.83,91.84,91.78,91.58,91.31]
# y5 = [92.21,92.04,92.03,92.03,91.86]
# y6 = [92.04,91.94,91.88,91.70,91.48]
# plt.title('Acc as edge ratio increasing',fontsize=15)
# my_x_ticks = np.arange(30, 71, 10)
# plt.xticks(my_x_ticks)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('edge ratio',fontsize=15)
# plt.ylabel('acc',fontsize=15)
# plt.plot(x[::-1], y1[::-1], marker='o', markersize=5)
# plt.plot(x[::-1], y2[::-1], marker='x', markersize=5)
# plt.plot(x[::-1], y4[::-1], marker='s', markersize=5)
# plt.plot(x[::-1], y5[::-1], marker='+', markersize=5)
# plt.plot(x[::-1], y6[::-1], marker='D', markersize=5)
# plt.tight_layout()
# plt.legend(['LPGNN-F', 'LPGNN-G', 'LPGNN w/o QA', 'LPGNN-F w/o CL', 'LPGNN-G w/o CL'],loc=4)
# plt.savefig('../fig/sage_products.pdf', )

# sage proteins compared dropedge and QUEEN-GFR
# x = [70,60,50,40,30]
# y1 = [90.44,90.31,90.19,90.15,89.85]
# y2 = [90.31,90.25,90.10,89.97,89.76]
# y4 = [90.28,90.21,90.01,89.82,89.55]
# y5 = [90.11,89.99,89.98,89.81,89.57]
# y6 = [90.13,89.93,89.78,89.61,89.47]
# plt.title('Acc as edge ratio increasing',fontsize=15)
# my_x_ticks = np.arange(30, 71, 10)
# plt.xticks(my_x_ticks)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('edge ratio',fontsize=15)
# plt.ylabel('acc',fontsize=15)
# plt.plot(x[::-1], y1[::-1], marker='o', markersize=5)
# plt.plot(x[::-1], y2[::-1], marker='x', markersize=5)
# plt.plot(x[::-1], y4[::-1], marker='s', markersize=5)
# plt.plot(x[::-1], y5[::-1], marker='+', markersize=5)
# plt.plot(x[::-1], y6[::-1], marker='D', markersize=5)
# plt.tight_layout()
# plt.legend(['LPGNN-F', 'LPGNN-G', 'LPGNN w/o QA', 'LPGNN-F w/o CL', 'LPGNN-G w/o CL'],loc=4)
# plt.savefig('../fig/sage_proteins.pdf', )

# sage amazon compared dropedge and QUEEN-GFR
x = [70,60,50,40,30]
y1 = [76.24,76.24,76.24,76.24,76.24]
y2 = [76.25,76.25,76.25,76.22,76.25]
y4 = [76.22,76.28,76.19,76.26,76.17]
y5 = [76.11,76.08,76.08,76.06,76.04]
y6 = [76.08,76.05,76.04,76.03,76.03]
plt.title('Acc as edge ratio increasing',fontsize=15)
my_x_ticks = np.arange(30, 71, 10)
plt.xticks(my_x_ticks)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('edge ratio',fontsize=15)
plt.ylabel('acc',fontsize=15)
plt.plot(x[::-1], y1[::-1], marker='o', markersize=5)
plt.plot(x[::-1], y2[::-1], marker='x', markersize=5)
plt.plot(x[::-1], y4[::-1], marker='s', markersize=5)
plt.plot(x[::-1], y5[::-1], marker='+', markersize=5)
plt.plot(x[::-1], y6[::-1], marker='D', markersize=5)
plt.tight_layout()
plt.legend(['LPGNN-F', 'LPGNN-G', 'LPGNN w/o QA', 'LPGNN-F w/o CL', 'LPGNN-G w/o CL'],loc=4)
plt.savefig('../fig/sage_amazon.pdf', )

# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

# name_list = ['Reddit','products','proteins','Amazon']
# # num_list_scene1 = [0.8750, 1.5670, 1.1617, 1.0463]
# # num_list_scene2 =[0.8752, 1.9769, 1.1506, 1.1714]
# # num_list_scene3 = [2.8193, 3.4410, 2.2677, 5.8413] # 3.21, 2.20, 1.95, 5.58
# # num_list_scene4 = [51.3030, 58.4994, 34.2332, 138.1929] # 37.33, 29.47, 132.08
# # labels = ['Reddit','products','proteins','Amazon']
# num_list_scene1 = [0.8713, 1.5217, 1.1920, 1.2643]
# num_list_scene2 = [0.8714, 1.9974, 1.1560, 1.1714]
# num_list_scene3 = [2.7955, 3.3873, 2.6457, 7.1480] # 3.21, 2.23, 2,22, 5.65
# num_list_scene4 = [55.3957, 53.4607, 32.1757, 126.5801] # 26.99, 100.46
# # # data1 = [0.8713, 1.5217, 1.1920, 1.2643]
# # # data2 = [0.8714, 1.9974, 1.1560, 1.1714]
# # # data3 = [2.7955, 3.3873, 2.6457, 7.1480]
# # # data4 = [55.3957, 53.4607, 32.1757, 126.5801]

# a1,a2=2,3
# gs = gridspec.GridSpec(2, 1,height_ratios=[a1,a2],hspace=0.1)
# ax = plt.subplot(gs[0,0:])
# ax2 = plt.subplot(gs[1,0:], sharex=ax)

# x = list(range(len(num_list_scene1)))
# total_width, n = 0.9, 6
# width = total_width / n
# ax.bar(x, num_list_scene1, width=width, label='LPGNN', fc='c')
# for j in range(len(x)):
#     x[j] = x[j] + width
# ax.bar(x, num_list_scene2, width=width, label='LPGNN w/o PEA', fc='#A52A2A')
# for j in range(len(x)):
#     x[j] = x[j] + width
# ax.bar(x, num_list_scene3, width=width, label='DropEdge', fc='b', align = 'center')
# for j in range(len(x)):
#     x[j] = x[j] + width
# ax.bar(x, num_list_scene4, width=width, label='PS', fc='#D2691E')

# x = list(range(len(num_list_scene1)))
# ax2.bar(x, num_list_scene1, width=width, label='LPGNN', fc='c')
# for j in range(len(x)):
#     x[j] = x[j] + width
# ax2.bar(x, num_list_scene2, width=width, label='LPGNN w/o PEA', fc='#A52A2A')
# for j in range(len(x)):
#     x[j] = x[j] + width
# ax2.bar(x, num_list_scene3, width=width, label='DropEdge', fc='b')
# for j in range(len(x)):
#     x[j] = x[j] + width
# ax2.bar(x, num_list_scene4, width=width, label='PS', tick_label=name_list, fc='#D2691E')

# ax.set_ylim(30, 150)  # outliers only
# ax2.set_ylim(0, 10)  # most of the data

# ax.spines['bottom'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax.xaxis.tick_top()
# ax.tick_params(labeltop='off')  # don't put tick labels at the top
# ax2.xaxis.tick_bottom()
# d = .015

# oa1,oa2=(a1+a2)/a1,(a1+a2)/a2
# kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
# ax.plot((-d, +d), (-d*oa1, +d*oa1), **kwargs)        # top-left diagonal
# ax.plot((1 - d, 1 + d), (-d*oa1, +d*oa1), **kwargs)  # top-right diagonal

# kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
# ax2.plot((-d, +d), (1 - d*oa2, 1 + d*oa2), **kwargs)  # bottom-left diagonal
# ax2.plot((1 - d, 1 + d), (1 - d*oa2, 1 + d*oa2), **kwargs)  # bottom-right diagonal

# plt.xlabel('datasets')
# plt.ylabel('time')
# plt.legend(bbox_to_anchor=(0.35,1.75))
# plt.savefig('../fig/time_consume_sage.pdf', dpi=600)
# plt.show()


# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from brokenaxes import brokenaxes
# #生成信息
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 'r', 'b']
# labels = ['Reddit','products','proteins','Amazon']
# data1 = [0.8750, 1.5670, 1.1617, 1.0463]
# data2 = [0.8752, 1.9769, 1.1506, 1.1714]
# data3 = [2.8193, 3.4410, 2.2677, 5.8413]
# data4 = [51.3030, 58.4994, 34.2332, 138.1929]

# # data1 = [0.8750, 0.8752, 2.8193, 51.3030]
# # data2 = [1.5670, 1.9769, 3.4410, 58.4994]
# # data3 = [1.1617, 1.1506, 2.2677, 34.2332]
# # data4 = [1.0463, 1.1714, 5.8413, 138.1929]
# # data1 = [0.8713, 1.5217, 1.1920, 1.2643]
# # data2 = [0.8714, 1.9974, 1.1560, 1.1714]
# # data3 = [2.7955, 3.3873, 2.6457, 7.1480]
# # data4 = [55.3957, 53.4607, 32.1757, 126.5801]
# width = 0.7
# xpos = np.arange(0,20,5)

# #生成柱状图
# fig, ax = plt.subplots(figsize=(10,8))
# bax = brokenaxes(ylims=((0, 60), (130, 140)), hspace=.05, despine=False)
# bars1 = bax.bar(xpos-3/2*width, data1, align='center', width=width, alpha=0.9, label = 'LPGNN')
# bars2 = bax.bar(xpos-width/2, data2, align='center', width=width, alpha=0.9, label = 'LPGNN w/o PEA')
# bars3 = bax.bar(xpos+width/2, data3, align='center', width=width, alpha=0.9, label = 'DropEdge')
# bars4 = bax.bar(xpos+3/2*width, data4, align='center', width=width, alpha=0.9, label = 'PS')

# bax.axs[1].get_yaxis().get_offset_text().set_visible(False)

# #设置每个柱子下面的记号
# ax.set_xticks(xpos) #确定每个记号的位置
# ax.set_xticklabels(labels)  #确定每个记号的内容

# bax.set_xlabel('datasets')
# bax.set_ylabel('time')

# #给每个柱子上面添加标注
# # def autolabel(rects):
# #     """Attach a text label above each bar in *rects*, displaying its height."""
# #     for rect in rects:
# #         height = rect.get_height()
# #         ax.annotate('{}'.format(height),
# #               xy=(rect.get_x() + rect.get_width() / 2, height),
# #               xytext=(0, 3),  # 3 points vertical offset
# #               textcoords="offset points",
# #               ha='center', va='bottom'
# #               )
# # autolabel(bars1)
# # autolabel(bars2)
# # autolabel(bars3)
# # autolabel(bars4)
# bax.legend()
# fig.savefig('../fig/time_consume_gcn.pdf', dpi=600)
# plt.show()
