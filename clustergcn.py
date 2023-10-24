import time

import dgl
import dgl.nn as dglnn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset
from dgl.data import RedditDataset
import GPUtil


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.GraphConv(in_size, hid_size, activation=F.relu, allow_zero_in_degree=True))
        self.layers.append(dglnn.GraphConv(hid_size, hid_size, activation=F.relu, allow_zero_in_degree=True))
        self.layers.append(dglnn.GraphConv(hid_size, out_size, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(0.5)

    def forward(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h
    

dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
# dataset = AsNodePredDataset(RedditDataset())
graph = dataset[
    0
]  # already prepares ndata['label'/'train_mask'/'val_mask'/'test_mask']

model = GCN(graph.ndata["feat"].shape[1], 128, dataset.num_classes).to("cuda:1")
opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=5e-4)

num_partitions = 1000
sampler = dgl.dataloading.ClusterGCNSampler(
    graph,
    num_partitions,
    cache_path='cluster_gcn_products.pkl',
    prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
)
# DataLoader for generic dataloading with a graph, a set of indices (any indices, like
# partition IDs here), and a graph sampler.
dataloader = dgl.dataloading.DataLoader(
    graph,
    torch.arange(num_partitions).to("cuda:1"),
    sampler,
    device="cuda:1",
    batch_size=100,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    use_uva=True,
)

durations = []
best_val_acc = 0
peak_memory = 0
for epoch in range(100):
    t0 = time.time()
    model.train()
    for it, sg in enumerate(dataloader):
        x = sg.ndata["feat"]
        y = sg.ndata["label"]
        m = sg.ndata["train_mask"].bool()
        y_hat = model(sg, x)
        loss = F.cross_entropy(y_hat[m], y[m])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 20 == 0:
            acc = MF.accuracy(
                y_hat[m],
                y[m],
                task="multiclass",
                num_classes=dataset.num_classes,
            )
            GPUs = GPUtil.getGPUs()
            if GPUs[1].memoryUsed > peak_memory:
                peak_memory = GPUs[1].memoryUsed
            print("Loss", loss.item(), "Acc", acc.item(), "GPU Mem", peak_memory, "MB")

    tt = time.time() - t0
    print("Run time for epoch# %d: %.2fs" % (epoch, tt))
    durations.append(tt)

    model.eval()
    with torch.no_grad():
        val_preds, test_preds = [], []
        val_labels, test_labels = [], []
        for it, sg in enumerate(dataloader):
            x = sg.ndata["feat"]
            y = sg.ndata["label"]
            m_val = sg.ndata["val_mask"].bool()
            m_test = sg.ndata["test_mask"].bool()
            y_hat = model(sg, x)
            val_preds.append(y_hat[m_val])
            val_labels.append(y[m_val])
            test_preds.append(y_hat[m_test])
            test_labels.append(y[m_test])
        val_preds = torch.cat(val_preds, 0)
        val_labels = torch.cat(val_labels, 0)
        test_preds = torch.cat(test_preds, 0)
        test_labels = torch.cat(test_labels, 0)
        val_acc = MF.accuracy(
            val_preds,
            val_labels,
            task="multiclass",
            num_classes=dataset.num_classes,
        )
        if val_acc.item() > best_val_acc:
            best_val_acc = val_acc.item()
        print("Validation acc:", val_acc.item())

print(
    "Average run time for last %d epochs: %.2fs standard deviation: %.3f"
    % ((epoch - 3), np.mean(durations[4:]), np.std(durations[4:]))
)
print("Best validation acc:", best_val_acc)