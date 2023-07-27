import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dgl.nn as dglnn
import dgl
from dgl import AddSelfLoop
from dgl import DropEdge
from dgl import add_reverse_edges, add_edges, to_simple
from dgl import graph, NID, EID, DGLGraph
from dgl.data import FlickrDataset, PPIDataset, CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset, RedditDataset, YelpDataset
from dgl.random import choice
import GPUtil
import time
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from dgl.dataloading import SAINTSampler, GraphDataLoader, ClusterGCNSampler, NeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from dgl import sampling
import dgl.function as fn
import random
import math
from copy import deepcopy
import scipy.sparse as sp
import numpy as np
import os
import sys
from GPU_Memory import get_gpu_process_info

from amazon import load_data
from yelp import load_yelp

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from torchmetrics.classification import MultilabelF1Score
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
import requests
import pandas


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu, allow_zero_in_degree=True)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        
        return h


class GNNModel(nn.Module):
    def __init__(self,  gnn_layer: str, n_layers: int, layer_dim: int,
                 input_feature_dim: int, n_classes: int, n_linear: int):
        super().__init__()

        assert n_layers >= 1, 'GNN must have at least one layer'
        dims = [input_feature_dim] + [layer_dim] * (n_layers-1) + [n_classes]
        print(dims)
        
        self.convs = nn.ModuleList()
        self.norm = nn.ModuleList()
        
        self.n_layers = n_layers
        self.n_linear = n_linear
        
        for idx in range(n_layers):
            if idx < n_layers - n_linear:
                if gnn_layer == 'gat':
                    # use 2 aattention heads
                    # layer = dglnn.GATConv(dims[idx], dims[idx+1], 1)  # pylint: disable=no-member
                    layer = dglnn.AGNNConv(learn_beta=False, allow_zero_in_degree=True)
                elif gnn_layer == 'gcn':
                    layer = dglnn.GraphConv(dims[idx], dims[idx+1], allow_zero_in_degree=True)  # pylint: disable=no-member
                elif gnn_layer == 'sage':
                    # Use mean aggregtion
                    # pylint: disable=no-member
                    layer = dglnn.SAGEConv(dims[idx], dims[idx+1],
                                            aggregator_type='mean')
                else:
                    raise ValueError(f'unknown gnn layer type {gnn_layer}')
                self.convs.append(layer)
            else: 
                self.convs.append(nn.Linear(dims[idx], dims[idx+1]))
                
            if idx < n_layers - 1:
                self.norm.append(nn.LayerNorm(dims[idx+1], elementwise_affine=True))
                
            
    def forward(self, graph, features):
        h = features
        for idx in range(self.n_layers):
            
            h = F.dropout(h, p=0.1)
            if idx < self.n_layers - self.n_linear:
                h = self.convs[idx](graph, h)
                if h.ndim == 3:  # GAT produces an extra n_heads dimension
                    h = h.mean(1)
            else:
                h = self.convs[idx](h)

            if idx < self.n_layers - 1:
                h = self.norm[idx](h)
                h = F.relu(h, inplace=True)
            
        return h
            
            
class PreModel:
    
    def __init__(self, model, device, args):
        self.device = device
        self.most_GPU_memory = args.memory
        self.best_graph_val_acc = 0
        self.best_graph_val_loss = 700
        self.best_val_acc = 0
        self.best_val_loss = 100
        self.weights = None
        self.estimator = None
        self.best_model = None
        self.model = model.to(device)
        # TODO: update a full-graph with less edges and add edges by potential graph training
        self.best_graph = None
        self.whether_train_graph = True
        self.whether_explain = False
        self.alpha1 = 0.000000005
        self.alpha2 = 1.0
        self.pre_logits = None
        self.pre_dict = deepcopy(self.model.state_dict())
        self.hdiff = []
        self.explain_threshold = args.th
        self.peak_memory = 0
        self.num_edges = args.edges
        self.epoch_record = 0
        self.prob = args.prob
        self.current_edges = 0
        self.drop = args.drop
        self.dataset = args.data
        if self.dataset == 'cora':
            self.first_size = 1000
            self.second_size = 35
        elif self.dataset == 'citeseer':
            self.first_size = 1000
            self.second_size = 30
        elif self.dataset == 'pubmed':
            self.first_size = 5000
            self.second_size = 200
        elif self.dataset == 'reddit' or self.dataset == 'amazon' or self.dataset == 'proteins':
            self.first_size = 500000
            self.second_size = 100000
        elif self.dataset == 'products':
            self.first_size = 1000000
            self.second_size = 100000
    
    def find_graph(self, g):
        self.input_size = g.ndata['feat'].size(1)
        # self.explain_threshold = torch.sum(g.ndata['train_mask'] == 1) * 0.12
        print("explain threshold: ", self.explain_threshold)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        
        # TODO: adding sampling graph
        # GraphSAINT sampling
        # sampler = SAINTSampler(mode='edge', budget=6000)
        # loader = DataLoader(g, torch.arange(1000), sampler)
        
        # print("get subgraph!")
            
        dropedge = DropEdge(p=self.drop)

        start = time.time()
        total_aug_time = 0
        each_iter_time = []
        # with torch.no_grad():
        #     g_s = g_s.to(self.device)
        #     self.pre_dict = deepcopy(self.model.state_dict())

        for epoch in range(20): 
        
            t1 = time.time() 
            # dropedge
            g_s = deepcopy(g)
            g_s = dropedge(g_s)
            g_s = g_s.to(self.device)
            t2 = time.time()   

            self.train_gcn(g_s, epoch)
            
            each_iter_time.append(t2-t1)
            print("each iteration time: ", t2 - t1)

            total_aug_time += t2 - t1
            
        #     # num_add_edges = g_s.edges()[0].size()[0] - self.current_edges
        #     # self.current_edges = g_s.edges()[0].size()[0]
        #     # with open("record.txt", "a") as f:
        #     #     f.write(str(epoch) + " " + str(t2 - t1) + " " + str(num_add_edges) + '\n')
        
        #     print("one epoch time: ", t2 - t1)
        
        end = time.time()
        print("time: ", end - start, total_aug_time)        
        print("best val acc: ", self.best_val_acc)

        print("avg iter time: ", sum(each_iter_time)/len(each_iter_time))

        with open("result.txt", "a") as f:
            f.write("gcn " + str(epoch) + " time:" + str(end - start) + " acc:" + str(self.best_val_acc) + " avg time: " + str(sum(each_iter_time)/len(each_iter_time)) + '\n')
        
        return self.best_graph
    
            
    def train_gcn(self, g, epoch):
        features = g.ndata['feat']
        train_idx = g.ndata['train_mask']
        labels = g.ndata['label']
        val_idx = g.ndata['val_mask']
        
        print("num of graph edges", g.edges()[0].size()[0])
               
        self.model.train()
        logits = self.model(g, features)

        if labels.dim() == 1:
            train_loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        else:
            train_loss = F.binary_cross_entropy_with_logits(logits[train_idx], labels[train_idx], reduction='mean')
        
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        
        # evaluate gcn
        # evaluator = Evaluator(name='ogbn-proteins')
        # print(labels.size())
        # acc = evaluator.eval({'y_true': labels[val_idx], 
        #                       'y_pred': logits[val_idx],
        #                       })['rocauc']
        acc = self.evaluate(logits, labels, val_idx)
        
        GPUs = GPUtil.getGPUs()
        if GPUs[0].memoryUsed > self.most_GPU_memory:
            self.epoch_record = epoch
            self.whether_train_graph = False
            self.whether_explain = False
            self.peak_memory = GPUs[0].memoryUsed
        
        # print("current process memory used: ", get_gpu_process_info(pid=str(os.getpid())))
        print("gcn training epoch{} train loss :{}, acc: {}, best acc:{}, memory: {}".format(epoch, train_loss, acc, self.best_val_acc, GPUs[0].memoryUsed))

        if acc > self.best_val_acc:
            self.best_val_acc = acc

        if self.whether_explain == True:
            with torch.no_grad():
                pre_logits = self.pre_logits
                if labels.dim() == 1:
                    self.train_explain(pre_logits, logits.cpu(), train_idx, epoch, g.num_edges(), GPUs[0].memoryUsed, acc)
                else:
                    self.train_explain_multi(pre_logits, logits.cpu(), train_idx, epoch, g.num_edges(), GPUs[0].memoryUsed, acc)
                del logits, train_loss, acc
        else:
            del logits, train_loss, acc

        torch.cuda.empty_cache()
        
         
    def evaluate(self, logits, labels, val_idx):
        model.eval()
        with torch.no_grad():
            val_logits = logits[val_idx]
            val_labels = labels[val_idx]
            val_acc = self.calc_acc(val_logits, val_labels)
        return val_acc
    
    
    def calc_acc(self, logits, labels):
        if labels.dim() == 1:
            print("single label")
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() / labels.shape[0]
        else:
            print("multi label")
            logits[logits > 0] = 1
            logits[logits <= 0] = 0
            print(logits.size())
            # logits = torch.round(torch.sigmoid(logits))
            return f1_score(labels.cpu(), logits.cpu(), average='micro')



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--prob', type=str, default=None)
    parser.add_argument('--memory', type=int, default=99999)
    parser.add_argument('--th', type=float, default=99999)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--edges', type=int, default=999999999)
    parser.add_argument('--size', type=int, default=999999999)
    parser.add_argument('--drop', type=float, default=1)
    args = parser.parse_args()
    print(args.prob, args.memory, args.th, args.data, args.edges, args.drop, args.size)

    if args.data == 'cora':
        transform = (AddSelfLoop()) 
        data = CoraGraphDataset()
        g = data[0]
        g = transform(g)
        num_class = data.num_classes

        num_node = g.num_nodes()
        
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
        g.ndata['label'] = g.ndata['label']
        g.ndata['train_mask'] = g.ndata['train_mask'].bool()
        g.ndata['val_mask'] = g.ndata['val_mask'].bool()
        g.ndata['test_mask'] = g.ndata['test_mask'].bool()  

    if args.data == 'citeseer':
        transform = (AddSelfLoop()) 
        data = CiteseerGraphDataset()
        g = data[0]
        g = transform(g)
        num_class = data.num_classes

        num_node = g.num_nodes()
        
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
        g.ndata['label'] = g.ndata['label']
        g.ndata['train_mask'] = g.ndata['train_mask'].bool()
        g.ndata['val_mask'] = g.ndata['val_mask'].bool()
        g.ndata['test_mask'] = g.ndata['test_mask'].bool()  

    if args.data == 'pubmed':
        transform = (AddSelfLoop()) 
        data = PubmedGraphDataset()
        g = data[0]
        g = transform(g)
        num_class = data.num_classes

        num_node = g.num_nodes()
        
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
        g.ndata['label'] = g.ndata['label']
        g.ndata['train_mask'] = g.ndata['train_mask'].bool()
        g.ndata['val_mask'] = g.ndata['val_mask'].bool()
        g.ndata['test_mask'] = g.ndata['test_mask'].bool()  
        
    if args.data == 'reddit':
        # load and preprocess Reddit dataset 32
        transform = (AddSelfLoop()) 
        data = RedditDataset()
        
        g = data[0]
        g = transform(g)
        num_class = data.num_classes
        num_node = g.num_nodes()
        g = g.int()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        g.ndata['label'] = g.ndata['label']
        g.ndata['train_mask'] = g.ndata['train_mask'].bool()
        g.ndata['val_mask'] = g.ndata['val_mask'].bool()
        g.ndata['test_mask'] = g.ndata['test_mask'].bool()  
        feats = g.ndata['feat']
        scaler = StandardScaler()
        scaler.fit(feats[g.ndata['train_mask']])
        feats = scaler.transform(feats)
        g.ndata['feat'] = torch.tensor(feats, dtype=torch.float)

    if args.data == 'amazon':
        # load and preprocess Amazon dataset 32
        transform = (AddSelfLoop()) 
        data = load_data('amazon', multilabel=True)
        
        g = data[2]
        g = transform(g)
        num_class = data[0]
        num_node = g.num_nodes()
        g = g.int()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        g.ndata['label'] = g.ndata['label']
        g.ndata['train_mask'] = g.ndata['train_mask'].bool()
        g.ndata['val_mask'] = g.ndata['val_mask'].bool()
        g.ndata['test_mask'] = g.ndata['test_mask'].bool()  
        feats = g.ndata['feat']
        scaler = StandardScaler()
        scaler.fit(feats[g.ndata['train_mask']])
        feats = scaler.transform(feats)
        g.ndata['feat'] = torch.tensor(feats, dtype=torch.float)
    
    if args.data == 'products':
        # load and preprocess products dataset 64
        transform = (AddSelfLoop()) 
        data = DglNodePropPredDataset(name='ogbn-products')
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )
        g, labels = data[0]
        g = transform(g)
        nfeat = g.ndata.pop("feat")
        labels = labels[:, 0]

        g.ndata['feat'] = nfeat
        g.ndata['label'] = labels
        train_mask = torch.zeros(g.num_nodes())
        train_mask[train_idx] = 1
        val_mask = torch.zeros(g.num_nodes())
        val_mask[val_idx] = 1
        g.ndata['train_mask'] = train_mask.byte()
        g.ndata['val_mask'] = val_mask.byte()
        
        num_class = (labels.max() + 1).item()
        print("train nodes: ", torch.sum(g.ndata['train_mask'] == 1))
    
    if args.data == 'papers100M':
        # load and preprocess products dataset
        transform = (AddSelfLoop()) 
        data = DglNodePropPredDataset(name='ogbn-papers100M')
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # labels = labels.view(-1).type(torch.int)
        
        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )
        g, labels = data[0]
        nfeat = g.ndata.pop("feat")
        labels = labels[:, 0]

        g.ndata['feat'] = nfeat
        g.ndata['label'] = labels
        train_mask = torch.zeros(g.num_nodes())
        train_mask[train_idx] = 1
        val_mask = torch.zeros(g.num_nodes())
        val_mask[val_idx] = 1
        g.ndata['train_mask'] = train_mask.byte()
        g.ndata['val_mask'] = val_mask.byte()
        
        num_class = 172
        print("train nodes: ", torch.sum(g.ndata['train_mask'] == 1))
    
    if args.data == 'proteins':  
        # load and preprocess proteins dataset 64
        transform = (AddSelfLoop()) 
        data = DglNodePropPredDataset(name='ogbn-proteins')
        
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
        # ogb dataset
        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )
        g = data.graph[0]
        g.ndata["label"] = data.labels.float()
        g.edata["feat"] = g.edata["feat"].float()

        g.update_all(fn.copy_e('feat', 'm'), fn.sum('m', 'feat'))
        print(g.ndata["feat"].size())
        train_mask = torch.zeros(g.num_nodes())
        train_mask[train_idx] = 1
        val_mask = torch.zeros(g.num_nodes())
        val_mask[val_idx] = 1
        g.ndata['train_mask'] = train_mask.byte()
        g.ndata['val_mask'] = val_mask.byte()

        num_class = 112
        print("train nodes: ", torch.sum(g.ndata['train_mask'] == 1))

        
    model = GNNModel('sage', 4, 256, g.ndata['feat'].size(1), num_class, 0).to(device)
    pre_model = PreModel(model, device, args)
    good_edges = pre_model.find_graph(g)
