import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dgl.nn as dglnn
import dgl
from dgl import AddSelfLoop
from dgl import DropEdge
from dgl import add_reverse_edges, add_edges, to_simple
from dgl import graph, NID, EID
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset, RedditDataset, YelpDataset
from dgl.random import choice
import GPUtil
import time
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from dgl.dataloading import SAINTSampler, DataLoader, ClusterGCNSampler, NeighborSampler
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


class GraphSAGE_model(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE_model, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=0., activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
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
            
            h = F.dropout(h, p=0.5)
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
            self.first_size = 500000
            self.second_size = 100000

    
    def find_graph(self, g):
        self.input_size = g.ndata['feat'].size(1)
        # self.explain_threshold = torch.sum(g.ndata['train_mask'] == 1) * 0.12
        print("explain threshold: ", self.explain_threshold)
        
        # A GNN to estimate features
        # TODO: estimator 
        self.estimator = EstimateGraph().to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
            
        dropedge = DropEdge(p=self.drop)
        g_s = deepcopy(g)
        g_s = dropedge(g_s)

        # src, dst= g.edges()
        # if self.prob == 'sage':
        #     print("ours prob")
        #     prob = g.ndata['p_norm'][dst.long()]
        # elif self.prob == 'saint':
        #     print("saints prob")
        #     prob = g.edata['p_norm']
        # selected_edges = torch.unique(choice(g.num_edges(), size=self.num_edges, prob=prob, replace=False))
        # add_nodes0, add_nodes1 = torch.cat([src[selected_edges]]), torch.cat([dst[selected_edges]])
        # g_s = g_s.to('cpu')
        # add_nodes0, add_nodes1 = self.simple_edges(g_s, add_nodes0.type(torch.int32), add_nodes1.type(torch.int32))
        # g_s = add_edges(g_s, add_nodes0, add_nodes1)        
        g_s = g_s.to(self.device)
        
        start = time.time()
        total_aug_time = 0
        each_iter_time = []
        # with torch.no_grad():
        #     g_s = g_s.to(self.device)
        #     self.pre_dict = deepcopy(self.model.state_dict())

        for epoch in range(3000): 
        
            t1 = time.time() 
            # dropedge
            # g_s = deepcopy(g)
            # g_s = dropedge(g_s)
            # g_s = g_s.to(self.device)
            t2 = time.time()   

            # ours
            if self.whether_train_graph == True:
                
                t3 = time.time()

                if self.whether_explain == True:
                    with torch.no_grad():
                        features = g_s.ndata['feat']
                        self.pre_logits = self.model(g_s, features).to('cpu')

                # TODO: potential graph shoud be consistant to sampled subset of nodes
                src, dst= g.edges()
                sampled_edges = torch.unique(choice(g.num_edges(), size=self.first_size, prob=None))
                # print(sampled_edges.size())
                if self.prob == 'gcn' or self.prob == 'sage':
                    print("ours prob")
                    prob = g.ndata['p_norm'][dst[sampled_edges].long()]
                    selected_edges = torch.unique(choice(len(sampled_edges), size=self.second_size, prob=prob))
                elif self.prob == 'saint':
                    print("saints prob")
                    prob = g.edata['p_norm'][sampled_edges]
                    selected_edges = torch.unique(choice(len(sampled_edges), size=self.second_size, prob=prob))
                else:
                    print("no prob")
                    selected_edges = torch.unique(choice(len(sampled_edges), size=self.second_size, prob=None))
                selected_edges = sampled_edges[selected_edges]
                add_nodes0, add_nodes1 = torch.cat([src[selected_edges]]), torch.cat([dst[selected_edges]])

                
                # remove redundant added edges
                g_s = g_s.to('cpu')
                add_nodes0, add_nodes1 = self.simple_edges(g_s, add_nodes0.type(torch.int32), add_nodes1.type(torch.int32))

                g_s = add_edges(g_s, add_nodes0, add_nodes1)
                g_s = g_s.to(self.device)

                t4 = time.time()
                each_iter_time.append(t4-t3)
                print("each iteration time: ", t4 - t3)

            self.train_gcn(g_s, epoch)
            
            total_aug_time += t4 - t3
            
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
            f.write(" gcn " + str(epoch) + " time:" + str(end - start) + " acc:" + str(self.best_val_acc) + '\n')
        
        return self.best_graph

                
        
    def train_graph(self, subg, epoch):
        self.estimator(subg)
        weights = self.estimator.estimated_weights
        num_samples = 200000
        selected_edges = torch.utils.data.WeightedRandomSampler(weights.tolist(), num_samples, replacement=False)

        add_edges = list(selected_edges)
        add_index0 = self.estimator.potential_graph.edges()[0][add_edges]
        add_index1 = self.estimator.potential_graph.edges()[1][add_edges]
        
        return subg.ndata[NID][add_index0.long()], subg.ndata[NID][add_index1.long()]

            
    def simple_edges(self, g, index0, index1):
        mask = g.has_edges_between(index0, index1)
        index0 = index0[~mask]
        index1 = index1[~mask]
        return index0, index1
                
            
    def train_gcn(self, g, epoch):
        features = g.ndata['feat']
        train_idx = g.ndata['train_mask']
        labels = g.ndata['label']
        val_idx = g.ndata['val_mask']
        test_idx = g.ndata['test_mask']
        
        print("num of graph edges", g.edges()[0].size()[0])
               
        self.model.train()
        logits = self.model(g, features)

        if labels.dim() == 1:
            train_loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        else:
            train_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='sum')
        
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
        if GPUs[1].memoryUsed > self.most_GPU_memory:
            self.epoch_record = epoch
            self.whether_train_graph = False
            self.whether_explain = False
            self.peak_memory = GPUs[1].memoryUsed
        
        # print("current process memory used: ", get_gpu_process_info(pid=str(os.getpid())))
        print("gcn training epoch{} train loss :{}, acc: {}, best acc:{}, memory: {}".format(epoch, train_loss, acc, self.best_val_acc, GPUs[1].memoryUsed))
        with open("record.txt", "a") as f:
            f.write(str(epoch) + " " + str(train_loss.item()) + " " + str(acc) + '\n')
        

        if acc > self.best_val_acc:
            self.best_val_acc = acc

        if self.whether_explain == True:
            with torch.no_grad():
                pre_logits = self.pre_logits
                self.train_explain(pre_logits, logits.cpu(), train_idx, epoch, g.num_edges(), GPUs[1].memoryUsed, acc)
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
            return f1_score(labels.cpu(), logits.cpu(), average='micro')
        
    
    def train_explain(self, pre_logits, logits, train_idx, epoch, num_edges, memory, acc):
        logits = F.one_hot(logits.argmax(dim=-1), num_classes = 47)
        pre_logits = F.one_hot(pre_logits.argmax(dim=-1), num_classes = 47)
        logits = np.reshape(logits, -1)
        pre_logits = np.reshape(pre_logits, -1)
        loss = mutual_info_score(pre_logits, logits)
        # pre_log_probs = pre_logits.log_softmax(dim=-1)
        # loss = torch.sum(-pre_log_probs[train_idx, label[train_idx]])
        print("graph difference: ", loss)
        self.hdiff.append(loss.item())
        with open("record0.txt", "a") as f:
            f.write(str(epoch) + " " + str(loss.item()) + " " + str(num_edges) + " " + str(memory) + " " + str(acc) + " " + str(self.explain_threshold)+ '\n')
        if np.percentile(self.hdiff, 75) >= self.explain_threshold and epoch > 100:
            self.whether_train_graph = False
            self.whether_explain = False
            GPUs = GPUtil.getGPUs()
            self.peak_memory = GPUs[1].memoryUsed
        del logits, pre_logits, loss
        torch.cuda.empty_cache()


    def train_explain_multi(self, pre_logits, logits, train_idx, epoch, num_edges, memory, acc):
        logits = logits[train_idx.cpu()]
        pre_logits = pre_logits[train_idx.cpu()]
        logits[logits > 0] = 1
        logits[logits <= 0] = 0
        pre_logits[pre_logits > 0] = 1
        pre_logits[pre_logits <= 0] = 0
        logits = np.reshape(logits, -1)
        pre_logits = np.reshape(pre_logits, -1)
        loss = mutual_info_score(pre_logits, logits)
        print("graph difference: ", loss)
        self.hdiff.append(loss)
        with open("record.txt", "a") as f:
            f.write(str(epoch) + " " + str(loss) + " " + str(num_edges) + " " + str(memory) + " " + str(acc) + " " + str(self.explain_threshold)+ '\n')
        if np.percentile(self.hdiff, 75) >= self.explain_threshold and epoch > 100:
            self.whether_train_graph = False
            self.whether_explain = False
            GPUs = GPUtil.getGPUs()
            self.peak_memory = GPUs[1].memoryUsed
        del logits, pre_logits, loss
        torch.cuda.empty_cache()
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--prob', type=str, default='ours')
    parser.add_argument('--memory', type=int, default=99999)
    parser.add_argument('--th', type=float, default=99999)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--edges', type=int, default=999999999)
    parser.add_argument('--drop', type=float, default=1)
    args = parser.parse_args()
    print(args.prob, args.memory, args.th, args.data, args.edges, args.drop)

    if args.data == 'cora':
        transform = (AddSelfLoop()) 
        data = CoraGraphDataset()
        g = data[0]
        g = transform(g)
        num_class = data.num_classes

        num_node = g.num_nodes()
        
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        g = g.int()
        
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
        g = g.int()
        
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
        g = g.int()
        
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
        
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        g = g.int()
        
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
        
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        g = g.int()
        
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
        
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

    if args.prob == 'sage':
        # for graphsage
        print("cal probability")
        d1 = torch.pow(g.out_degrees(), -1.0)
        p_norm = []
        for i in g.nodes():
            neighbors = g.out_edges(i)[1].long()
            d2 = d1[neighbors]
            norm_2 = torch.norm(d2, p=2)
            p_norm.append(norm_2)
        g.ndata['p_norm'] = torch.tensor(p_norm)

    if args.prob == 'saint':
        # for graphsaint's prob
        print("cal probability graphsaint")
        src, dst = g.edges()
        in_deg = g.in_degrees().float().clamp(min=1)
        out_deg = g.out_degrees().float().clamp(min=1)
        # We can reduce the sample space by half if graphs are always symmetric.
        prob = 1.0 / in_deg[dst.long()] + 1.0 / out_deg[src.long()]
        prob /= prob.sum()
        g.edata['p_norm'] = prob

    if args.data == 'cora' or args.data == 'citeseet' or args.data == 'pubmed':
        learning_rate = 0.01
        model = GCN(g.ndata['feat'].size(1), 128, num_class)
    elif args.data == 'reddit':
        learning_rate = 0.01
        model = GNNModel('sage', 4, 256, g.ndata['feat'].size(1), num_class, 0).to(device)
    elif args.data == 'amazon':
        learning_rate = 0.01
        model = GNNModel('sage', 3, 128, g.ndata['feat'].size(1), num_class, 0).to(device)
    elif args.data == 'products':
        learning_rate = 0.003
        model = GNNModel('sage', 3, 128, g.ndata['feat'].size(1), num_class, 0).to(device)
    elif args.data == 'proteins':
        learning_rate = 0.01
        model = GNNModel('sage', 3, 256, g.ndata['feat'].size(1), num_class, 0).to(device)
        
    pre_model = PreModel(model, device, args, learning_rate)
    good_edges = pre_model.find_graph(g)
