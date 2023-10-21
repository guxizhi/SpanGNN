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

from utils.data import load_data
from Models import GraphSAGE, GNNModel

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from torchmetrics.classification import MultilabelF1Score
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
import requests
import pandas

            
class PreModel:
    
    def __init__(self, model, device, arg):
        self.device = device
        self.most_GPU_memory = args.memory
        self.best_val_acc = 0
        self.best_val_loss = 100
        self.weights = None
        self.estimator = None
        self.best_model = None
        self.model = model.to(device)
        self.best_graph = None
        self.whether_train_graph = True
        if args.explain == 'Y':
            self.whether_explain = True
        else: 
            self.whether_explain = True
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
        if self.dataset == 'reddit' or self.dataset == 'amazon' or self.dataset == 'proteins':
            self.learning = 0.01
        elif self.dataset == 'products':
            self.learning = 0.003

    
    def find_graph(self, g):
        self.input_size = g.ndata['feat'].size(1)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
            
        sampler = NeighborSampler([10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
        )
        train_dataloader = DataLoader(
            g,
            train_idx,
            sampler,
            device=device,
            batch_size=1024,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
                
        start = time.time()

        for epoch in range(10): 

            for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
                
                t1 = time.time()
                self.train_gcn(blocks, epoch)
                t2 = time.time()
            
            total_train_time += t2 - t1
        
        end = time.time()
        print("time: ", end - start, total_train_time)        
        print("best val acc: ", self.best_val_acc)

        with open("result.txt", "a") as f:
            f.write("SAGE " + str(epoch) + " time:" + str(end - start) + " aug time: " + str(total_train_time) + " precision:" + str(self.best_val_acc) + '\n')
        
        return self.best_graph
                
            
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
        if self.dataset == 'proteins':
            evaluator = Evaluator(name='ogbn-proteins')
            print(labels.size())
            acc = evaluator.eval({'y_true': labels[val_idx], 
                                'y_pred': logits[val_idx],
                                })['rocauc']
        else:
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
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    args = parser.parse_args()
    print(args.data)

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
        
    pre_model = PreModel(model, device, args)
    good_edges = pre_model.find_graph(g)
