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
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import dgl.function as fn
import random
import math
from copy import deepcopy
import scipy.sparse as sp
import numpy as np
import os
import sys

from utils.data import load_data
from Models import GCN, GNNModel

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.metrics import mutual_info_score
import requests
import pandas
            
            
class PreModel:
    
    def __init__(self, model, device, args, learning_rate):
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
            self.whether_explain = False
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
        self.edge_per = args.edge_per
        self.learning_rate = learning_rate
        self.name = args.model
        self.peak_edges = 99999999
        self.edge_weight = None
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
            self.first_size = args.first_size
            self.second_size = args.second_size
        elif self.dataset == 'products':
            self.first_size = 500000
            self.second_size = 100000

    
    def find_graph(self, g):
        self.input_size = g.ndata['feat'].size(1)
        self.orignal_edges = g.number_of_edges()

        print("explain threshold: ", self.explain_threshold)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=5e-4)
            
        dropedge = DropEdge(p=self.drop)
        g_s = deepcopy(g)
        g_s = dropedge(g_s)

        # for subgraph trianing with probability
        src, dst= g.edges()
        if self.prob == 'gradient':
            print("ours prob")
            prob = g.ndata['p_norm'][dst.long()]
            selected_edges = torch.unique(choice(g.num_edges(), size=g.num_edges() * self.edge_per, prob=prob, replace=False))
        elif self.prob == 'feature':
            print("saints prob")
            prob = g.edata['p_norm']
            selected_edges = torch.unique(choice(g.num_edges(), size=g.num_edges() * self.edge_per, prob=prob, replace=False))
        else:
            print("random")
            selected_edges = torch.unique(choice(g.num_edges(), size=g.num_edges() * self.edge_per, prob=None, replace=False))
        add_nodes0, add_nodes1 = torch.cat([src[selected_edges]]), torch.cat([dst[selected_edges]])
        g_s = g_s.to('cpu')
        g_s = add_edges(g_s, add_nodes0, add_nodes1)      
        self.edge_weight = torch.ones(g_s.number_of_edges())
        
        print("initialized")

        self.peak_edges = g.num_edges() * self.edge_per
              
        g_s = g_s.to(self.device)
        
        if self.whether_explain == True:
            with torch.no_grad():
                g_s = g_s.to(self.device)
                self.pre_dict = deepcopy(self.model.state_dict())
                
        start = time.time()
        total_aug_time = 0
        each_iter_time = []

        for epoch in range(3000): 

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
                if self.prob == 'gradient':
                    print("ours prob")
                    prob = g.ndata['p_norm'][dst[sampled_edges].long()]
                    selected_edges = torch.unique(choice(len(sampled_edges), size=self.second_size, prob=prob))
                elif self.prob == 'feature':
                    print("saints prob")
                    prob = g.edata['p_norm'][sampled_edges]
                    selected_edges = torch.unique(choice(len(sampled_edges), size=self.second_size, prob=prob))
                else:
                    print("random")
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
        
        end = time.time()
        print("time: ", end - start, total_aug_time)        
        print("best val acc: ", self.best_val_acc)

        # print("avg iter time: ", sum(each_iter_time)/len(each_iter_time))

        with open("result.txt", "a") as f:
            f.write("amazon " + str(self.name) + " " + str(self.prob) + " " + str(epoch) + " time:" + str(end - start) + " aug time: " + str(total_aug_time) + " precision:" + str(self.best_val_acc) + " peak memory:" + str(self.peak_memory) + '\n')
        
        return self.best_graph

            
    def simple_edges(self, g, index0, index1):
        mask = g.has_edges_between(index0, index1)
        # edge_ids = g.edge_ids(index0[mask], index1[mask])
        # self.edge_weight[edge_ids] += 1
        # self.edge_weight = torch.concat([self.edge_weight, torch.ones(torch.sum(~mask).item())])
        index0 = index0[~mask]
        index1 = index1[~mask]
        return index0, index1
                
            
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
        self.peak_memory = GPUs[1].memoryUsed
        if g.number_of_edges() > self.peak_edges:
            # index0 = g.edges()[0]
            # index1 = g.edges()[1]
            self.epoch_record = epoch
            dropedge = DropEdge(p=0.01)
            g = dropedge(g)
            # mask = g.has_edges_between(index0, index1)
            # self.edge_weight = self.edge_weight[mask.cpu()]
            self.whether_explain = False
            # self.whether_train_graph = False
            self.peak_memory = GPUs[1].memoryUsed
            print(g.number_of_edges())
        
        # print("current process memory used: ", get_gpu_process_info(pid=str(os.getpid())))
        print("gcn training epoch{} train loss :{}, acc: {}, best acc:{}, memory: {}".format(epoch, train_loss, acc, self.best_val_acc, GPUs[1].memoryUsed))
        # with open("record.txt", "a") as f:
        #     f.write(str(epoch) + " " + str(train_loss.item()) + " " + str(acc) + '\n')
        

        if acc > self.best_val_acc:
            self.best_val_acc = acc

        if self.whether_explain == True:
            with torch.no_grad():
                pre_logits = self.pre_logits
                self.train_explain_multi(pre_logits, logits.cpu(), train_idx, epoch, g.num_edges(), GPUs[1].memoryUsed, acc)
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
        with open("record_gcn.txt", "a") as f:
            f.write(str(epoch) + " " + str(loss.item()) + " " + str(num_edges) + " " + str(memory) + " " + str(acc) + " " + str(self.explain_threshold)+ '\n')
        if np.percentile(self.hdiff, 25) >= self.explain_threshold and epoch > 100:
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
        # with open("record_gcn_multi.txt", "a") as f:
        #     f.write(str(epoch) + " " + str(loss) + " " + str(num_edges) + " " + str(memory) + " " + str(acc) + " " + str(self.explain_threshold)+ '\n')
        if np.percentile(self.hdiff, 75) >= self.explain_threshold and epoch > 100:
            # self.whether_train_graph = False
            self.peak_edges = num_edges
            self.whether_explain = False
            GPUs = GPUtil.getGPUs()
            self.peak_memory = GPUs[1].memoryUsed
        del logits, pre_logits, loss
        torch.cuda.empty_cache()
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--prob', type=str, default='gradient')
    parser.add_argument('--memory', type=int, default=99999)
    parser.add_argument('--th', type=float, default=99999)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--edges', type=int, default=999999999)
    parser.add_argument('--drop', type=float, default=1)
    parser.add_argument('--edge_per', type=float, default=1)
    parser.add_argument('--explain', type=str, default='N')
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--first_size', type=int, default=500000)
    parser.add_argument('--second_size', type=int, default=100000)
    args = parser.parse_args()
    print(args.prob, args.memory, args.th, args.data, args.edges, args.drop, args.model, args.edge_per)

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
        # feats = g.ndata['feat']
        # scaler = StandardScaler()
        # scaler.fit(feats[g.ndata['train_mask']])
        # feats = scaler.transform(feats)
        # g.ndata['feat'] = torch.tensor(feats, dtype=torch.float)

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

    if args.prob == 'gradient':
        # for ours gcn
        print("cal probability gcn")
        d1 = torch.pow(g.out_degrees(), -0.5)
        p_norm = []
        for i in g.nodes():
            neighbors = g.out_edges(i)[1].long()
            d2 = d1[neighbors] * d1[i]
            norm_2 = torch.norm(d2, p=2)
            p_norm.append(norm_2)
        g.ndata['p_norm'] = torch.tensor(p_norm)
        

    if args.prob == 'feature':
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
        model = GCN(g.ndata['feat'].size(1), 128, num_class).to(device)
    elif args.data == 'reddit':
        learning_rate = 0.01
        model = GNNModel(args.model, 4, 256, g.ndata['feat'].size(1), num_class, 0).to(device)
    elif args.data == 'amazon':
        learning_rate = 0.01
        model = GNNModel(args.model, 3, 128, g.ndata['feat'].size(1), num_class, 0).to(device)
    elif args.data == 'products':
        learning_rate = 0.003
        model = GNNModel(args.model, 3, 128, g.ndata['feat'].size(1), num_class, 0).to(device)
    elif args.data == 'proteins':
        learning_rate = 0.01
        model = GNNModel(args.model, 3, 256, g.ndata['feat'].size(1), num_class, 0).to(device)
        
    pre_model = PreModel(model, device, args, learning_rate)
    good_edges = pre_model.find_graph(g)
