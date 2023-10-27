import argparse

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
    ClusterGCNSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import GPUtil
from dgl.data import RedditDataset
import dgl.function as fn
from utils.data import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y
    
    
class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.GraphConv(in_size, hid_size, activation=F.relu, allow_zero_in_degree=True))
        self.layers.append(dglnn.GraphConv(hid_size, hid_size, activation=F.relu, allow_zero_in_degree=True))
        self.layers.append(dglnn.GraphConv(hid_size, out_size, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            if l != len(self.layers) - 1:
                h = self.dropout(h)
            h = layer(block, h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = torch.tensor([]).to("cuda:1")
    y_hats = torch.tensor([]).to("cuda:1")
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys = torch.cat((ys, blocks[-1].dstdata["label"]), dim=0)
            y_hats = torch.cat((y_hats, model(blocks, x)), dim=0)
    y_hats[y_hats > 0] = 1
    y_hats[y_hats <= 0] = 0
    return f1_score(ys.cpu(), y_hats.cpu(), average='micro')


def layerwise_infer(device, graph, nid, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(
            pred, label, task="multiclass", num_classes=num_classes
        )


def train(args, device, g, dataset, model, num_classes):
    # create sampler & dataloader
    train_idx = dataset[0]
    val_idx = dataset[1]
    
    use_uva = args.mode == "mixed"
    # neighbor sampling
    sampler = NeighborSampler(
        [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
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
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    peak_memory = 0
    for epoch in range(50):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            if y.dim() == 1:
                loss = F.cross_entropy(y_hat, y)
            else:
                loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction='sum')
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, g, val_dataloader, num_classes)
        
        GPUs = GPUtil.getGPUs()
        if GPUs[1].memoryUsed > peak_memory:
            peak_memory = GPUs[1].memoryUsed
            
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | peak memory {:.4f} ".format(
                epoch, total_loss / (it + 1), acc.item(), peak_memory
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="puregpu",
        choices=["cpu", "puregpu", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    # load and preprocess dataset
    print("Loading data")

    # amazon
    data = load_data('amazon', multilabel=True)
        
    g = data[2]
    num_classes = data[0]
    num_node = g.num_nodes()
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    g = g.int()
    
    g.ndata['label'] = g.ndata['label']
    g.ndata['train_mask'] = g.ndata['train_mask'].bool()
    g.ndata['val_mask'] = g.ndata['val_mask'].bool()
    g.ndata['test_mask'] = g.ndata['test_mask'].bool()  
    train_idx = torch.nonzero(g.ndata['train_mask']).squeeze().type(torch.int32)
    val_idx = torch.nonzero(g.ndata['val_mask']).squeeze().type(torch.int32)
    train_val_idx = (train_idx, val_idx)
    feats = g.ndata['feat']
    scaler = StandardScaler()
    scaler.fit(feats[g.ndata['train_mask']])
    feats = scaler.transform(feats)
    g.ndata['feat'] = torch.tensor(feats, dtype=torch.float)
    in_size = g.ndata["feat"].shape[1]
    out_size = num_classes
    
    model = GCN(in_size, 128, out_size).to(device)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        model = model.to(dtype=torch.bfloat16)

    # model training
    print("Training...")
    train(args, device, g, train_val_idx, model, num_classes)

    # # test the model
    # print("Testing...")
    # acc = layerwise_infer(
    #     device, g, dataset.test_idx, model, num_classes, batch_size=4096
    # )
    # print("Test Accuracy {:.4f}".format(acc.item()))