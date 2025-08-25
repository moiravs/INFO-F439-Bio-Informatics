import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from typing_extensions import AbstractSet
from torch.utils.data import Dataset, TensorDataset
from torch_geometric.nn import GATConv, GATv2Conv, ChebConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.nn import GINConv, global_mean_pool
from torch.nn import BatchNorm1d as BN
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import pickle
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dataclasses import dataclass

@dataclass
class ModelConfig:
    method: str
    parallel: bool
    l2: float
    decoder: bool
    poolsize: int
    poolrate: float
    edge_weights: list
    edge_attributes: list
    num_gene: int
    num_mirna: int
    omic_mode: int
    num_classes: int
    dropout_rate: float
    n_epochs: int

    

class AbstractGraph(torch.nn.Module):
    def __init__(self, method, 
                    parallel, 
                    l2, 
                    decoder, 
                    poolsize, 
                    poolrate,
                    edge_weights, 
                    edge_attributes, 
                    num_gene,
                    num_mirna, 
                    omic_mode, 
                    num_classes, 
                    dropout_rate,
                    n_epochs):
        super(AbstractGraph, self).__init__()
        self.omic_mode = omic_mode
        self.method = method
        self.parallel = parallel
        self.decoder = decoder
        self.l2 = l2
        self.poolsize = poolsize
        self.poolrate = poolrate
        self.edge_weights = edge_weights
        self.edge_attributes = edge_attributes
        self.num_gene = num_gene
        self.num_mirna = num_mirna
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.epochs = n_epochs
        
        self.hid = 6
        self.head = 8
        self.raised_dimension = 8
        self.concate_layer = 64

        if self.omic_mode < 3:
            self.num_features = 1
        else:
            self.num_features = 2

        self.pre_conv_linear_gene = nn.Linear(self.num_features, self.raised_dimension)
        self.pre_conv_linear_mirna = nn.Linear(1, self.raised_dimension)
        
    def create_batch_index(self, batches):
        batch_index = []
        for i in range(batches):
            batch_index += [i]*(self.num_gene + self.num_mirna)
        return(torch.Tensor(batch_index).type(torch.int64))

    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0,2,1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p
            x = x.permute(0,2,1).contiguous()  # x = B x V/p x F
        return x

    def loss(self, x_reconstruct, x_target, y, y_target, l2_regularization):
        loss1 = 0
        if self.decoder:
            if self.num_mirna == 0 or self.num_features == 1:
                x_target = x_target.view(x_target.size()[0], -1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target)
            else:
                x_target_exp_mirna = x_target[:,:,0]
                x_target_cnv = x_target[:,:,1]
                ## separate mirna from the rest
                x_target_cnv = x_target_cnv[:,:-100]
                x_target_exp = x_target_exp_mirna[:,:-100]
                x_target_mirna = x_target_exp_mirna[:,-100:]
                x_target_flatten = torch.cat([x_target_exp, x_target_cnv, x_target_mirna], dim=1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target_flatten)
        loss2 = nn.CrossEntropyLoss()(y, y_target)
        loss = loss1 + loss2
        
        if self.l2:
            l2_loss = 0.0
            for param in self.parameters():
                data = param* param
                l2_loss += data.sum()
            loss += 0.2* l2_regularization* l2_loss
        return loss

    
    def forward(self, x, edge_index, edge_weight):
        batches = x.shape[0]
        num_node = x.shape[1]
        
        if self.num_mirna == 0 or self.num_features == 1:
            x = self.pre_conv_linear_gene(x)
            x = F.relu(x)
        else:
            ## the second matrix cnv_data has padding
            x_exp_mirna = x[:,:,0]
            x_cnv = x[:,:,1]
            ## separate mirna from the rest
            x_cnv = x_cnv[:,:-100]
            x_exp = x_exp_mirna[:,:-100]

            x_cnv = x_cnv.view(batches,-1,1)
            x_exp = x_exp.view(batches,-1,1)
            x_gene = torch.cat([x_exp,x_cnv],dim=1)
            x_gene = x_gene.view(-1,self.num_features)
            x_mirna = x_exp_mirna[:,-100:]
            x_mirna = torch.flatten(x_mirna)
            x_mirna = x_mirna.view(-1, 1)
            x_gene = self.pre_conv_linear_gene(x_gene)
            x_gene = F.relu(x_gene)
            x_mirna = self.pre_conv_linear_mirna(x_mirna)
            x_mirna = F.relu(x_mirna)
            x_gene = x_gene.view(batches, -1, self.raised_dimension)
            x_mirna = x_mirna.view(batches, -1, self.raised_dimension)
            x = torch.cat([x_gene,x_mirna],dim=1)
            
        x_parallel = x
        x = x.view(-1, self.raised_dimension)
        x_parallel = x_parallel.view(batches,-1)

        if self.edge_weights:
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_weight)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index) ## output shape: [batches * num_node, hid * head]
        x = F.relu(x)

        ## pooling on the graph to reduce nodes
        x = x.view(batches, num_node, -1) ## output shape: [batches, num_node, hid * head]
        x = self.graph_max_pool(x, self.poolsize)   ## if "gat", then output shape: [batches, floor(num_node / poolsize), hid * head]
                                                    ## if "gcn", then output shape: [batches, floor(num_node / poolsize), hid]

        if self.method == 'gcn':
            x = x.view(-1, self.hid) ## output shape:[batches * floor(num_node / poolsize), hid]

        x = x.view(batches, -1) ## output size: [batches, floor(num_node / poolsize) * hid * head]
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)

        if self.decoder:
            x_reconstruct = x
            x_reconstruct = self.decoder_1(x_reconstruct)
            x_reconstruct = F.relu(x_reconstruct)

            x_reconstruct  = nn.Dropout(0.2)(x_reconstruct)
            x_reconstruct = self.decoder_2(x_reconstruct)

        if self.parallel:
            ## the two layer shallow FC network
            x_parallel = self.parallel_linear1(x_parallel)
            x_parallel = F.relu(x_parallel)
            x_parallel = self.parallel_linear2(x_parallel)
            x_parallel = F.relu(x_parallel)

            x = torch.cat((x,x_parallel),1)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.classifier(x)

        if self.decoder:
            return x_reconstruct, F.log_softmax(x, dim=1)
        return F.log_softmax(x, dim=1)

    def create_classifier(self):
        self.linear1 = nn.Linear(self.linear_input, self.linear_input//4)
        self.linear2 = nn.Linear(self.linear_input//4, self.concate_layer)

        if self.decoder:
            if self.num_features == 1:
                ## Omic mode: Exp, mi, Exp+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene+self.num_mirna)
            elif self.num_features == 2:
                ## omic_mode: Exp+CNV, Exp+CNV+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene*self.num_features + self.num_mirna)


        if self.parallel:
            parallel_input = self.raised_dimension*(self.num_gene + self.num_mirna)
            self.parallel_linear1 = nn.Linear(parallel_input, parallel_input//4)
            self.parallel_linear2 = nn.Linear(parallel_input//4, self.concate_layer)
            self.classifier = nn.Linear(self.concate_layer*2, self.num_classes)
        else:
            self.classifier = nn.Linear(self.concate_layer, self.num_classes)

# Add this at the top of layer_model2.py, after the imports:

def extend_graph(edge_index, edge_weight, edge_attr, batch_size, n_nodes, use_attr):
    """
    Replicate single‐graph edge_index and edge_weight/edge_attr
    across 'batch_size' graphs by block‐diagonal tiling.
    """
    base_ei = edge_index.long()
    base_ew = edge_attr if use_attr else edge_weight

    # build list of shifted adjacency blocks
    ei_blocks = [base_ei + i * n_nodes for i in range(batch_size)]
    ew_blocks = [base_ew for _ in range(batch_size)]

    batch_ei = torch.cat(ei_blocks, dim=1)
    batch_ew = torch.cat(ew_blocks, dim=0)
    return batch_ei, batch_ew

# Training function for the ensemble
def train_stacking_ensemble(train_loader, val_loader, test_loader, device, args, num_mirna, labels, edge_index, edge_weight):
    """
    Train a stacking ensemble of multiple models
    """
    
    # Define configurations for different base models
    base_configs = []
    
    # GAT configuration
    gat_config = ModelConfig(
        method='gat',
        parallel=args.parallel,
        l2=args.l2,
        decoder=args.decoder,
        poolsize=args.poolsize,
        poolrate=args.poolrate,
        edge_weights=args.edge_weight,
        edge_attributes=args.edge_attribute,
        num_gene=args.num_gene,
        num_mirna=num_mirna,
        omic_mode=args.omic_mode,
        num_classes=len(np.unique(labels)),
        dropout_rate=args.dropout,
        n_epochs=args.epochs
    )
    base_configs.append(gat_config)
    
    # GIN configuration
    gin_config = ModelConfig(
        method='gin',
        parallel=args.parallel,
        l2=args.l2,
        decoder=False,
        poolsize=args.poolsize,
        poolrate=args.poolrate,
        edge_weights=False,
        edge_attributes=False,
        num_gene=args.num_gene,
        num_mirna=num_mirna,
        omic_mode=args.omic_mode,
        num_classes=len(np.unique(labels)),
        dropout_rate=args.dropout,
        n_epochs=args.epochs
    )
    base_configs.append(gin_config)
    
    # GraphSAGE configuration
    sage_config = ModelConfig(
        method='sage',
        parallel=args.parallel,
        l2=args.l2,
        decoder=args.decoder,
        poolsize=args.poolsize,
        poolrate=args.poolrate,
        edge_weights=args.edge_weight,
        edge_attributes=args.edge_attribute,
        num_gene=args.num_gene,
        num_mirna=num_mirna,
        omic_mode=args.omic_mode,
        num_classes=len(np.unique(labels)),
        dropout_rate=args.dropout,
        n_epochs=args.epochs
    )
    base_configs.append(sage_config)
    
    # Create ensemble
    ensemble = StackingEnsemble(base_configs, meta_model_type='logistic', cv_folds=3)
    ensemble.to(device)
    
    # Generate meta-features (now passing edge_index and edge_weight)
    train_meta_features, val_meta_features, train_meta_labels, val_meta_labels = \
        ensemble.generate_meta_features(train_loader, val_loader, device, edge_index, edge_weight)
    
    # Train meta-model
    ensemble.train_meta_model(train_meta_features, train_meta_labels)
    
    # Evaluate on validation set
    val_meta_preds = ensemble.meta_model.predict(val_meta_features)
    val_accuracy = np.mean(val_meta_preds == val_meta_labels)
    
    print(f"Stacking Ensemble Validation Accuracy: {val_accuracy:.4f}")
    
    # Save ensemble
    ensemble.save_ensemble('stacking_ensemble.pth')
    
    return ensemble, val_accuracy


