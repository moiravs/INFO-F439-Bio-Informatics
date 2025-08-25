import os
from random import shuffle
import time
import gc
import tqdm
from sklearn.metrics import f1_score
from prettytable import PrettyTable

import numpy as np
import pandas as pd
import scipy.sparse as sp
import argparse
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from torch_geometric.loader import DataLoader
from sklearn import preprocessing
from torch_sparse import SparseTensor

from utils import *
import itertools

from graphs_models.gin import GIN
from graphs_models.gat import GAT
from graphs_models.gcn import GCN
from graphs_models.gsage import GraphSAGE
from graphs_models.baseline import Baseline, ModelConfig
from graphs_models import *


np.random.seed(2022)
torch.manual_seed(2022)
torch.cuda.manual_seed(2022)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
gc.collect()
torch.cuda.empty_cache()
l2_regularization = 5e-4

def load_data():
    path = 'data/cancer/'
    expression_variance_path = path + 'expression_variance.tsv'
    non_null_index_path = path + 'biogrid_non_null.csv'
    if args.cancer_subtype:
        if args.specific_type.lower() == 'brca':
            shuffle_index_path = path + 'brca_shuffle_index.tsv'
            cancer_subtype_label_path = path + 'brca_subtype.csv'
            expression_data_path = path + 'expression_data_brca.tsv'
            cnv_data_path = path + 'cnv_data_brca.tsv'
            mirna_data_path = path +'mirna_data_brca.tsv'
    else:
        expression_data_path = path + 'standardized_expression_data_with_labels.tsv'
        cnv_data_path = path + 'standardized_cnv_data_with_labels.tsv'
        mirna_data_path = path +'top_100_mirna_data.tsv'
        shuffle_index_path = path + 'common_trimmed_shuffle_index_'+ str(args.shuffle_index) + '.tsv'
    adjacency_matrix_path = path + 'adj_matrix_biogrid.npz'
    mirna_to_gene_matrix_path = path + 'standardized_mirna_mrna_edge_filtered_at_eight_with_top_100_mirna.npz'

    ## use the loading function to load the data
    ## use the loading function to load the data
    if args.omic_mode < 3:
        expr_all_data, mirna_all_data = load_exp_and_real_mirna_data(expression_data_path, mirna_data_path)

        adj, train_data_all, labels, shuffle_index = down_sampling_exp_cnv_and_real_mirna_data(expression_variance_path=expression_variance_path,
                                                                            expression_data=expr_all_data,
                                                                            cnv_data=pd.DataFrame(),
                                                                            mirna_data=mirna_all_data,
                                                                            omic_mode=args.omic_mode,
                                                                            non_null_index_path=non_null_index_path,
                                                                            shuffle_index_path=shuffle_index_path,
                                                                            adjacency_matrix_path=adjacency_matrix_path,
                                                                            mirna_to_gene_matrix_path=mirna_to_gene_matrix_path,
                                                                            gene_gene=args.gene_gene,
                                                                            mirna_gene=args.mirna_gene,
                                                                            mirna_mirna=args.mirna_mirna,
                                                                            number_gene=args.num_gene,
                                                                            singleton=False)
    else:
        expr_all_data, cnv_all_data, mirna_all_data = load_exp_cnv_and_real_mirna_data(expression_data_path, cnv_data_path, mirna_data_path)

        adj, train_data_all, labels, shuffle_index = down_sampling_exp_cnv_and_real_mirna_data(expression_variance_path=expression_variance_path,
                                                                            expression_data=expr_all_data,
                                                                            cnv_data=cnv_all_data,
                                                                            mirna_data=mirna_all_data,
                                                                            omic_mode=args.omic_mode,
                                                                            non_null_index_path=non_null_index_path,
                                                                            shuffle_index_path=shuffle_index_path,
                                                                            adjacency_matrix_path=adjacency_matrix_path,
                                                                            mirna_to_gene_matrix_path=mirna_to_gene_matrix_path,
                                                                            gene_gene=args.gene_gene,
                                                                            mirna_gene=args.mirna_gene,
                                                                            mirna_mirna=args.mirna_mirna,
                                                                            number_gene=args.num_gene,
                                                                            singleton=False)

    if args.cancer_subtype:
        train_data_all, labels = filter_data_by_cancer_type(cancer_subtype_label_path, train_data_all,expr_all_data)
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)   ## process the labels to make sure it starts from 0
    shuffle_index = shuffle_index.astype(np.int32).reshape(-1)
    return adj, train_data_all, labels, shuffle_index

def split_training_data(train_data_all, shuffle_index):
    train_size, val_size = int(len(shuffle_index)* args.train_ratio), int(len(shuffle_index)* (1- args.test_ratio))

    train_data = np.asarray(train_data_all).astype(np.float32)[shuffle_index[0:train_size]]
    val_data = np.asarray(train_data_all).astype(np.float32)[shuffle_index[train_size:val_size]]
    test_data = np.asarray(train_data_all).astype(np.float32)[shuffle_index[val_size:]]

    train_labels = labels[np.array(shuffle_index[0:train_size])].astype(np.int64)
    val_labels = labels[shuffle_index[train_size:val_size]].astype(np.int64)
    test_labels = labels[shuffle_index[val_size:]].astype(np.int64)

    train_data = torch.FloatTensor(train_data)
    test_data = torch.FloatTensor(test_data)
    val_data = torch.FloatTensor(val_data)

    train_labels = torch.LongTensor(train_labels)
    test_labels = torch.LongTensor(test_labels)
    val_labels = torch.LongTensor(val_labels)

    dset_train = TensorDataset(train_data, train_labels)
    dset_test = TensorDataset(test_data, test_labels)
    dset_val = TensorDataset(val_data, val_labels)

    train_loader = DataLoader(dset_train, batch_size = args.batch_size, shuffle = True)
    test_loader = DataLoader(dset_test, shuffle = False)
    val_loader = DataLoader(dset_val, batch_size = args.batch_size, shuffle = True)

    return train_loader, test_loader, val_loader, test_labels, train_size


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default = 0.01, help='learning rate.')
    parser.add_argument('--big_lr', type=str2bool, nargs='?', default = True, help='use the larger learning rate.')
    parser.add_argument('--num_gene', type=int, default = 1000, help='# of genes')
    parser.add_argument('--omic_mode', type=int, default = 0, help='which modes of omic to use')
    parser.add_argument('--num_omic', type=int, default = 1, help='# of the omic(s) used')
    parser.add_argument('--cancer_subtype', type=str2bool, nargs='?', default = False, help='if use the cancer subtype for classification')
    parser.add_argument('--specific_type',type=str, default='brca', choices=['brca','luad'], help='which cancer type to use for subtype classification')
    parser.add_argument('--shuffle_index',type=int, default=0, help='which shuffle index to use')
    parser.add_argument('--batch_size', type=int, default = 16, help='# of genes')
    parser.add_argument('--epochs', type=int, default = 100, help='# of epoch')
    parser.add_argument('--dropout', type=float, default = 0.6, help='dropout rate')
    parser.add_argument('--model', type=str, default='gat', 
                   choices=['gat','gatv2','gcn','baseline','gin','sage'], 
                   help='which model to use')
    parser.add_argument('--decay', type=float, default = 0.9, help='decay rate of the learing rate')
    parser.add_argument('--poolsize', type=int, default = 8, help='the max pooling size')
    parser.add_argument('--poolrate', type=float, default = 0.8, help='the pooling rate used in the self-attention pooling layer')
    parser.add_argument('--gene_gene', type=str2bool, nargs='?', default = True, help='if use the Gene-gene inner connections')
    parser.add_argument('--mirna_gene', type=str2bool, nargs='?', default = True, help='if use mirna-mrna connections')
    parser.add_argument('--mirna_mirna', type=str2bool, nargs='?', default = True, help='include the meta-path within the mirna')
    parser.add_argument('--parallel', type=str2bool, nargs='?', default = True, help='if use the parallel structure')
    parser.add_argument('--l2', type=str2bool, nargs='?', default = True, help='if use the l2 regularization')
    parser.add_argument('--decoder', type=str2bool, nargs='?', default = True, help='if use the decoder for the graph')
    parser.add_argument('--edge_attribute', type=str2bool, nargs='?', default = False, help='if use multi-demension attributes for edges')
    parser.add_argument('--edge_weight', type=str2bool, nargs='?', default = False, help='if use score as the edge weight instead of binary edges')
    parser.add_argument('--train_ratio', type=float, default = 0.8, help='the ratio of the training data')
    parser.add_argument('--test_ratio', type=float, default = 0.1, help='the ratio of the test data')
    parser.add_argument('--ensemble', type=str2bool, nargs='?', default=False, 
                   help='use stacking ensemble of multiple models')
    return parser.parse_args()

def test(loader, num_classes, model):
    model.eval()
    correct = 0
    test_accuracy = 0
    predictions = pd.DataFrame()
    all_true = []
    all_pred = []
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        batch_edge_index = edge_index.type(torch.int64)
        
        if args.edge_attribute:
            batch_edge_weight = edge_attribute
        else:
            batch_edge_weight = edge_weight

        for i in range(batch_y.shape[0] - 1):
            if args.edge_weight and args.edge_attribute == False:
                batch_edge_weight = torch.cat([batch_edge_weight, edge_weight], axis=0)
            elif args.edge_attribute:
                batch_edge_weight = torch.cat([batch_edge_weight, edge_attribute], axis=0)
            batch_edge_index = torch.cat([batch_edge_index, edge_index+i*(args.num_gene+num_mirna)], axis=1)
        batch_edge_index = batch_edge_index.to(device)
        batch_edge_weight = batch_edge_weight.to(device)

        if args.decoder:
            x_reconstruct, out = model(batch_x, batch_edge_index, batch_edge_weight)
        else:
            out = model(batch_x, batch_edge_index, batch_edge_weight)
        
        px = pd.DataFrame(out.detach().cpu().numpy())            
        predictions = pd.concat((predictions, px), axis=0)

        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == batch_y).sum())  # Check against ground-truth labels.
        test_accuracy += accuracy(out, batch_y)

        all_true.extend(batch_y.cpu().numpy())
        all_pred.extend(pred.cpu().numpy())
    f1 = f1_score(all_true, all_pred, average='weighted')
    return correct / len(loader.dataset), f1  # Derive ratio of correct predictions.


def train(args, model_params, num_mirna, nclass):
    if args.model == 'gcn':
        model = GCN(model_params).to(device)       
    elif args.model == 'gat' or args.model == 'gatv2':
        model = GAT(model_params).to(device)
    elif args.model == 'baseline':
        model = Baseline(model_params).to(device)
    elif args.model == 'gin':
        model = GIN(model_params).to(device)
    elif args.model == 'sage':
        model = GraphSAGE(model_params).to(device)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    global_step = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    for _ in tqdm.tqdm(range(args.epochs)):
        model.train()
        loss_all = 0.0
        accuracy_all = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            batch_edge_index = edge_index.type(torch.int64)

            if args.edge_attribute:
                batch_edge_weight = edge_attribute
            else:
                batch_edge_weight = edge_weight

            for i in range(batch_y.shape[0] - 1):
                if args.edge_weight and args.edge_attribute == False:
                    batch_edge_weight = torch.cat([batch_edge_weight, edge_weight], axis=0)
                elif args.edge_attribute:
                    batch_edge_weight = torch.cat([batch_edge_weight, edge_attribute], axis=0)
                batch_edge_index = torch.cat([batch_edge_index, edge_index+i*(args.num_gene+num_mirna)], axis=1)

            batch_edge_index = batch_edge_index.to(device)
            batch_edge_weight = batch_edge_weight.to(device)

            optimizer.zero_grad()
            if args.decoder:
                x_reconstruct, out = model(batch_x, batch_edge_index, batch_edge_weight)
                loss_batch = model.loss(x_reconstruct, batch_x, out, batch_y, l2_regularization)
                
            else:
                out = model(batch_x, batch_edge_index, batch_edge_weight)
                loss_batch = model.loss(batch_x.view(batch_x.size()[0], -1), batch_x, out, batch_y, l2_regularization)

            accuracy_batch = accuracy(out, batch_y)
            loss_batch.backward()
            optimizer.step()
            loss_all += loss_batch.item()
            accuracy_all += accuracy_batch
            global_step += args.batch_size

        accuracy_all = accuracy_all / len(train_loader)

    return model

args_grid = {
    "lr": [0.01],
    "big_lr": [True],
    "num_gene": [500],
    "omic_mode": [0],
    "cancer_subtype": [True],
    "specific_type": ["brca"],
    "shuffle_index": [0],
    "batch_size": [16],
    "epochs": [5],
    "dropout": [0.6],
    "model": ["gcn"],
    "decay": [0.9],
    "poolsize": [8],
    "poolrate": [0.8],
    "gene_gene": [True],
    "mirna_gene": [True],
    "mirna_mirna": [True],
    "parallel": [False],
    "l2": [True],
    "decoder": [False],
    "edge_attribute": [False],
    "edge_weight": [False],
    "train_ratio": [0.8],
    "test_ratio": [0.1],
    "name": ["lala"],
}


desired_combinations = [
    {
        "name": "GCN (Original)",
        "model": "gcn",
        "omic_mode": 0,
        "decoder": False,
        "parallel": False,
    },
    {
        "name": "GCN (Modified)",
        "model": "gcn",
        "omic_mode": 4,
        "decoder": False,
        "parallel": False,
    },
    {
        "name": "Multi-omics GCN (Original)",
        "model": "gcn",
        "omic_mode": 3,
        "decoder": True,
        "parallel": True,
    },
    {
        "name": "Multi-omics GCN (Modified)",
        "model": "gcn",
        "omic_mode": 0,
        "decoder": True,
        "parallel": True,
    },

    {
        "name": "Multi-omics GAT (Original)",
        "model": "gat",
        "omic_mode": 2,
        "decoder": False,
        "parallel": False,
    },
    {
        "name": "Multi-omics GAT (Modified)",
        "model": "gat",
        "omic_mode": 4,
        "decoder": False,
        "parallel": False,
    },
]

table = PrettyTable()
table.field_names = ["Model", "Time", "Accuracy", "F1 Score"]

num_iter = 5
final_table = "#table(columns:(auto,auto, auto),"
for combo in tqdm.tqdm(desired_combinations):
    result_avg = np.zeros(num_iter)
    f1_avg = np.zeros(num_iter)
    time_avg = np.zeros(num_iter)
    args = argparse.Namespace()
    print("here")
    for key, value in args_grid.items():
        if key in combo:
            val = combo[key]
            setattr(args, key, val)
        else:
            val = args_grid[key]
            setattr(args, key, val[0])
    print(combo)
    print(args)
    args.num_omic = omic_mode_translation(args.omic_mode)
    args.gene_gene, args.mirna_gene, args.mirna_mirna, args.num_mirna = validate_network_choice(args.omic_mode, args.gene_gene, args.mirna_gene, args.mirna_mirna)

    if args.omic_mode == 1:
        args.num_gene = 0

    if args.model == 'baseline':
        args.decoder = False
        args.parallel = False

    adj, train_data_all, labels, shuffle_index = load_data()
    train_loader, test_loader, val_loader, test_labels, train_size = split_training_data(train_data_all, shuffle_index)
    adj_for_loss = adj.todense()
    adj = adj/np.max(adj)
    adj = adj.astype('float32')
    adj.setdiag(0)  # we don't care about interactions between a gene and itself
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))

    device = 'cuda' if torch.cuda.is_available() else ('mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'cpu')
    print('Device', device)

    edge_index = torch.stack([torch.tensor(adj.row), torch.tensor(adj.col)], dim=0).long().to(device)
    edge_weight = torch.tensor(adj.data, dtype=torch.float32).to(device)
    if args.edge_attribute:
        edge_attribute = disassemble_edge_weights(edge_weight, edge_index, args.num_gene, args.num_omic).to(device)

    model_params = ModelConfig(
        args.model,
        args.parallel,
        args.l2,
        args.decoder,
        args.poolsize,
        args.poolrate,
        args.edge_weight,
        args.edge_attribute,
        args.num_gene,
        args.num_mirna,
        args.omic_mode,
        len(np.unique(labels)),
        args.dropout,
        args.epochs
    )
    nclass = len(np.unique(labels))
    
    for i in range(num_iter):

        t_total_train = time.time()
        if args.model == 'ensemble':
            model_trained = train_stacking_ensemble(train_loader, val_loader, device, args, args.num_mirna, labels, edge_index, edge_weight)
        else:
            model_trained = train(args, model_params, args.num_mirna, nclass)
        time_taken = time.time() - t_total_train
        result = test(test_loader, nclass, model_trained)
        time_avg[i] = time_taken
        result_avg[i] = result[0]
        f1_avg[i] = result[1]
        

    final_table += f"[{args.name}],[{time_avg.mean():.2f}],[{result_avg.mean():.3f} + {result_avg.std():.2f}],[{f1_avg.mean():.3f} + {f1_avg.std():.2f}], \n"

final_table += ")"
print(final_table)



