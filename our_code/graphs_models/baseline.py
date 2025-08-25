from graphs_models.layer_model import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(AbstractGraph):
    def __init__(self, config: ModelConfig):
        super(Baseline, self).__init__(**vars(config))

        parallel_input = self.raised_dimension*(self.num_gene + self.num_mirna)

        self.parallel_linear1 = nn.Linear(parallel_input, parallel_input//2)
        self.parallel_linear2 = nn.Linear(parallel_input//2, parallel_input//4)
        self.parallel_linear3 = nn.Linear(parallel_input//4, self.concate_layer)
        self.classifier = nn.Linear(self.concate_layer, self.num_classes)

        
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
        x_parallel = x_parallel.view(batches,-1)
        
        x_parallel = self.parallel_linear1(x_parallel)
        x_parallel = F.relu(x_parallel)
        x_parallel = self.parallel_linear2(x_parallel)
        x_parallel = F.relu(x_parallel)
        x_parallel = self.parallel_linear3(x_parallel)
        x_parallel = F.relu(x_parallel)

        x_parallel = F.dropout(x_parallel, p=self.dropout_rate, training=self.training)
        x_parallel = self.classifier(x_parallel)
        return F.log_softmax(x_parallel, dim=1)
    
    def loss(self, x_reconstruct, x_target, y, y_target, l2_regularization):
        return nn.CrossEntropyLoss()(y, y_target)

