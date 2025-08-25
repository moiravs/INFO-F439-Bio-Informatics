from graphs_models.layer_model import *


class GIN(AbstractGraph):
    def __init__(self, config: ModelConfig):
        super(GIN, self).__init__(**vars(config))
        self.hid = 120
        
        mlp1 = nn.Sequential(
            nn.Linear(self.raised_dimension, self.hid),
            nn.ReLU(),
            nn.Linear(self.hid, self.hid),
            nn.ReLU()
        )
        self.conv1 = GINConv(mlp1)
        
        mlp2 = nn.Sequential(
            nn.Linear(self.hid, self.hid),
            nn.ReLU(),
            nn.Linear(self.hid, self.hid),
            nn.ReLU()
        )
        self.conv2 = GINConv(mlp2)
        
        if self.poolsize <= 1:
            self.linear_input = (self.num_gene + self.num_mirna) * self.hid
        else:
            self.linear_input = math.floor((self.num_gene + self.num_mirna) / self.poolsize) * self.hid
            
        self.create_classifier()
    
    def forward(self, x, edge_index, edge_weight=None):
        batches = x.shape[0]
        num_node = x.shape[1]
        
        if self.num_mirna == 0 or self.num_features == 1:
            x = self.pre_conv_linear_gene(x)
            x = F.relu(x)
        else:
            x_exp_mirna = x[:,:,0]
            x_cnv = x[:,:,1]
            
            x_cnv = x_cnv[:,:-100]
            x_exp = x_exp_mirna[:,:-100]
            
            x_cnv = x_cnv.view(batches, -1, 1)
            x_exp = x_exp.view(batches, -1, 1)
            x_gene = torch.cat([x_exp, x_cnv], dim=1)
            x_gene = x_gene.view(-1, self.num_features)
            
            x_mirna = x_exp_mirna[:, -100:]
            x_mirna = torch.flatten(x_mirna)
            x_mirna = x_mirna.view(-1, 1)
            
            x_gene = self.pre_conv_linear_gene(x_gene)
            x_gene = F.relu(x_gene)
            
            x_mirna = self.pre_conv_linear_mirna(x_mirna)
            x_mirna = F.relu(x_mirna)
            
            x_gene = x_gene.view(batches, -1, self.raised_dimension)
            x_mirna = x_mirna.view(batches, -1, self.raised_dimension)
            
            x = torch.cat([x_gene, x_mirna], dim=1)
        
        x_parallel = x
        x = x.view(-1, self.raised_dimension)
        x_parallel = x_parallel.view(batches, -1)
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = x.view(batches, num_node, -1)
        x = self.graph_max_pool(x, self.poolsize)
        
        x = x.view(batches, -1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        
        if self.decoder:
            x_reconstruct = x
            x_reconstruct = self.decoder_1(x_reconstruct)
            x_reconstruct = F.relu(x_reconstruct)
            x_reconstruct = nn.Dropout(0.2)(x_reconstruct)
            x_reconstruct = self.decoder_2(x_reconstruct)
        
        if self.parallel:
            x_parallel = self.parallel_linear1(x_parallel)
            x_parallel = F.relu(x_parallel)
            x_parallel = self.parallel_linear2(x_parallel)
            x_parallel = F.relu(x_parallel)
            
            x = torch.cat((x, x_parallel), 1)
        
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.classifier(x)
        
        if self.decoder:
            return x_reconstruct, F.log_softmax(x, dim=1)
        return F.log_softmax(x, dim=1)

