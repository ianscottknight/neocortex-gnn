import torch
import torch.nn.functional as F
import torch.nn as nn

from layers import GATGate


class GNN(torch.nn.Module):
    NUM_ATOM_FEATURES = 28

    def __init__(self, args):
        super().__init__()
        
        #
        n_graph_layer = args.n_graph_layer
        d_graph_layer = args.d_graph_layer
        n_fc_layer = args.n_fc_layer
        d_fc_layer = args.d_fc_layer
        
        #
        self.dropout_rate = args.dropout_rate 
        self.layers1 = [d_graph_layer for i in range(n_graph_layer+1)]
        self.gconv1 = nn.ModuleList([GATGate(self.layers1[i], self.layers1[i+1]) for i in range(len(self.layers1)-1)]) 
        self.fc = nn.ModuleList([nn.Linear(self.layers1[-1], d_fc_layer) if i == 0 else
                                 nn.Linear(d_fc_layer, 1) if i == n_fc_layer - 1 else
                                 nn.Linear(d_fc_layer, d_fc_layer) for i in range(n_fc_layer)])
        self.mu = nn.Parameter(torch.Tensor([args.initial_mu]).float())
        self.dev = nn.Parameter(torch.Tensor([args.initial_dev]).float())
        self.embede = nn.Linear(2 * self.NUM_ATOM_FEATURES, d_graph_layer, bias=False)
        
    def embed_graph(self, data):
        #
        c_hs, c_adjs1, c_adjs2, c_valid = data
        c_hs = self.embede(c_hs)
        hs_size = c_hs.size()
        c_adjs2 = torch.exp(-torch.pow(c_adjs2-self.mu.expand_as(c_adjs2), 2) / self.dev) + c_adjs1
        regularization = torch.empty(len(self.gconv1), device=c_hs.device)

        #
        for k in range(len(self.gconv1)):
            c_hs1 = self.gconv1[k](c_hs, c_adjs1)
            c_hs2 = self.gconv1[k](c_hs, c_adjs2)
            c_hs = c_hs2-c_hs1
            c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)

        #
        c_hs = c_hs*c_valid.unsqueeze(-1).repeat(1, 1, c_hs.size(-1))
        c_hs = c_hs.sum(1)

        return c_hs

    def fully_connected(self, c_hs):
        #
        regularization = torch.empty(len(self.fc)*1-1, device=c_hs.device)

        #
        for k in range(len(self.fc)):
            #c_hs = self.fc[k](c_hs)
            if k<len(self.fc)-1:
                c_hs = self.fc[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
                c_hs = F.relu(c_hs)
            else:
                c_hs = self.fc[k](c_hs)

        #
        c_hs = torch.sigmoid(c_hs)

        return c_hs

    def train_model(self, data):
        # embed graph
        c_hs = self.embed_graph(data)

        # fully connected NN
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1) 

        # note that if you don't use concrete dropout, regularization 1-2 is zero
        return c_hs
    
    def test_model(self, data1):
        c_hs = self.embed_graph(data1)
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1)

        return c_hs
