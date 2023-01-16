import torch
import torch.nn.functional as F
import torch.nn as nn

from layers import GATGate


class GNN(torch.nn.Module):
    NUM_ATOM_FEATURES = 65
    NUM_SPHERES = 45
    NUM_SPHERE_COORDINATES = 3

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
                                 nn.Linear(d_fc_layer, self.NUM_SPHERES * self.NUM_SPHERE_COORDINATES) if i == n_fc_layer - 1 else
                                 nn.Linear(d_fc_layer, d_fc_layer) for i in range(n_fc_layer)])
        self.mu = nn.Parameter(torch.Tensor([args.initial_mu]).float())
        self.dev = nn.Parameter(torch.Tensor([args.initial_dev]).float())
        self.embede = nn.Linear(self.NUM_ATOM_FEATURES, d_graph_layer, bias=False)
        
    def embed_graph(self, data):
        #
        c_p, c_a, c_d = data
        c_p = self.embede(c_p)
        p_size = c_p.size()
        c_d = torch.exp(-torch.pow(c_d-self.mu.expand_as(c_d), 2) / self.dev) + c_a
        regularization = torch.empty(len(self.gconv1), device=c_p.device)

        #
        for k in range(len(self.gconv1)):
            c_p1 = self.gconv1[k](c_p, c_a)
            c_p2 = self.gconv1[k](c_p, c_d)
            c_p = c_p2 - c_p1
            c_p = F.dropout(c_p, p=self.dropout_rate, training=self.training)

        #
        c_p = c_p.sum(1)

        return c_p

    def fully_connected(self, c_p):
        #
        regularization = torch.empty(len(self.fc) * 1 - 1, device=c_p.device)

        #
        for k in range(len(self.fc)):
            #c_p = self.fc[k](c_p)
            if k < len(self.fc) - 1:
                c_p = self.fc[k](c_p)
                c_p = F.dropout(c_p, p=self.dropout_rate, training=self.training)
                c_p = F.relu(c_p)
            else:
                c_p = self.fc[k](c_p)

        #
        c_p = torch.sigmoid(c_p)

        return c_p

    def train_model(self, data):
        # embed graph
        c_p = self.embed_graph(data)

        # fully connected NN
        c_p = self.fully_connected(c_p)

        # note that if you don't use concrete dropout, regularization 1-2 is zero
        return c_p
    
    def test_model(self, data):
        # embed graph
        c_p = self.embed_graph(data)

        # fully connected NN
        c_p = self.fully_connected(c_p)

        # note that if you don't use concrete dropout, regularization 1-2 is zero
        return c_p
