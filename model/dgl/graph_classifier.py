from .rgcn_model import RGCN
from .layers import GraphLearner, GCN
from dgl import mean_nodes
from scipy.sparse import csr_matrix
import torch.nn as nn
import torch
import dgl
"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""

# global count=0
class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id
        # self.graph_count = 0
        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)

        self.gsl = GraphLearner(params)
        self.g_learn = GCN(self.params.emb_dim, self.params.emb_dim)

        if self.params.add_ht_emb:
            self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim + self.params.emb_dim, 1)
            # self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim + self.params.emb_dim, 1)
            # self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)

    def batch_graph_learner(self, batch_g):
        graphs = []
        for graph in dgl.unbatch(batch_g):
            
            node_features, learned_adj = self.gsl(graph.ndata['h'])
            csr_adj = csr_matrix(learned_adj.detach().cpu().numpy())
            learned_g = dgl.add_self_loop(dgl.from_scipy(csr_adj, eweight_name='e_w'))
            learned_g.ndata['h'] = node_features.detach().cpu()
            graphs.append(learned_g)
            
        return dgl.batch(graphs).to(self.params.device)
        # return dgl.batch(graphs)

    def forward(self, data):
        g, rel_labels = data
        # g.ndata['h'] = g.ndata['feat']
        
        g.ndata['h'], kl_losses = self.gnn(g)
        g_out = mean_nodes(g, 'repr')
        # dgl.save_graphs(f'graphs/fb237/graph_original.bin',g)
        ### structure learning
        learned_g = self.batch_graph_learner(g)
        learned_g.ndata['id'] = g.ndata['id']
        # dgl.save_graphs(f'graphs/fb237/graph_learned.bin',learned_g)
        # import ipdb;ipdb.set_trace();
        g_repre = self.g_learn(learned_g, learned_g.ndata['h'])
        
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]

        if self.params.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               self.rel_emb(rel_labels),
                               g_repre], dim=1)
            # g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
            #                    head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
            #                    tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
            #                    self.rel_emb(rel_labels)
            #                    ], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim), self.rel_emb(rel_labels), g_repre], dim=1)
            # g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim), self.rel_emb(rel_labels)], dim=1)

        output = self.fc_layer(g_rep)
        if len(kl_losses) > 0:
            kl_loss = torch.mean(kl_losses)
        else:
            kl_loss = None
        return output, kl_loss
