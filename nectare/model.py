import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import HeteroGraphConv, GraphConv

class Model(nn.Module):
    def __init__(self, hyp, n_gene_nodes, rel_names):
        super(Model, self).__init__()

        self.conv1 = HeteroGraphConv(
            {rel: dgl.nn.GraphConv(in_feats=hyp['common_dim'], out_feats=hyp['common_dim']) for rel in rel_names})
        self.conv2 = HeteroGraphConv(
            {rel: dgl.nn.GraphConv(in_feats=hyp['common_dim'], out_feats=hyp['common_dim']) for rel in rel_names})

        self.gene_node_embeddings = nn.Embedding(n_gene_nodes, hyp['common_dim'])

        self.drug_enc = nn.Linear(hyp['n_drug_feats'], hyp['common_dim'])
        self.cell_enc = nn.Linear(hyp['n_cl_feats'], hyp['common_dim'])
        self.expr_enc = nn.Linear(hyp['n_cl_feats'], hyp['expr_enc'])

        self.mid = nn.Linear(hyp['expr_enc'] + hyp['common_dim'], hyp['mid'])
        self.out = nn.Linear(hyp['mid'], 1)

        drop = hyp['drop']
        self.in_drop = nn.Dropout(drop[0])
        self.mid_drop = nn.Dropout(drop[1])   
        self.alpha = 0.5

    def forward(self, network, drug_features, cell_features, cell_index, drug_index, gene_index):
        
        expr_enc, drug_emb, _ = self.get_embeddings(network, drug_features, cell_features, cell_index, drug_index, gene_index)
        # expr_enc = h2['cell_line']
        # drug_emb = h2['drug']

        x = torch.cat([expr_enc,drug_emb],-1) # (batch, expr_enc_size+drugs_emb_size)
        x = self.in_drop(x)
        x = F.leaky_relu(self.mid(x)) 
        x = self.mid_drop(x) 
        out = self.out(x)
        return out

    def _graph_forward(self, network, node_features):
        h1 = self.conv1(network, node_features)
        h1 = {k: F.leaky_relu(v + self.alpha*node_features[k]) for k, v in h1.items()}
        h2 = self.conv2(network, h1)
        h2 = {k: F.leaky_relu(v + self.alpha*h1[k]) for k, v in h2.items()}
        return h2

    def get_node_embeddings(self, network, drug_features, cell_features, gene_index=[]):
        cell_enc = F.leaky_relu(self.cell_enc(cell_features))
        drug_enc = F.leaky_relu(self.drug_enc(drug_features))

        if len(gene_index) == 0:
            gene_emb = self.gene_node_embeddings.weight 
        else:
            gene_emb = self.gene_node_embeddings(gene_index)

        node_features = {'drug': drug_enc, 'cell_line': cell_enc, 'gene': gene_emb}
        h2 = self._graph_forward(network, node_features)
        return h2

    def get_embeddings(self, network, drug_features, cell_features, cell_index=[], drug_index=[], gene_index=[]):
        h2 = self.get_node_embeddings(network, drug_features, cell_features, gene_index)
        
        if len(cell_index) == 0:
            expr_enc = F.leaky_relu(self.expr_enc(cell_features))
        else:
            expr_enc = F.leaky_relu(self.expr_enc(cell_features[cell_index]))

        if len(drug_index) > 0:
            h2['drug'] = h2['drug'][drug_index]

        return expr_enc, h2['drug'], h2['gene']

    def predict_from_embedding(self, expr_enc, drug_emb):
        expr_enc = expr_enc.unsqueeze(1) # (batch, 1, expr_enc_size)
        drug_emb = drug_emb.unsqueeze(0) # (1, n_drugs, drug_emb_size)

        expr_enc = expr_enc.repeat(1,drug_emb.shape[1],1) # (n_samples, n_drugs, expr_enc_size)
        drug_emb = drug_emb.repeat(expr_enc.shape[0],1,1) # (n_samples, n_drugs, drug_emb_size)

        x = torch.cat([expr_enc,drug_emb],-1) # (batch, n_drugs, expr_enc_size+drugs_enc_size)
        x = self.in_drop(x)
        x = F.leaky_relu(self.mid(x)) # (batch, n_drugs, 1)
        x = self.mid_drop(x)
        out = self.out(x) # (batch, n_drugs, 1)
        out = out.view(-1, drug_emb.shape[1])
        return out

    def predict_masked(self, cell_features_masked, drug_emb, expand_size):
        # Used by the gene-level explainer only

        expr_enc = F.leaky_relu(self.expr_enc(cell_features_masked))
        drug_emb = drug_emb.expand(expand_size)
        x = torch.cat([expr_enc,drug_emb],-1) # (n_genes, expr_enc_size+drugs_enc_size)
        x = self.in_drop(x)
        x = F.leaky_relu(self.mid(x)) # (n_genes, 1)
        x = self.mid_drop(x)
        out = self.out(x) 
        return out