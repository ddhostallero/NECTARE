import numpy as np
import pandas as pd
from copy import deepcopy

import torch
import torch.nn.functional as F
import dgl

class Explainer:
    def __init__(self, hyperparams, model):
        # self.network = network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

    def predict_matrix(self, network, cell_index):
        """
        returns a prediction matrix of (N, n_drugs)
        """

        cell_index = cell_index.to(self.device)
        network = network.to(self.device)

        self.model.eval()
        preds = []

        with torch.no_grad():
            ccl_enc,drug_emb,_ = self.model.get_embeddings(network,
                drug_features=network.ndata['features']['drug'],
                cell_features=network.ndata['features']['cell_line'],
                cell_index=cell_index,
                gene_index=network.ndata['features']['gene'])
            # drug_emb = embeddings['drug']

            for i in range(len(cell_index)):
                expr_emb = ccl_enc[i].unsqueeze(0)
                pred = self.model.predict_from_embedding(expr_emb, drug_emb)
                preds.append(pred)

        preds = torch.cat(preds, axis=0).cpu().detach().numpy()
        return preds

    def per_drug_mse(self, labels, preds):
        mse = pd.Series(index=labels['drug_name'].unique(), dtype='float')

        for drug, df in labels.groupby('drug_name'):
            y_hat = preds.loc[df['ccl_name'].values, drug]
            y_true = df['response'].values

            if len(y_true) < 2 or len(y_hat) < 2:
                mse[drug] = np.nan # exclude if too little samples
            else:
                mse[drug] = ((y_true - y_hat)**2).mean()
        return mse


    def explain_drug_marginal_loss(self, network, cell_index, drug_index,
        bg_ccl_index, labels):

        n_genes = network.ndata['features']['cell_line'].shape[1]
        network = network.to(self.device)
        background = network.ndata['features']['cell_line'][bg_ccl_index]
        labels = labels.to(self.device)

        self.model.eval()
        marginal_errors = np.zeros((len(cell_index), n_genes))

        with torch.no_grad():
            ccl_enc, drug_emb,_ = self.model.get_embeddings(network,
                drug_features=network.ndata['features']['drug'],
                cell_features=network.ndata['features']['cell_line'],
                drug_index=drug_index,
                cell_index=cell_index)

            # drug_emb = node_embeddings['drug'][drug_index]
            pred = self.model.predict_from_embedding(ccl_enc, drug_emb)#.squeeze().cpu().numpy()
            baseline = F.mse_loss(pred, labels,reduction='none').squeeze().cpu().numpy()
            expand_size = (len(background), -1)

            for i, idx in enumerate(cell_index):
                sample = network.ndata['features']['cell_line'][idx]
                print(idx)
                
                for j in range(n_genes):
                    sample_repeat = sample.repeat(len(background), 1)
                    sample_repeat[:,j] = background[:,j]
                    background_out = self.model.predict_masked(sample_repeat, drug_emb, expand_size)
                    error = F.mse_loss(background_out, labels[i].repeat(len(background), 1))

                    marginal_errors[i,j] = error.cpu().numpy()

        return marginal_errors, baseline

    def explain_nodes_alldrugs(self, network, labels, maps):

        node_mse = pd.DataFrame(columns=maps['drug'].values)
        type_series = pd.Series(dtype=str)

        preds = self.predict_matrix(network, network.nodes('cell_line'))
        preds = pd.DataFrame(preds, index=maps['cell_line'].values, columns=maps['drug'].values)
        scores = self.per_drug_mse(labels, preds)
        node_mse.loc["full_graph", scores.index] = scores.values

        dst_etypes = {
            'cell_line': ['is_effective', 'is_ineffective', 'is_essential'],
            'drug': ['is_targeted_by', 'is_sensitive', 'is_resistant'],
            'gene': ['lowest_crispr_of', 'targets']
        }

        for ntype in network.ntypes:
            nodes = network.nodes(ntype)
            print(ntype)
            for node in nodes:
                node_name = maps[ntype][int(node)]

                # node removal with destination replacement
                h = deepcopy(network)
                null_node_idx = h.num_nodes(ntype)
                h.add_nodes(1, ntype=ntype)

                for etype in dst_etypes[ntype]:
                    src,_ = h.in_edges(node, etype=etype)
                    h.add_edges(u=src, v=null_node_idx, etype=etype)

                h = dgl.remove_nodes(h, nids=[node], ntype=ntype, store_ids=True) # will create a new network
                # drug_index_h = (h.ndata['_ID']['drug'] == drug_index_new).nonzero(as_tuple=True)[0][0]
                # preds = self.predict(h.to(device), expr, drug_index_h)
                # error = torch.nn.functional.mse_loss(preds, labels).item()

                
                # h = dgl.remove_nodes(network, nids=[node], ntype=ntype, store_ids=True) # will create a new network
                _ccl_ids = h.ndata['_ID']['cell_line'].tolist()
                _drug_ids = h.ndata['_ID']['drug'].tolist()
                ccl_nodes = h.nodes('cell_line')
                if ntype == 'cell_line':
                    _ccl_ids = _ccl_ids[:-1] # exclude null node
                    ccl_nodes = ccl_nodes[:-1]
                elif ntype == 'drug':
                    _drug_ids = _drug_ids[:-1] # exclude null node

                ccl_names = maps['cell_line'][_ccl_ids]
                drug_names = maps['drug'][_drug_ids]

                preds = self.predict_matrix(h, ccl_nodes)
                preds = pd.DataFrame(preds[:,:len(drug_names)], index=ccl_names, columns=drug_names) #exclude null node
                
                if ntype == 'cell_line' or ntype == 'drug':
                    labels_wo_node = labels.loc[(labels['ccl_name'].isin(ccl_names)) & (labels['drug_name'].isin(drug_names))]
                    scores = self.per_drug_mse(labels_wo_node, preds)
                else:
                    scores = self.per_drug_mse(labels, preds)

                type_series[node_name] = ntype
                node_mse.loc[node_name, drug_names] = scores[drug_names]

        node_mse['ntype'] = type_series
        return node_mse

    def explain_edges_alldrugs(self, network, labels, maps):
        from tqdm import tqdm

        edge_mse = pd.DataFrame(columns=maps['drug'].values)
        edge_ids = pd.DataFrame(columns=['src', 'dst', 'etype'])

        preds = self.predict_matrix(network, network.nodes('cell_line'))
        preds = pd.DataFrame(preds, index=maps['cell_line'].values, columns=maps['drug'].values)

        label_mat = labels.pivot(index='ccl_name', columns='drug_name', values='response')
        _drugs = label_mat.columns
        _ccls = label_mat.index
        edge_mse.loc[0] = ((label_mat - preds.loc[_ccls, _drugs])**2).mean()
        edge_ids.loc[0] = ['full_graph', 'full_graph', ""]

        i = 1
        for (src_ntype, etype, dst_ntype) in network.canonical_etypes:
            src, dst, eid = network.edges(form='all', etype=etype)

            # copy network and add null node
            net_with_null = deepcopy(network)
            null_node = net_with_null.num_nodes(dst_ntype)
            net_with_null.add_nodes(1, ntype=dst_ntype)
            
            for s,d,e in tqdm(zip(src, dst, eid)):
                # remove the edge from net_with_null
                h = dgl.remove_edges(net_with_null, eids=[e], etype=etype) # will create a new network

                # add null edge
                h.add_edges(u=s, v=null_node, etype=etype)

                ccl_nodes = h.nodes('cell_line')
                if dst_ntype == 'cell_line':
                    ccl_nodes = ccl_nodes[:-1]

                preds = self.predict_matrix(h, ccl_nodes)
                preds = pd.DataFrame(preds[:,:len(_drugs)], index=maps['cell_line'].values, columns=maps['drug'].values) #exclude null node
                # preds = self.predict(h.to(device), expr, drug_index_new)
                # error = torch.nn.functional.mse_loss(preds, labels).item()

                edge_mse.loc[i] = ((label_mat - preds.loc[_ccls, _drugs])**2).mean()
                edge_ids.loc[i] = [maps[src_ntype][int(s)],maps[dst_ntype][int(d)],etype]

                # row.append(error)
                # edge_mse.loc[i] = row
                i+=1


        return edge_ids.join(edge_mse)