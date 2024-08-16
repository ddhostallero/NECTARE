import pandas as pd
import torch
import pickle
import networkx as nx

from utils.utils import mkdir, reindex_tuples, load_hyperparams
from utils.network_generator import dgl_graph_to_nx
from utils.data_initializer import initialize
from nectare.explainer import Explainer
from base.main_cv import BaseCV 
from nectare.model import Model

class ExplainCV(BaseCV):
    def __init__(self, FLAGS):
        super(MainCV, self).__init__(FLAGS)
        _, _, self.labels, _, _ = initialize(FLAGS)

    def _load_net_and_tuples(self, labels, train_folds, val_folds, fold_id):
        net_dir = self.directory + '/graphs/test_network_fold_%d.pkl'%fold_id
        map_dir = self.directory + '/graphs/test_network_map_fold_%d.pkl'%fold_id
        print('Loading network from %s.\nIf this is not the correct graph, the program will still run but the scores will be wrong.'%net_dir)
        network = pickle.load(open(net_dir, 'rb'))
        maps = pickle.load(open(map_dir, 'rb'))
        
        train_tuples = labels.loc[labels['fold'].isin(train_folds)]
        train_samples = list(train_tuples['cell_line'].unique())

        val_tuples = labels.loc[labels['fold'].isin(val_folds)]
        val_samples = list(val_tuples['cell_line'].unique())

        train_tuples,_,_= reindex_tuples(train_tuples, maps['drug'].index, train_samples,
            c_map=maps['cell_line'], d_map=maps['drug']) 
        val_tuples,_,_ = reindex_tuples(val_tuples, maps['drug'].index, val_samples,
            c_map=maps['cell_line'], d_map=maps['drug'])

        return network, train_tuples, val_tuples, train_samples, val_samples, maps

    def explain_gene_level(self, fold_id, train_folds, val_fold, pred_fold):
        
        hyperparams = load_hyperparams(self.directory+'/model_config_fold_%d.txt'%fold_id)
        network, train_tuples, test_tuples, _, test_samples, maps =\
            self._load_net_and_tuples(self.labels, train_folds, pred_fold, fold_id)

        model_path = self.directory + '/model_weights_fold_%d'%fold_id
        model = Model(hyperparams, network.num_nodes('gene'), network.etypes)
        model.load_state_dict(torch.load(model_path), strict=True)
        explainer = Explainer(hyperparams, model)

        drug = self.FLAGS.drug
        c_map = maps['cell_line']
        test_df = test_tuples.loc[test_tuples['drug_name']==drug].copy()
        test_df.index = test_df['ccl_name'].values
        ccls = list(test_df.index)
        ccl_idx = torch.LongTensor(c_map[ccls].values)
        drug_idx = torch.LongTensor([maps['drug'][drug]])
        labels = torch.FloatTensor(test_df['response'].values).unsqueeze(1)

        # all CCLs not in the training set of the selected drug (so it works for LPO too)
        bg_ccl_idx = torch.LongTensor(c_map[~c_map.index.isin(ccls)].values)

        marginal_loss, baseline = explainer.calculate_drug_marginal_loss(network, ccl_idx, drug_idx, bg_ccl_idx, labels)
        marginal_loss = pd.DataFrame(marginal_loss, index=ccls, columns=self.expr.columns)
        marginal_loss['baseline'] = baseline
        mkdir(self.directory + '/gex_expl/')
        marginal_loss.to_parquet(self.directory + '/gex_expl/loss_%s_%d.parquet'%(drug, fold_id))

    def explain_graph(self, fold_id, train_folds, val_fold, pred_fold):
        hyperparams = load_hyperparams(self.directory+'/model_config_fold_%d.txt'%fold_id)
        network, train_tuples, test_tuples, _, test_samples, maps =\
            self._load_net_and_tuples(self.labels, train_folds, pred_fold, fold_id)

        idx2name = {x: pd.Series(maps[x].index, index=maps[x].values) for x in maps.keys()}
        nxnet = dgl_graph_to_nx(network, idx2name)

        model_path = self.directory + '/model_weights_fold_%d'%fold_id
        model = Model(hyperparams, network.num_nodes('gene'), network.etypes)
        model.load_state_dict(torch.load(model_path), strict=True)
        explainer = Explainer(hyperparams, model)

        # Calculate node scores for all drugs
        print('Calculating node-removal scores...')

        mkdir(self.directory + '/node_expl/')
        node_omit_mse = explainer.explain_nodes_alldrugs(network, test_tuples, idx2name)
        baseline = node_omit_mse.loc['full_graph']
        node_score = pd.DataFrame(index=node_omit_mse.index[1:], columns=node_omit_mse.columns)

        # remove unreachable nodes for each drug
        for drug in node_score.columns[:-1]:
            spathlen = pd.Series(nx.shortest_path_length(nxnet, target=drug)) # shortest path length to drug
            valid_nodes = spathlen[(spathlen==1) | (spathlen==2)].index       # only reachable within 2-hops
            node_score.loc[valid_nodes, drug] = node_omit_mse.loc[valid_nodes, drug] - baseline[drug] # score
        node_score['ntype'] = node_omit_mse['ntype']
        
        # save
        savepath = self.directory + '/node_expl/node_score_%d.parquet'%(fold_id)
        node_score.to_parquet(savepath)
        print('Raw node scores saved in %s\n'%savepath)

        # Calculate edge scores for all drugs
        print('Calculating edge-removal losses...')

        mkdir(self.directory + '/edge_expl/')
        edge_omit_mse = explainer.explain_edges_alldrugs(network, test_tuples, idx2name)
        baseline = edge_omit_mse.loc[0]

        drugs = edge_omit_mse.columns[3:]
        edge_scores = pd.DataFrame(index=edge_omit_mse.index[1:], columns=drugs)
    
        # remove unreachable edges for each drug
        for drug in drugs:
            spathlen = pd.Series(nx.shortest_path_length(nxnet, target=drug))     # shortest path length to drug
            valid_dst = spathlen[spathlen<2].index
            valid_edges = edge_omit_mse.loc[edge_omit_mse['dst'].isin(valid_dst)] # destination is either the drug or 1-hop from the drug
            edge_scores.loc[valid_edges.index, drug] = valid_edges[drug] - baseline[drug] # score
        edge_scores = edge_omit_mse.loc[edge_omit_mse.index[1:], ['src', 'dst', 'etype']].join(edge_scores)

        # save
        savepath = self.directory + '/edge_expl/edge_score_%d.parquet'%(fold_id)
        edge_scores.to_parquet(savepath)
        print('Raw node scores saved in %s\n'%savepath)