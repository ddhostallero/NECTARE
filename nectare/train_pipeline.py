import pandas as pd
import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from utils.utils import mkdir, reindex_tuples, reset_seed, load_hyperparams
from utils.network_generator import create_dir_resp_ess_tar_net, append_to_ret_net
from utils.data_initializer import initialize, load_crispr, load_drug_targets

from nectare.trainer import Trainer
from base.main_cv import BaseCV 
from nectare.model import Model

class MainCV(BaseCV):
    def __init__(self, FLAGS):
        super(MainCV, self).__init__(FLAGS)

        self.drug_feats, self.expr, self.labels, self.drug_bias, self.drug_scale = initialize(FLAGS)
        self.gene_essentiality, self.common_ess = load_crispr(FLAGS)
        self.drug_targets = load_drug_targets(FLAGS).loc[self.drug_feats.index]

        self.hyperparams = {
            'learning_rate': 1e-4,
            'num_epoch': 10,
            'batch_size': 128,
            'n_cl_feats': self.expr.shape[1],
            'n_drug_feats': self.drug_feats.shape[1],
            'expr_enc': 1024,
            'common_dim': 512,
            'mid': 512,
            'drop': [0.2,0.5]}
            
    def _create_dataset(self, labels, cell_lines, drug_feats, drug_targets, 
        gene_ess, network_perc, common_ess, train_folds, val_folds, normalizer,
        return_dataset=True):
        drug_list = drug_feats.index
        
        train_tuples, train_samples, train_x =\
            self._filter_folds(train_folds, labels, cell_lines)
        val_tuples, val_samples, val_x =\
            self._filter_folds(val_folds, labels, cell_lines)

        train_tuples,c_map,d_map = reindex_tuples(train_tuples, drug_list, train_samples) 
        val_tuples,c_map,d_map = reindex_tuples(val_tuples, drug_list, val_samples, c_map=c_map, d_map=d_map)
        train_val_x = cell_lines.loc[c_map.index]

        if normalizer is None:
            normalizer = StandardScaler()
            train_x = normalizer.fit_transform(train_x)
        else:
            train_x = normalizer.transform(train_x)
        train_val_x = normalizer.transform(train_val_x)

        # make sure index is aligned with the gene expression index (use DataFrame to preserve indexing)
        gene_ess = gene_ess.loc[gene_ess.index.isin(c_map.index)].copy()
        gene_ess.index = c_map[gene_ess.index] 

        # make sure index is aligned with the drug features index (use DataFrame to preserve indexing)
        drug_targets = drug_targets.loc[drug_targets.index.isin(d_map.index)].copy()
        drug_targets.index = d_map[drug_targets.index]

        train_network, gene_to_idx = create_dir_resp_ess_tar_net(
            train_tuples, network_perc, gene_ess, drug_targets, common_ess)

        if return_dataset:
            train_data = TensorDataset(
                torch.LongTensor(train_tuples['cell_line'].values),
                torch.LongTensor(train_tuples['drug'].values),
                torch.FloatTensor(train_tuples['response'].values).unsqueeze(1))
        else:
            train_data = train_tuples

        # if CCL is already in training graph, then no need to re-add the CCL
        val_only_ccls = set(val_tuples['cell_line'].unique()) - set(train_tuples['cell_line'].unique()) 
        val_ess = gene_ess.loc[gene_ess.index.isin(val_only_ccls), gene_ess.columns.isin(gene_to_idx.index)].copy()
        val_ess.columns = gene_to_idx[val_ess.columns]

        # calulate number of CCL nodes to add (including those without gene_ess data). 
        # All CCLs whose index < max for training has already been added in the graph 
        n_nodes_to_add = val_tuples['cell_line'].max() - train_tuples['cell_line'].max()
        val_network = append_to_ret_net(train_network, val_ess, n_nodes_to_add, fix_threshold=-1, min_dst_edge=1)

        if return_dataset:
            val_data = TensorDataset(
                torch.LongTensor(val_tuples['cell_line'].values),
                torch.LongTensor(val_tuples['drug'].values),
                torch.FloatTensor(val_tuples['response'].values).unsqueeze(1))
        else:
            val_data = val_tuples

        # add data to the network
        train_network.ndata['features'] = {
                    'drug': torch.FloatTensor(drug_feats.values), 
                    'cell_line': torch.FloatTensor(train_x),
                    'gene': torch.arange(train_network.num_nodes('gene'))}

        val_network.ndata['features'] = {
                    'drug': torch.FloatTensor(drug_feats.values), 
                    'cell_line': torch.FloatTensor(train_val_x),
                    'gene': torch.arange(train_network.num_nodes('gene'))}

        return train_network, val_network, train_data, val_data, train_samples, val_samples,\
            c_map, d_map, normalizer, gene_to_idx

    def _fold_validation(self, hyperparams, seed, train_network, val_network,
        train_data, val_data, tuning, epoch, maxout):
        
        reset_seed(seed)
        train_loader = DataLoader(train_data, batch_size=hyperparams['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=hyperparams['batch_size'], shuffle=False)
        model = Model(hyperparams, train_network.num_nodes('gene'), train_network.etypes)
        trainer = Trainer(hyperparams, train_network, model)

        val_error, metric_names, best_epoch = trainer.fit(
                val_blocks=val_network,
                num_epoch=epoch, 
                train_loader=train_loader, 
                val_loader=val_loader,
                maxout=maxout)

        if not maxout:
            hyperparams['num_epoch'] = int(max(best_epoch, 2))
        
        return val_error, trainer, metric_names, hyperparams

    def fold_validation(self, fold_id, train_folds, val_fold, hyperparams, maxout, pred_fold=[]):
        hp = hyperparams.copy()

        train_network, val_network, train_data, val_data, _,_,_,_, normalizer,_ = self._create_dataset(
            self.labels, self.expr, self.drug_feats, self.drug_targets, self.gene_essentiality, 
            self.FLAGS.network_perc, self.common_ess, train_folds, [val_fold], 
            normalizer=None)

        val_error, trainer, metric_names, best_hyp = self._fold_validation(hp, 
            seed=self.FLAGS.seed, 
            train_network=train_network,
            val_network=val_network,
            train_data=train_data, 
            val_data=val_data, 
            tuning=False, 
            epoch=hp['num_epoch'], 
            maxout=maxout)

        if len(pred_fold) > 0: # actual test fold

            # save logs
            test_metrics = pd.DataFrame(val_error, columns=metric_names)
            test_metrics.to_csv(self.directory + '/fold_%d.csv'%fold_id, index=False)

            # save model
            trainer.model.hyp = best_hyp
            trainer.save_model(self.directory, fold_id, best_hyp)
            pickle.dump(normalizer, open(self.directory + "/normalizer_fold_%d.pkl"%fold_id, "wb"))

            _, test_network, _, test_data, _, test_samples, c_map, d_map, _, g_map = self._create_dataset(
                self.labels, self.expr, self.drug_feats, self.drug_targets, self.gene_essentiality, 
                self.FLAGS.network_perc, self.common_ess, train_folds, pred_fold,
                normalizer=normalizer)

            cell_index = torch.tensor(c_map[test_samples].values, dtype=torch.long)
            prediction_matrix = trainer.predict_matrix(test_network, cell_index)
            prediction_matrix = pd.DataFrame(prediction_matrix, index=test_samples, columns=d_map.index)
            prediction_matrix.to_csv(self.directory + '/cv_prediction_testfold_%d_norm.csv'%fold_id)

            # rescale back to original ranges
            prediction_matrix = prediction_matrix*self.drug_scale[d_map.index] + self.drug_bias[d_map.index]
            prediction_matrix.to_csv(self.directory + '/cv_prediction_testfold_%d.csv'%fold_id)

            if self.FLAGS.savegraph:
                mkdir(self.directory+'/graphs/')
                pickle.dump(test_network, open(self.directory + "/graphs/test_network_fold_%d.pkl"%fold_id, "wb"))
                maps = {'cell_line': c_map, 'drug': d_map, 'gene': g_map}
                pickle.dump(maps, open(self.directory + "/graphs/test_network_map_fold_%d.pkl"%fold_id, "wb"))
            
        return val_error, trainer, metric_names, best_hyp