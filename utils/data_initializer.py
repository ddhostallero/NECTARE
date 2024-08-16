from utils.utils import mkdir, reset_seed
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_crispr(FLAGS):
    filename = FLAGS.ess_file
    crispr = pd.read_csv(FLAGS.dataroot + filename, index_col=0).T
    common_ess = pd.read_csv(FLAGS.dataroot + FLAGS.common_ess_file)['gene_symbol'].values
    return crispr, common_ess

def load_drug_targets(FLAGS):
    tar = pd.read_csv(FLAGS.dataroot + FLAGS.drug_tar_file, index_col=0)
    return tar

def load_drug_features(FLAGS):
    DRUG_FEATURE_FILE = FLAGS.dataroot + FLAGS.drug_feat_file
    drug_feats = pd.read_csv(DRUG_FEATURE_FILE, index_col=0)
    
    # normalize 
    if drug_feats.min().min() == 0 and drug_feats.max().max() == 1:
        print("Skipping drug feature normalization: binary features for drugs")
    else:
        df = StandardScaler().fit_transform(drug_feats.values)
        drug_feats = pd.DataFrame(df, index=drug_feats.index, columns=drug_feats.columns)

    # remove columns with missing data
    valid_cols = drug_feats.columns[~drug_feats.isna().any()] 
    drug_feats = drug_feats[valid_cols]
        
    return drug_feats

def initialize(FLAGS):#, multitask=False, include_unlabeled=False):
    reset_seed(FLAGS.seed)
    mkdir(FLAGS.outroot + "/results/" + FLAGS.folder)

    LABEL_FILE = FLAGS.dataroot + FLAGS.label_file    
    GENE_EXPRESSION_FILE = FLAGS.dataroot + FLAGS.gex_file
   
    drug_feats = load_drug_features(FLAGS)
    cell_lines = pd.read_csv(GENE_EXPRESSION_FILE, index_col=0).T
    labels = pd.read_csv(LABEL_FILE)
    labels['cell_line'] = labels['cell_line'].astype(str)

    labels = labels.loc[labels['drug'].isin(drug_feats.index)] # use only cell lines with data
    labels = labels.loc[labels['cell_line'].isin(cell_lines.index)] # use only drugs with data

    unlabeled_ccls = cell_lines.loc[~cell_lines.index.isin(labels['cell_line'].unique())]
    
    cell_lines = cell_lines.loc[~cell_lines.index.isin(unlabeled_ccls.index)].copy() # use only cell lines with labels
    cell_lines.index.names = ['cell_line'] # fix melt bug
    drug_feats = drug_feats.loc[drug_feats.index.isin(labels['drug'].unique())]      # use only drugs in labels
    drug_feats.index.names = ['drug'] # fix melt bug

    label_matrix = labels.pivot(index='cell_line', columns='drug', values=FLAGS.response_type)
    label_matrix = label_matrix.loc[cell_lines.index, drug_feats.index]

    print("Using the entire dataset for drug response normalization...")
    ss = StandardScaler() # normalize IC50
    temp = ss.fit_transform(label_matrix.values)
    label_matrix = pd.DataFrame(temp, index=label_matrix.index, columns=label_matrix.columns)

    # add a z-scored response column
    x = label_matrix.melt(value_name='response', ignore_index=False).dropna()
    x = x.reset_index()
    labels = labels.merge(x, on=['cell_line', 'drug'])
            
    if FLAGS.split == 'lpo': # leave pairs out
        labels['fold'] = labels['pair_fold']
    else: # default: leave cell lines out
        labels['fold'] = labels['cl_fold']

    print('tuples per fold:')
    print(labels.groupby('fold').size())
    drug_bias = labels.groupby('drug')[FLAGS.response_type].mean()
    drug_scale = labels.groupby('drug')[FLAGS.response_type].std()
    labels = labels[['drug', 'cell_line', 'response', 'fold']]

    return drug_feats, cell_lines, labels, drug_bias, drug_scale