import pandas as pd
import numpy as np
import networkx as nx
import dgl
import torch
import copy

def _filter_edges(tuples, score_col, percentile, dst='drug'):
    sen_edges = pd.DataFrame()
    res_edges = pd.DataFrame()
    for x in tuples[dst].unique():
        x_edges = tuples.loc[tuples[dst]==x]        
        thresh = np.percentile(x_edges[score_col], percentile)
        sen_edges = pd.concat([sen_edges, x_edges.loc[x_edges[score_col]<thresh]])
        thresh = np.percentile(x_edges[score_col], (100-percentile))
        res_edges = pd.concat([res_edges, x_edges.loc[x_edges[score_col]>thresh]])

    return sen_edges, res_edges

def _filter_edges_from_matrix(score_matrix, percentile=1, fix_threshold=None, min_dst_edge=0):

    if fix_threshold is None:
        # connect to bottom percentile
        adj_matrix = pd.DataFrame(index=score_matrix.index, columns=score_matrix.columns)

        for i in score_matrix.index:
            x_edges = score_matrix.loc[i]
            thresh = np.percentile(x_edges, percentile)
            adj_matrix.loc[i] = 1*(x_edges < thresh)
    else:
        if min_dst_edge > 0:
            minimum_graph = score_matrix.apply(lambda x: x.isin(x.nsmallest(min_dst_edge)), axis=1)*1
        else:
            minimum_graph = np.zeros(score_matrix.shape)

        adj_matrix = 1*((1*(score_matrix < fix_threshold) + minimum_graph) > 0) 
    
    edges = adj_matrix.melt(value_name='ess', var_name='col', ignore_index=False)
    edges.index.names=['row']
    edges = edges.loc[edges['ess'] == 1].reset_index()[['row', 'col']]
    return edges


def create_dir_resp_ess_tar_net(tuples, percentile, gene_ess, target, common_ess):
    
    gene_ess = gene_ess.loc[gene_ess.index.isin(tuples['cell_line'].unique())] # so that we don't add ccls that do not exist
    if 'edge_score' in tuples.columns:
        score_col = 'edge_score'
    else:
        score_col = 'response'

    # src = CCL; dst = drug
    drug_sen, drug_res = _filter_edges(tuples, score_col, percentile, 'drug')

    # src = drug; dst = CCL
    ccl_sen, ccl_res = _filter_edges(tuples, score_col, percentile, 'cell_line')

    # gene - drug (undirected)
    target_genes = target.columns
    target = target.melt(value_name='is_target', var_name='gene', ignore_index=False)
    target.index.names = ['drug']
    target = target.reset_index()
    target = target.loc[target['is_target']==1]

    # src = gene; dst = CCL
    local_gene_ess = gene_ess.loc[:,~gene_ess.columns.isin(common_ess)]
    gene_to_ccl = _filter_edges_from_matrix(local_gene_ess, fix_threshold=-1, min_dst_edge=1)
    gene_to_ccl.columns = ['cell_line', 'gene']

    genes_in_network = list(target_genes.union(gene_to_ccl['gene'].unique()))
    gene_to_idx = pd.Series(range(len(genes_in_network)), index=genes_in_network)
    
    # src = CCL; dst = gene
    ccl_to_gene = _filter_edges_from_matrix(gene_ess.loc[:, gene_ess.columns.isin(genes_in_network)].T, percentile)
    ccl_to_gene.columns = ['gene', 'cell_line']

    target['gene'] = target['gene'].replace(gene_to_idx)
    gene_to_ccl['gene'] = gene_to_ccl['gene'].replace(gene_to_idx)
    ccl_to_gene['gene'] = ccl_to_gene['gene'].replace(gene_to_idx)

    print("generated a network with:\n%d sensitive CCL-to-drug edges and %d CCL-to-drug resistant edges "%(len(drug_sen), len(drug_res)))
    print("%d sensitive drug-to-CCL edges and %d drug-to-CCL resistant edges "%(len(ccl_sen), len(ccl_res)))
    print("%d undirected drug-gene edges"%(len(target)))
    print("%d gene-to-CCL edges and %d CCL-to-gene edges "%(len(gene_to_ccl), len(ccl_to_gene)))

    graph_data = {
            ('cell_line', 'is_sensitive', 'drug'): (drug_sen['cell_line'].values, drug_sen['drug'].values),
            ('drug', 'is_effective', 'cell_line'): (ccl_sen['drug'].values, ccl_sen['cell_line'].values),
            ('cell_line', 'is_resistant', 'drug'): (drug_res['cell_line'].values, drug_res['drug'].values),
            ('drug', 'is_ineffective', 'cell_line'): (ccl_res['drug'].values, ccl_res['cell_line'].values),
            ('drug', 'targets', 'gene'): (target['drug'].values, target['gene'].values),
            ('gene', 'is_targeted_by', 'drug'): (target['gene'].values, target['drug'].values),
            ('gene', 'is_essential', 'cell_line'): (gene_to_ccl['gene'].values, gene_to_ccl['cell_line'].values),
            ('cell_line', 'lowest_crispr_of', 'gene'): (ccl_to_gene['cell_line'].values, ccl_to_gene['gene'].values)
            }
    network = dgl.heterograph(graph_data, 
        num_nodes_dict={'cell_line': tuples['cell_line'].max()+1, # this will add CCLs that are not in the graph
                        'drug': tuples['drug'].max()+1,
                        'gene': len(gene_to_idx)})

    return network, gene_to_idx

def append_to_ret_net(network, gene_ess, n_nodes_to_add, fix_threshold=-1, min_dst_edge=1):
    gene_to_ccl = _filter_edges_from_matrix(gene_ess, fix_threshold=fix_threshold, min_dst_edge=min_dst_edge)
    gene_to_ccl.columns = ['cell_line', 'gene']

    val_network = copy.deepcopy(network)

    if n_nodes_to_add > 0:
        val_network = dgl.add_nodes(val_network, n_nodes_to_add, ntype='cell_line')

    if len(gene_to_ccl) > 0:
        val_network = dgl.add_edges(val_network, gene_to_ccl['gene'].values, gene_to_ccl['cell_line'].values, etype='is_essential')

    print(val_network)

    return val_network

def dgl_graph_to_nx(network, maps):
    
    homog = dgl.to_homogeneous(network)
    nxnet = dgl.to_networkx(homog)
    netmap = pd.Series(dtype=str)

    for new_idx, old_idx, ntype in zip(nxnet.nodes, homog.ndata['_ID'],homog.ndata['_TYPE']):
        if ntype == 0:
            ntype = 'cell_line'
        elif ntype == 1:
            ntype = 'drug'
        elif ntype == 2:
            ntype = 'gene'
        netmap[new_idx] = maps[ntype][int(old_idx)]
    nxnet = nx.relabel_nodes(nxnet, netmap)
    return nxnet