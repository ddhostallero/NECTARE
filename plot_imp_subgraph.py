import argparse
import os
import pandas as pd
import numpy as np
import networkx as nx
from kneed import KneeLocator
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

categ_color = {
    'is_sensitive':'#e60049', 
    'is_resistant':'#0bb4ff', 
    'targets':'#e6d800', 
    'lowest_crispr_of':'#50e991', 
    'is_effective':'#e60049',
    'is_ineffective':'#0bb4ff',
    'is_targeted_by':'#e6d800',
    'is_essential':'#50e991'}

color = {'cell_line':'tab:blue', 'drug':'tab:orange', 'gene':'tab:green'}

def plot(drug, edge_scores, node_scores, seed, iterations, savefig=True, suffix='', col='clip_score'):
    plt.figure(frameon=False)
    net = nx.DiGraph()
    for i in node_scores.index:
        net.add_node(i, ntype=node_scores.loc[i, 'ntype'], delta=node_scores.loc[i, col])

    for i in edge_scores.index:
        e = edge_scores.loc[i]
        net.add_edge(e['src'], e['dst'], etype=e['etype'], 
                     delta=e[col], viz=e['delta_viz'],
                    edge_color=e['color'])#, lstyle=e['lstyle'])
        
    fig, ax = plt.subplots(figsize=(12,12))

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='drug',markerfacecolor='tab:orange', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='cell line',markerfacecolor='tab:blue', markersize=10),     
        Line2D([0], [0], marker='o', color='w', label='gene',markerfacecolor='tab:green', markersize=10),
        Line2D([0], [0],  color=categ_color['is_resistant'], label='resistance', lw=2, ls='solid'),
        Line2D([0], [0],  color=categ_color['is_sensitive'], label='sensitivity', lw=2, ls='solid'),
        Line2D([0], [0],  color=categ_color['targets'], label='target', lw=2, ls='solid'),
        Line2D([0], [0],  color=categ_color['is_essential'], label='essentiality', lw=2, ls='solid'),
    ]

    weight = [i['viz']*4 for i in dict(net.edges).values()]
    edge_color = [i['edge_color'] for i in dict(net.edges).values()]

    node_color = node_scores.loc[list(net.nodes), 'color']
    node_size = node_scores.loc[list(net.nodes), 'delta_viz']*700
    node_size[drug]=2000+node_size.max()
    
    pos = nx.nx_agraph.graphviz_layout(net, prog="fdp")

    nx.draw_networkx_nodes(net, pos, ax=ax, node_color=node_color, node_size=node_size)
    nx.draw_networkx_labels(net, pos)
    nx.draw_networkx_edges(net, pos, arrows=True, width=weight, ax=ax, 
                           node_size=node_size, edge_color=edge_color, min_target_margin=15,
                           connectionstyle='arc3,rad=0.1', alpha=0.9)

    plt.legend(handles=legend_elements, loc='upper right', frameon=False)
    
def logistic(x, alpha=700):
    return 1/(1+np.exp(-1*alpha*x))

def process_drug_double_thresh(drug, ns, es, ccl_name, seed=777, iterations=50, implicate_edges=True):
    ns = ns.loc[(ns['clip_score']>0)].copy()
    if ccl_name is not None:
        ns['node'] = ns['node'].replace(ccl_name)
    ns = ns.set_index('node')
    ns.loc[drug] = ['drug', ns['clip_score'].max()]
    
    if ccl_name is not None:
        es['src'] = es['src'].replace(ccl_name)
        es['dst'] = es['dst'].replace(ccl_name)
    es = es.loc[(es['src'].isin(ns.index)) & (es['dst'].isin(ns.index))].copy()
    
    ns['delta_viz'] = ns['clip_score'].apply(logistic, alpha=100)
    es['delta_viz'] = es['clip_score'].apply(logistic, alpha=500)
    
    ns['color'] = ns['ntype'].replace(color)         # scale for visualization
    es['color'] = es['etype'].replace(categ_color)   # scale for visualization
    
    # find threshold for top nodes
    y = ns.loc[ns.index!=drug, 'clip_score'].sort_values(ascending=False)
    kneedle = KneeLocator(np.arange(len(y)),y, curve='convex', direction='decreasing',S=2)
    thresh = kneedle.knee_y
    
    # get top nodes
    top_nodes_filt1 = ns.loc[ns['clip_score']>=thresh].index
    ns.loc[top_nodes_filt1, 'delta_viz'] = ns.loc[top_nodes_filt1, 'delta_viz']*1.5
    
    # get edges that point to the top nodes (i.e., 2-hop edges to the drug)
    twohop_edges = es.loc[(es['dst'].isin(top_nodes_filt1)) & (es['dst']!=drug)]
    
    # find threshold for 2-hop edges
    y = twohop_edges['clip_score'].sort_values(ascending=False)
    kneedle = KneeLocator(np.arange(len(y)),y, curve='convex', direction='decreasing',S=2)
    thresh = kneedle.knee_y
    
    # get top edges
    top_edges = twohop_edges.loc[twohop_edges['clip_score']>=thresh]               # top 2-hop edges (thresholded)
    top_edges_1hop = es.loc[(es['src'].isin(top_nodes_filt1)) & (es['dst']==drug)] # top 1-hop edges (implied)
    top_edges = pd.concat([top_edges, top_edges_1hop])

    # get top nodes
    # either from the first filter (thresholded) or from the top 2-hop edges (implied)
    top_nodes = ns.loc[(ns.index.isin(top_edges['src'].unique())) | (ns.index.isin(top_edges['dst'].unique()))].copy()
    
    # get reverse edges of top edges, but are not top edges themselves
    top_edge_inverse = []
    for i in top_edges.index:
        temp = es.loc[(~es.index.isin(top_edges.index))& # not a top edge
                      (es['src']==es.loc[i, 'dst'])&     # inverse of the current edge
                      (es['dst']==es.loc[i, 'src'])].index
        if len(temp)>0:
            top_edge_inverse+=list(temp)
    
    # include edges that exist between top nodes, except reverse non-top edges
    if implicate_edges:
        top_edges_idx = top_edges.index
        top_edges = es.loc[(~es.index.isin(top_edge_inverse))& # not a reverse edge of a top edge
                           (es['src'].isin(top_nodes.index)) & # implied 2-hop edges 
                           (es['dst'].isin(top_nodes.index))].copy()

    # plot the subgraph
    plot(drug, top_edges, top_nodes, seed, iterations)
    
    # add annotations to the node scores
    ns = ns[['ntype', 'clip_score']]
    ns.loc[drug, 'clip_score'] = np.nan # the drug of interest does not have a score
    ns['rank'] = ns['clip_score'].rank(ascending=False)
    ns['top_node'] = ns.index.isin(top_nodes.index)
    edge_implied = []
    for i in ns.index:
        if i in top_nodes.index:
            edge_implied.append(i not in top_nodes_filt1)
        else:
            edge_implied.append(np.nan)
    ns['edge_implied'] = edge_implied
    
    # add annotations to the edge scores
    es = es[['src','dst','etype','clip_score']].copy()
    es['rank'] = es['clip_score'].rank(ascending=False)
    es['top_edge'] = es.index.isin(top_edges.index)
    
    if implicate_edges:
        node_implied = []
        for i in es.index:
            if i in top_edges.index:
                node_implied.append(i not in top_edges_idx)
            else:
                node_implied.append(np.nan)        
        es['node_implied'] = node_implied
    
    return es.sort_values('rank'), ns.sort_values('rank')


def main(args):
    node_scores = []
    edge_scores = []
    for i in range(5):
        ns = pd.read_parquet('%s/node_expl/node_score_%d.parquet'%(args.folder, i))[[args.drug, 'ntype']].dropna()
        ns['clip_score'] = np.clip(ns[args.drug], a_min=0, a_max=None)
        ns.index.names = ['node']
        node_scores.append(ns.reset_index())

        es = pd.read_parquet('%s/edge_expl/edge_score_%d.parquet'%(args.folder, i))[['src','dst','etype',args.drug]].dropna()
        es['clip_score'] = np.clip(es[args.drug], a_min=0, a_max=None)
        edge_scores.append(es.reset_index())

    node_scores = pd.concat(node_scores)
    edge_scores = pd.concat(edge_scores)
    
    # summarize across folds
    node_scores = node_scores.groupby(['node','ntype'])['clip_score'].mean().reset_index()
    edge_scores = edge_scores.groupby(['src','dst','etype'])['clip_score'].mean().reset_index()

    if args.ccl_name!="" and os.path.exists(args.ccl_name):
        ccls = pd.read_csv(args.ccl_name, index_col=0)
        ccl_name = ccls['CCLE_Name'].astype(str).apply(lambda x: x[:12])
    else:
        print("No CCL medata found. Using IDs for labels.")
        ccl_name=None

    es,ns = process_drug_double_thresh(
            drug=args.drug, ns=node_scores, 
            es=edge_scores, ccl_name=ccl_name,
            implicate_edges=args.imp_edge)

    out_dir = '%s/importance_subgraphs/'%args.folder
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    plt.savefig('%s/%s_subgraph.pdf'%(out_dir, args.drug), dpi=300)
    es.to_csv('%s/%s_summary_edge_scores.csv'%(out_dir, args.drug))
    ns.to_csv('%s/%s_summary_node_scores.csv'%(out_dir, args.drug))

    print("Results saved in %s"%out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="results/", help="root directory of the model")
    parser.add_argument("--drug", default="", help="drug to summarize attributes")
    parser.add_argument("--ccl_name", default="data/ctrp/ccle_name.csv", help="CCL metadata to map to CCL names instead of ID")

    impedge_parser = parser.add_mutually_exclusive_group(required=False)
    impedge_parser.add_argument('--imp_edge', dest='imp_edge', action='store_true')
    impedge_parser.add_argument('--no-imp_edge', dest='imp_edge', action='store_false')
    parser.set_defaults(imp_edge=True)

    args = parser.parse_args() 
    main(args)