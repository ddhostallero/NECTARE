import argparse
import pandas as pd
import numpy as np
from kneed import KneeLocator
import os

def main(args):

	all_attr = []
	for i in range(5):
		x = pd.read_parquet('%s/gex_expl/loss_%s_%d.parquet'%(args.folder, args.drug, i))
		baseline = x['baseline']
		gex_attr = x.loc[:, x.columns!='baseline'].T - baseline
		all_attr.append(gex_attr.T)

	all_attr = pd.concat(all_attr)

	if os.path.exists(args.gene_conversion):
		attr = pd.DataFrame(index=all_attr.columns, columns=['gene_symbol', 'avg_delta'])
		conv = pd.read_csv(args.gene_conversion, index_col=0)['gene_symbol']

		missing_genes = set(all_attr.columns) - set(conv.index)
		if len(missing_genes)>0:
			print('Found no gene symbol for the following:')
		for g in missing_genes:
			print(g)
			conv[g] = ""
		attr['gene_symbol'] = conv[all_attr.columns]
	else:
		attr = pd.DataFrame(index=all_attr.columns, columns=['avg_delta'])
		print("Cannot find gene symbol conversion file at %s"%args.gene_conversion)

	attr['avg_delta'] = all_attr.mean()
	attr = attr.sort_values('avg_delta', ascending=False)
	attr['rank'] = np.arange(1, len(attr)+1)

	kneedle = KneeLocator(attr['rank'], attr['avg_delta'], curve='convex', direction='decreasing', S=2)
	thresh = kneedle.knee_y

	attr['top_gene'] = attr['avg_delta'] >= thresh
	out_dir = '%s/gex_expl/attr_summary_%s.csv'%(args.folder, args.drug)
	attr.to_csv(out_dir)
	print('output saved in %s'%out_dir)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--folder", default="results/", help="root directory of the model")
	parser.add_argument("--drug", default="", help="drug to summarize attributes")
	parser.add_argument("--gene_conversion", default="./data/ctrp/ensembl2genesymbol.csv", help="path to gene name conversion file")
	args = parser.parse_args() 

	main(args)