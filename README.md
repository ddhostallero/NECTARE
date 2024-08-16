# NECTARE Project
### Knowledge Embeddings of Compounds using Targets, Response, and Essentiality


## Running NECTARE
To run NECTARE, use the following command:
```
python main.py
```

## Additional Parameters

- `--split`: the type of data-splitting to use (`lco` or `lpo`, default: `lco`)
- `--dataroot`: the root directory of your data (default: `./`)
- `--outroot`: the root directory of your outputs (default: `./`)
- `--folder`: output folder if in `train` mode or model location if other modes (`<outroot>/results/<folder>`) 
- `--mode`: run mode (`train`, `test`, `calculate.gex_mse`, or `explain.graph`, default: `train`) 
- `--seed`: the seed number for 5-fold CV (default: 0)
- `--network_perc`: percentile used for the bipartite graph threshold (default: 1)
- `--response_type`: type of drug response (`auc` of `ln_ic50`, default: `auc`)
- `--label_file`: path to the label and fold-splitting file
- `--gex_file`: path to the GEx file
- `--ess_file`: path to the gene essentiality scores file
- `--common_ess_file`: path to the common essential genes file
- `--drug_feat_file`: path to the drug feature file
- `--drug_tar_file`: path to the drug target file
- `--drug`: drug for calculate.gex_mse mode (not used for other modes)

## Running the GEx-level explainer

The GEx-level can only run for one drug at a time. First, calculate the marginal MSEs for a specific drug.
```
python main.py --mode=calculate.gex_mse --folder=<directory of the model> --drug=<drug>
```

This will output the marginal MSEs in `<outroot>/results/<folder>/gex_expl/loss_<drug>_<fold>.csv`.
**NOTE:** This is not the attribution score. 

To calculate the attribution scores, run the following:

```
python gex_attr_rank.py --folder=<directory of the model> --drug=<drug>
```
This will output the summarized attributions (summarized across samples and folds) for `<drug>` in `<outroot>/results/<folder>/gex_expl/attr_summary_<drug>.csv`.
 
## Running the Graph-level explainer

The Graph-level explainer calculates the node scores and edge scores for all drugs. First, calculate the raw scores:
```
python main.py --mode=explain.graph --folder=<directory of the model>
```
This will output the node/edge-removal scores in `<outroot>/results/<folder>/node_expl/` and `<folder>/edge_expl/`.


To plot the importance subgraph for a specific drug:
```
python plot_imp_subgraph.py --folder=<directory of the model> --drug=<drug>
```
This will output the importance subgraph plot in `<outroot>/results/<folder>/importance_subgraphs`

## Data Availability
Preprocessed data can be accessed here: TBA

