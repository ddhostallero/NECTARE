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

## Data Availability
Preprocessed data can be accessed here: TBA

