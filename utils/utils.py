import random
import os
import numpy as np
import pandas as pd
import torch
import dgl
import json

_SEED = 0

def reset_seed(seed=None):
    if seed is not None:
        global _SEED
        _SEED = seed

    torch.manual_seed(_SEED)
    np.random.seed(_SEED)
    random.seed(_SEED)
    dgl.seed(_SEED)
    dgl.random.seed(_SEED)

def mkdir(directory):
    directories = directory.split("/")   

    folder = ""
    for d in directories:
        folder += d + '/'
        if not os.path.exists(folder):
            print('creating folder: %s'%folder)
            os.mkdir(folder)

def reindex_tuples(tuples, drugs, ccls, c_map=None, d_map=None):
    tuples = tuples.copy()

    if d_map is not None:
        no_map = [i for i in drugs if i not in d_map.index] # drugs that do not have existing index
        start_index = d_map.max() + 1
        for i, drug in enumerate(no_map):
            d_map[drug] = i + start_index
    else:
        d_map = pd.Series(np.arange(len(drugs)), index=drugs) 

    tuples['drug_name'] = tuples['drug'].values  
    tuples['drug'] = tuples['drug'].replace(d_map.astype(str)) # remove futurewarning downcast
    tuples['drug'] = tuples['drug'].astype(int)
        
    if c_map is not None:
        no_map = [i for i in ccls if i not in c_map.index] # ccls that don't have exisiting index
        start_index = c_map.max()+1
        for i, ccl in enumerate(no_map):
            c_map[ccl] = i + start_index
    else:
        c_map = pd.Series(np.arange(len(ccls)), index=ccls)    

    tuples['ccl_name'] = tuples['cell_line'].values  
    tuples['cell_line'] = tuples['cell_line'].replace(c_map.astype(str)) # remove futurewarning downcast
    tuples['cell_line'] = tuples['cell_line'].astype(int)

    return tuples, c_map, d_map

def load_hyperparams(hyp_dir):
    with open(hyp_dir) as f:
            hyperparams = f.read()
    return json.loads(hyperparams)