import argparse
from utils.utils import mkdir
from nectare.train_pipeline import MainCV
from nectare.explain_pipeline import ExplainCV
import os
import sys
import time
from datetime import timedelta

def save_flags(FLAGS):
    filename = "%s/results/%s/%s_flags.txt"%(FLAGS.outroot, FLAGS.folder, FLAGS.mode)

    with open(filename,'w') as f:
        for arg in vars(FLAGS):
            f.write('--%s=%s\n'%(arg, getattr(FLAGS, arg)))

class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        pass    

if __name__ == '__main__':
    start_time = time.time()

    print("started main function")
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="train", help="[train], test, calculate.gex_mse, explain.graph")
    parser.add_argument("--split", default="lco", help="leave-cells-out [lco], leave-pairs-out (lpo)")
    parser.add_argument("--dataroot", default="./", help="root directory of the data")
    parser.add_argument("--folder", default="nectare/", help="directory of the output")
    parser.add_argument("--outroot", default="./", help="root directory of the output")
    parser.add_argument("--seed", default=0, help="seed number for pseudo-random generation", type=int)    
    parser.add_argument("--network_perc", default=1, help="percentile for network generation", type=float)
    parser.add_argument("--response_type", default="auc", help="[auc], ln_ic50")
    parser.add_argument("--comment", default="", help="comment about the experiment")

    parser.add_argument("--label_file", default="/data/ctrp/ctrp_tuple_labels_folds.csv")
    parser.add_argument("--gex_file", default="/data/ctrp/ccle_log2tpm.csv")
    parser.add_argument("--ess_file", default="/data/ctrp/ccle_chronos_symbol.csv")
    parser.add_argument("--common_ess_file", default="/data/CRISPR_common_essentials_ensembl.csv")
    parser.add_argument("--drug_feat_file", default="/data/ctrp/ctrp_drug_descriptors.csv")
    parser.add_argument("--drug_tar_file", default="/data/ctrp/ctrp_drug_targets.csv")

    parser.add_argument("--drug", default="", help="drug to explain (in explain mode)")

    savegraph_parser = parser.add_mutually_exclusive_group(required=False)
    savegraph_parser.add_argument('--savegraph', dest='savegraph', action='store_true')
    savegraph_parser.add_argument('--no-savegraph', dest='savegraph', action='store_false')
    savegraph_parser.set_defaults(savegraph=True)

    log_parser = parser.add_mutually_exclusive_group(required=False)
    log_parser.add_argument('--logfile', dest='logfile', action='store_true')
    log_parser.add_argument('--no-logfile', dest='logfile', action='store_false')
    parser.set_defaults(logfile=True)

    args = parser.parse_args() 
    mkdir(args.outroot + "/results/" + args.folder)

    if args.logfile:
        sys.stdout = Logger(args.outroot + "/results/" + args.folder + "/" + args.mode + "_log.txt")

    save_flags(args)

    if args.mode == 'train':
        main_fxn = MainCV(args)
    elif args.mode in ['calculate.gex_mse', 'explain.graph']:
        main_fxn = ExplainCV(args)
    else:
        print("Invalid mode: %s"%args.mode)
        exit()
    main_fxn.main()

    elapsed_time = time.time() - start_time
    elapsed_time = str(timedelta(seconds=elapsed_time))
    print("ELAPSED TIME: %s"%elapsed_time)

