import pandas as pd
import numpy as np 
from abc import abstractmethod
from utils.utils import reset_seed

class BaseCV(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.directory = FLAGS.outroot + "/results/" + FLAGS.folder

    def _filter_folds(self, folds, labels, cell_lines):
        tuples = labels.loc[labels['fold'].isin(folds)]
        samples = list(tuples['cell_line'].unique())
        x = cell_lines.loc[samples]
        return tuples, samples, x

    def nested_cross_validation(self, start_fold, end_fold, hyperparams):
        reset_seed(self.FLAGS.seed)
        final_metrics = None

        for test_fold in range(start_fold, end_fold):
            val_fold = (test_fold+1)%5
            train_folds = [x for x in range(5) if (x != test_fold) and (x != val_fold)]

            if self.FLAGS.mode == 'train':
                print('train_fold(s):', train_folds)
                print('val_fold(s):', val_fold)
                val_error, _, _, best_hyp = self.fold_validation(test_fold, 
                    train_folds, val_fold, hyperparams, maxout=False)

                # === actual test fold ===
                train_folds = train_folds + [val_fold]
                print('train_fold(s):', train_folds)
                print('test_fold:', test_fold)
                test_error, trainer, metric_names, _ = self.fold_validation(test_fold,
                    train_folds, test_fold, best_hyp, maxout=True, pred_fold=[test_fold])

                if test_fold == start_fold:
                    final_metrics = np.zeros((end_fold-start_fold, test_error.shape[1]))
                final_metrics[test_fold-start_fold] = test_error[-1]
                test_metrics = pd.DataFrame(test_error, columns=metric_names)
                test_metrics.to_csv(self.directory + '/fold_%d.csv'%test_fold, index=False)
            
            elif self.FLAGS.mode == 'test':
                train_folds = train_folds + [val_fold]
                print('train_fold(s):', train_folds)
                print('test_fold:', test_fold)
                self.test_on_bulk(test_fold, train_folds, test_fold, pred_fold=[test_fold], pickle_network=True)

            elif self.FLAGS.mode == 'calculate.gex_mse':
                train_folds = train_folds + [val_fold]
                print('train_fold(s):', train_folds)
                print('test_fold:', test_fold)
                self.explain_gene_level(test_fold, train_folds, test_fold, pred_fold=[test_fold])

            elif self.FLAGS.mode == 'explain.graph':
                train_folds = train_folds + [val_fold]
                print('train_fold(s):', train_folds)
                print('test_fold:', test_fold)
                self.explain_graph(test_fold, train_folds, test_fold, pred_fold=[test_fold])#, element='edge')

        if self.FLAGS.mode == 'train':
            final_metrics = pd.DataFrame(final_metrics, columns=metric_names, index=range(start_fold, end_fold))
        return final_metrics

    def main(self):
        test_metrics = self.nested_cross_validation(0, 5, self.hyperparams)