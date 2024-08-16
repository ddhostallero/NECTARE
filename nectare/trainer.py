import numpy as np
import torch
import time
from collections import deque
from scipy.stats import pearsonr, spearmanr
import json


class Trainer:
    def __init__(self, hyperparams, network, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.network = network
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hyperparams['learning_rate'])
        
        self.loss = torch.nn.MSELoss()
        self.val_loss = torch.nn.MSELoss(reduction='sum') # for easier logging
        self.metric_names = metric_names = ['val MSE', 'val pearsonr', 'val spearmanr', 'train MSE']

        self.log_idx = [0,3,2]   # metrics to log on console        
        self.objective_index = 0 # metric to track for early stopping
        if 'obj_idx' in hyperparams.keys():
            self.objective_index = hyperparams['obj_idx']

        self.objective_direction = 1 # -1 if maximize
        if 'obj_dir' in hyperparams.keys():
            self.objective_direction = hyperparams['obj_dir']

        self.model = self.model.to(self.device)

    def _console_log(self, epoch, metrics, idx, elapsed_time):
        s = str(epoch+1)
        for i in idx:
            s += "\t%s:%.4f"%(self.metric_names[i], metrics[i])
        s += "\t%ds"%int(elapsed_time)
        print(s)

    def _train_step(self, train_loader, device):

        g = self.network.to(device)
        cell_feats = g.ndata['features']['cell_line']
        drug_feats = g.ndata['features']['drug']
        gene_index = g.ndata['features']['gene']

        _loss = 0
        self.model.train()
        for (x1, d1, y) in train_loader:
            x1, d1, y = x1.to(device), d1.to(device), y.to(device)
            self.optimizer.zero_grad()
            pred = self.model(g, 
                            drug_feats, 
                            cell_feats, 
                            x1, d1, gene_index)
            loss = self.loss(pred, y)
            _loss += loss

            loss.backward()
            self.optimizer.step()

        # average loss across batches
        return [_loss.item()/len(train_loader)]

    def _validation_step(self, val_blocks, val_loader, device):

        blocks = val_blocks.to(device)
        cell_feats = blocks.ndata['features']['cell_line']
        drug_feats = blocks.ndata['features']['drug']
        gene_index = blocks.ndata['features']['gene']

        self.model.eval()
        val_loss = 0
        preds = []
        ys = []
        with torch.no_grad():
            for (x1, d1, y) in val_loader:
                ys.append(y)
                x1, d1, y = x1.to(device), d1.to(device), y.to(device)
                pred = self.model(blocks, drug_feats, cell_feats, x1, d1, gene_index)
                preds.append(pred)
                val_loss += self.val_loss(pred, y) #((pred - y)**2).sum()


        preds = torch.cat(preds, axis=0).cpu().detach().numpy().reshape(-1)
        ys = torch.cat(ys, axis=0).reshape(-1)
        pcc = pearsonr(ys, preds)[0]
        scc = spearmanr(ys, preds)[0]

        return val_loss.item()/len(ys), pcc, scc

    def fit(self, val_blocks, num_epoch, train_loader, val_loader, 
        tuning=False, maxout=False, deque_size=3):

        start_time = time.time()
        ret_matrix = np.zeros((num_epoch, len(self.metric_names))) # return logs per epoch
        loss_deque = deque([], maxlen=deque_size)                  # moving loss

        # initialize early stopping
        best_epoch = num_epoch
        if self.objective_direction == 1:
            best_loss_avgd = np.inf
        else:
            best_loss_avgd = 0
        count = 0

        # initial (random) performance
        temp = np.zeros(len(self.metric_names))*np.nan
        start_time = time.time()
        val_metrics = self._validation_step(val_blocks, val_loader, self.device)
        temp[:len(val_metrics)] = val_metrics
        self._console_log(-1, temp, self.log_idx, time.time()-start_time) # random performance
        loss_deque.append(val_metrics[self.objective_index])

        # actual fitting
        start_time = time.time()
        for epoch in range(num_epoch):

            train_metrics = self._train_step(train_loader, self.device)
            val_metrics = self._validation_step(val_blocks, val_loader, self.device)

            ret_matrix[epoch] = list(val_metrics) + list(train_metrics)

            # metric for early stopping
            loss_deque.append(val_metrics[self.objective_index])
            loss_avgd = sum(loss_deque)/len(loss_deque)

            # counter for early stopping
            if self.objective_direction*best_loss_avgd > self.objective_direction*loss_avgd:
                best_loss_avgd = loss_avgd
                count = 0
                best_epoch = epoch+1
            else:
                count += 1

            elapsed_time = time.time() - start_time
            start_time = time.time()
            self._console_log(epoch, ret_matrix[epoch], self.log_idx, elapsed_time)

            # stop if no improvement in 10 consecutive epochs (during tuning)
            if ((not maxout) and (count == 10)):
                ret_matrix = ret_matrix[:epoch+1]
                break       

        return ret_matrix, self.metric_names, best_epoch

    def predict_matrix(self, network, cell_index):
        """
        returns a prediction matrix of (N, n_drugs)
        """

        cell_index = cell_index.to(self.device)
        network = network.to(self.device)

        self.model.eval()
        preds = []

        with torch.no_grad():
            ccl_enc,drug_emb,_ = self.model.get_embeddings(network,
                drug_features=network.ndata['features']['drug'],
                cell_features=network.ndata['features']['cell_line'],
                cell_index=cell_index,
                gene_index=network.ndata['features']['gene'])
            # drug_emb = embeddings['drug']

            for i in range(len(cell_index)):
                expr_emb = ccl_enc[i].unsqueeze(0)
                pred = self.model.predict_from_embedding(expr_emb, drug_emb)
                preds.append(pred)

        preds = torch.cat(preds, axis=0).cpu().detach().numpy()
        return preds

    def save_model(self, directory, fold_id, hyp):
        torch.save(self.model.state_dict(), directory+'/model_weights_fold_%d'%fold_id)

        x = json.dumps(hyp)
        f = open(directory+"/model_config_fold_%d.txt"%fold_id,"w")
        f.write(x)
        f.close()