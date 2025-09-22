import os
import time
import json
import glob
import sys
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn


from .homophily import *
from .model import Model
from .utils.generic_utils import to_cuda
from .utils.data_utils import prepare_datasets, DataStream, vectorize_input
from .utils import Timer, DummyLogger, AverageMeter
from .utils import constants as Constants
from .layers.common import dropout
from .layers.anchor import sample_anchors, batch_sample_anchors, batch_select_from_tensor, compute_anchor_adj

import torchmetrics
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import f1_score, precision_score, recall_score



class ModelHandler(object):
    """High level model_handler that trains/validates/tests the network,
    tracks and logs metrics.
    """
    def __init__(self, config, args, data_index):
        # Evaluation Metrics:
        self._train_loss = AverageMeter()
        self._dev_loss = AverageMeter()
        if config['task_type'] == 'classification':
            self._train_metrics = {'nloss': AverageMeter(),
                                'acc': AverageMeter()}
            self._dev_metrics = {'nloss': AverageMeter(),
                                'acc': AverageMeter()}
        elif config['task_type'] == 'regression':
            self._train_metrics = {'nloss': AverageMeter(),
                                'r2': AverageMeter()}
            self._dev_metrics = {'nloss': AverageMeter(),
                                'r2': AverageMeter()}
        else:
            raise ValueError('Unknown task_type: {}'.format(config['task_type']))


        self.logger = DummyLogger(config, dirname=config['out_dir'], pretrained=config['pretrained'])
        self.dirname = self.logger.dirname
        if not config['no_cuda'] and torch.cuda.is_available():
            print('[ Using CUDA ]')
            self.device = torch.device('cuda' if config['cuda_id'] < 0 else 'cuda:%d' % config['cuda_id'])
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        config['device'] = self.device


        seed = config.get('seed', 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device:
            torch.cuda.manual_seed(seed)


        datasets = prepare_datasets(config, args, data_index)
        



        y = datasets['labels'].detach().cpu()
        self.logger.write_to_file(f"\n************************************ Data info ******************************************")
   
        self.logger.write_to_file(f"Data split Ratio : {args['ratio']},  Train : Valid : Test = {len(datasets['idx_train'])} : {len(datasets['idx_val'])} : {len(datasets['idx_test'])}")
        for i in range(len(np.unique(y))):
                self.logger.write_to_file(f"Number of samples in class {i}:  {len(np.where(y==i)[0])} ")

        idx_train_ = datasets['idx_train']
        if args['fake_samples'] == True:
            self.logger.write_to_file(f"-------Fake rate : {args['fake_rate']}")
            idx_train_ = datasets['idx_train'].detach().cpu()
            for i in range(len(np.unique(y[idx_train_]))):
                self.logger.write_to_file(f"Number of samples in train class {i}:  {len(np.where(y[idx_train_]==i)[0])} ")    
        if args['smote'] == True:
            if args['updaters']:
                self.logger.write_to_file(f"------- SMOTE + nd-prompt -------")
          
            else:    
                self.logger.write_to_file(f"------- SMOTE -------")
            idx_train_ = datasets['idx_train'].detach().cpu()
            for i in range(len(np.unique(y[idx_train_]))):
                self.logger.write_to_file(f"Number of samples in train class {i}:  {len(np.where(y[idx_train_]==i)[0])} ")
                
        # Prepare datasets
        if config['data_type'] in ('network', 'uci'):
            config['num_feat'] = datasets['features'].shape[-1]
            config['num_class'] = datasets['labels'].max().item() + 1
            if args['updaters']==True :
                config['fake_idx'] = datasets['fake_idx']


            # Initialize the model
            self.model = Model(config, train_set=datasets.get('train', None))
           
            self.model.network = self.model.network.to(self.device)
            self._n_test_examples = datasets['idx_test'].shape[0]
            self.run_epoch = self._scalable_run_whole_epoch if config.get('scalable_run', False) else self._run_whole_epoch

            self.train_loader = datasets
            self.dev_loader = datasets
            self.test_loader = datasets

        else:
            train_set = datasets['train']
            dev_set = datasets['dev']
            test_set = datasets['test']

            config['num_class'] = max([x[-1] for x in train_set + dev_set + test_set]) + 1

            self.run_epoch = self._run_batch_epoch

            # Initialize the model
            self.model = Model(config, train_set=datasets.get('train', None))
            self.model.network = self.model.network.to(self.device)


            self._n_train_examples = 0
            if train_set:
                self.train_loader = DataStream(train_set, self.model.vocab_model.word_vocab, config=config, isShuffle=True, isLoop=True, isSort=True)
                self._n_train_batches = self.train_loader.get_num_batch()
            else:
                self.train_loader = None

            if dev_set:
                self.dev_loader = DataStream(dev_set, self.model.vocab_model.word_vocab, config=config, isShuffle=False, isLoop=True, isSort=True)
                self._n_dev_batches = self.dev_loader.get_num_batch()
            else:
                self.dev_loader = None

            if test_set:
                self.test_loader = DataStream(test_set, self.model.vocab_model.word_vocab, config=config, isShuffle=False, isLoop=False, isSort=True, batch_size=config['batch_size'])
                self._n_test_batches = self.test_loader.get_num_batch()
                self._n_test_examples = len(test_set)
            else:
                self.test_loader = None



        self.config = self.model.config
        self.is_test = False
        
        ####
        self.dataset_name =config['dataset_name']
        self.fake_samples = args['fake_samples']
        self.updaters = args['updaters']


    def train(self,fold):
        if self.train_loader is None or self.dev_loader is None:
            print("No training set or dev set specified -- skipped training.")
            return


        self.is_test = False
        timer = Timer("Train")
        self._epoch = self._best_epoch = 0

        self._best_metrics = {}
        for k in self._dev_metrics:
            self._best_metrics[k] = -float('inf')
        self._reset_metrics()


        
        while self._stop_condition(self._epoch, self.config['patience']):
            self._epoch += 1


            # Train phase
            if self._epoch % self.config['print_every_epochs'] == 0:
                format_str = "\n>>> Train Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs'])
                print(format_str)
                self.logger.write_to_file(format_str)

            self.run_epoch(self.train_loader, fold, training=True, verbose=self.config['verbose'])
            
            if self._epoch % self.config['print_every_epochs'] == 0:
                format_str = "Training Epoch {} -- Loss: {:0.5f}".format(self._epoch, self._train_loss.mean())
                format_str += self.metric_to_str(self._train_metrics)
                train_epoch_time_msg = timer.interval("Training Epoch {}".format(self._epoch))
                self.logger.write_to_file(train_epoch_time_msg + '\n' + format_str)
                print(format_str)
                format_str = "\n>>> Validation Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs'])
                print(format_str)
                self.logger.write_to_file(format_str)



            # Validation phase 
            dev_output, dev_gold = self.run_epoch(self.dev_loader,fold, training=False, verbose=self.config['verbose'],
                                out_predictions=self.config['out_predictions'])
            idx = self.test_loader['idx_val']
            dev_output, dev_gold = dev_output[idx], dev_gold[idx]
            


            
            if self.config['out_predictions']:
                dev_metric_score = self.model.score_func(dev_gold, dev_output)
            else:
                dev_metric_score = None

 
            if self._epoch % 100 == 0:
                
                format_str = "Validation Epoch {} -- Loss: {:0.5f}".format(self._epoch, self._dev_loss.mean())
                format_str += self.metric_to_str(self._dev_metrics)
                if dev_metric_score is not None:
                    format_str += '\n Dev score: {:0.5f}'.format(dev_metric_score)
                dev_epoch_time_msg = timer.interval("Validation Epoch {}".format(self._epoch))
                self.logger.write_to_file(dev_epoch_time_msg + '\n' + format_str)
                print(format_str)

            if not self.config['data_type'] in ('network', 'uci', 'text'):
                self.model.scheduler.step(self._dev_metrics[self.config['eary_stop_metric']].mean())


            if self.config['eary_stop_metric'] == self.model.metric_name and dev_metric_score is not None:
                cur_dev_score = dev_metric_score
            else:
                cur_dev_score = self._dev_metrics[self.config['eary_stop_metric']].mean()

   
            if self._best_metrics[self.config['eary_stop_metric']] < cur_dev_score:
                self._best_epoch = self._epoch
                for k in self._dev_metrics:
                    self._best_metrics[k] = self._dev_metrics[k].mean()

                if dev_metric_score is not None:
                    self._best_metrics[self.model.metric_name] = dev_metric_score

                if self.config['save_params']:
                    self.model.save(self.dirname)
                    if self._epoch % self.config['print_every_epochs'] == 0:
                        format_str = 'Saved model to {}'.format(self.dirname)
                        self.logger.write_to_file(format_str)
                        print(format_str)

                if self._epoch % self.config['print_every_epochs'] == 0:
                    format_str = "!!! Updated: " + self.best_metric_to_str(self._best_metrics)
                    self.logger.write_to_file(format_str)
                    print(format_str)

            self._reset_metrics()


        timer.finish()

        format_str = "Finished Training: {}\nTraining time: {}".format(self.dirname, timer.total) + '\n' + self.summary()
        print(format_str)
        self.logger.write_to_file(format_str)
        return self._best_metrics


    def test(self, args, fold):
        if self.test_loader is None:
            print("No testing set specified -- skipped testing.")
            return
        
        idx = self.test_loader['idx_test']
 
        # Restore best models
        print('Restoring best model')
        self.model.init_saved_network(self.dirname)
        self.model.network = self.model.network.to(self.device)

        self.is_test = True
        self._reset_metrics()
        timer = Timer("Test")
        for param in self.model.network.parameters():
            param.requires_grad = False


        output, gold, adj_homo = self.run_epoch(self.test_loader, fold,training=False, verbose=0,
                                 out_predictions=self.config['out_predictions'])
        output, gold = output[idx], gold[idx]
        

    
        _, pred = torch.max(output,1)
        
        from torchmetrics.classification import BinaryAccuracy
        from torcheval.metrics.functional import multiclass_confusion_matrix
        CM = multiclass_confusion_matrix(output, gold, self.config['num_class'])
        print(CM)
        
        metric = torchmetrics.AUROC(task='multiclass', num_classes=self.config['num_class'])

        
        auc = metric(output, gold)
        f1 = f1_score(gold.cpu(), pred.cpu())
        precision = precision_score(gold.cpu(), pred.cpu())
        recall = recall_score(gold.cpu(), pred.cpu())
        
        acc_met = BinaryAccuracy()
        acc = acc_met(gold.cpu(), pred.cpu())


        
        metrics = self._dev_metrics
        format_str = "[test] | test_exs = {} | step: [{} / {}]".format(
            self._n_test_examples, 1, 1)
        format_str += self.metric_to_str(metrics)
       

        if self.config['out_predictions']:
            test_score = self.model.score_func(gold, output)
            format_str += '\nFinal score on the testing set: {:0.5f}\n'.format(test_score)
        else:
            test_score = None

        print(format_str)
        self.logger.write_to_file(format_str)
        timer.finish()

        format_str = "Finished Testing: {}\nTesting time: {}".format(self.dirname, timer.total)
        print(format_str)
        self.logger.write_to_file(format_str)
        self.logger.write_to_file(f"ROC-AUC : {auc}") 
        self.logger.write_to_file(f"precision score : {precision}") 
        self.logger.write_to_file(f"recall score : {recall}") 
        self.logger.write_to_file(f"f1 score : {f1}") 
        self.logger.write_to_file(f'{CM}') 
        self.logger.close()


        test_metrics = {}
        for k in metrics:
            test_metrics[k] = metrics[k].mean()

        if test_score is not None:
            test_metrics[self.model.metric_name] = test_score
        return acc.detach().cpu(),auc.detach().cpu(),precision, recall, f1, adj_homo

    # 훈련!
    def _run_whole_epoch(self, data_loader,fold, training=True, verbose=None, out_predictions=False):
        '''BP after all iterations'''
        mode = "train" if training else ("test" if self.is_test else "dev")
        self.model.network.train(training)

     
        init_adj, features, labels = data_loader['adj'], data_loader['features'], data_loader['labels']
        
        if self.updaters==True or self.fake_samples==True: 
            fake_idx = data_loader['fake_idx']

        if mode == 'train':
            idx = data_loader['idx_train']
        elif mode == 'dev':
            idx = data_loader['idx_val']
        else:
            idx = data_loader['idx_test']
            
 

        network = self.model.network
        
        features = F.dropout(features, network.config.get('feat_adj_dropout', 0), training=network.training)
        init_node_vec = features
        
 
        cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner, init_node_vec, network.graph_skip_conn, graph_include_self=network.graph_include_self, init_adj=init_adj)



        if self.config['graph_learn'] and self.config.get('max_iter', 10) > 0:
            cur_raw_adj = F.dropout(cur_raw_adj, network.config.get('feat_adj_dropout', 0), training=network.training)
        cur_adj = F.dropout(cur_adj, network.config.get('feat_adj_dropout', 0), training=network.training)
        # print("A1 size",cur_adj.shape)

        if network.graph_module == 'gat':
 
            assert self.config['graph_learn'] is False and self.config.get('max_iter', 10) == 0
            node_vec = network.encoder(init_node_vec, init_adj)
            output = F.log_softmax(node_vec, dim=-1)


            

        elif network.graph_module == 'graphsage':
            assert self.config['graph_learn'] is False and self.config.get('max_iter', 10) == 0
            # Convert adj to DGLGraph
            import dgl
            from scipy import sparse
            binarized_adj = sparse.coo_matrix(init_adj.detach().cpu().numpy() != 0)
            dgl_graph = dgl.DGLGraph(binarized_adj)

            node_vec = network.encoder(dgl_graph, init_node_vec)
            output = F.log_softmax(node_vec, dim=-1)

        else:
            node_vec = torch.relu(network.encoder.graph_encoders[0](init_node_vec, cur_adj))
            node_vec = F.dropout(node_vec, network.dropout, training=network.training)


            
            # Add mid GNN layers
            for it, encoder in enumerate(network.encoder.graph_encoders[1:-1]):
                node_vec = torch.relu(encoder(node_vec, cur_adj))
                node_vec = F.dropout(node_vec, network.dropout, training=network.training)

                

            # BP to update weights
            output = network.encoder.graph_encoders[-1](node_vec, cur_adj)
                  
            output = F.log_softmax(output, dim=-1)


        score = self.model.score_func(labels[idx], output[idx])
        loss1 = self.model.criterion(output[idx], labels[idx])

        
        if self.config['graph_learn'] and self.config['graph_learn_regularization']:
            loss1 += self.add_graph_loss(cur_raw_adj, init_node_vec)

        first_raw_adj, first_adj = cur_raw_adj, cur_adj

        if not mode == 'test':
            if self._epoch > self.config.get('pretrain_epoch', 0):
                max_iter_ = self.config.get('max_iter', 10) # Fine-tuning
                if self._epoch == self.config.get('pretrain_epoch', 0) + 1:
                    for k in self._dev_metrics:
                        self._best_metrics[k] = -float('inf')

            else:
                max_iter_ = 0 # Pretraining
        else:
            max_iter_ = self.config.get('max_iter', 10)


        if training:
            eps_adj = float(self.config.get('eps_adj', 0)) # cora: 5.5e-8, cora w/o input graph: 1e-8, citeseer w/o input graph: 1e-8, wine: 2e-5, cancer: 2e-5, digtis: 2e-5
        else:
            eps_adj = float(self.config.get('test_eps_adj', self.config.get('eps_adj', 0)))


        pre_raw_adj = cur_raw_adj
        pre_adj = cur_adj

        loss = 0
        iter_ = 0

        while self.config['graph_learn'] and (iter_ == 0 or diff(cur_raw_adj, pre_raw_adj, first_raw_adj).item() > eps_adj) and iter_ < max_iter_:
            iter_ += 1
            pre_adj = cur_adj
            pre_raw_adj = cur_raw_adj
    
            cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner2, node_vec, network.graph_skip_conn, graph_include_self=network.graph_include_self, init_adj=init_adj)

            update_adj_ratio = self.config.get('update_adj_ratio', None)
            if update_adj_ratio is not None:
                cur_adj = update_adj_ratio * cur_adj + (1 - update_adj_ratio) * first_adj


            ##### helper nodes + updater
            if self.updaters:
                zero_matrix = torch.zeros((features.shape[0]-fake_idx.shape[0],self.config['num_feat'])).to(self.device)
                node_vec = torch.relu(network.encoder.graph_encoders[0](network.encoder.update_zero_node(init_node_vec, zero_matrix), cur_adj))
         

            else:
                node_vec_ = network.encoder.graph_encoders[0](init_node_vec, cur_adj)
                node_vec = torch.relu(node_vec_)
            node_vec = F.dropout(node_vec, network.dropout, training=network.training)



            # Add mid GNN layers
            for encoder in network.encoder.graph_encoders[1:-1]:
                node_vec_ = encoder(node_vec_, cur_adj)
    
                node_vec = torch.relu(node_vec_)
                node_vec = F.dropout(node_vec, network.dropout, training=network.training)
  
                
     

            # BP to update weights
            output = network.encoder.graph_encoders[-1](node_vec, cur_adj)###
            output = F.log_softmax(output, dim=-1)
            
            
   
            
            
            score = self.model.score_func(labels[idx], output[idx])
            loss += self.model.criterion(output[idx], labels[idx])


            if self.config['graph_learn'] and self.config['graph_learn_regularization']:
                loss += self.add_graph_loss(cur_raw_adj, init_node_vec)

            if self.config['graph_learn'] and not self.config.get('graph_learn_ratio', None) in (None, 0):
                loss += SquaredFrobeniusNorm(cur_raw_adj - pre_raw_adj) * self.config.get('graph_learn_ratio')

       
        
        if mode == 'test' and self.config.get('out_raw_learned_adj_path', None):
            out_raw_learned_adj_path = os.path.join(self.dirname, self.config['out_raw_learned_adj_path'])
            out_raw_learned_adj_path = os.path.join(self.dirname, f"{self.config['out_raw_learned_adj_path']}_iter{fold}.npy")
            np.save(out_raw_learned_adj_path, cur_raw_adj.cpu())
            print('Saved raw_learned_adj to {}'.format(out_raw_learned_adj_path))

        if iter_ > 0:
            loss = loss / iter_ + loss1
        else:
            loss = loss1
        
        adj_homo=None 
                
                
       
        
        if training:
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.clip_grad()
            self.model.optimizer.step()

        self._update_metrics(loss.item(), {'nloss': -loss.item(), self.model.metric_name: score}, 1, training=training)
        if mode == 'test':
            return output, labels,   adj_homo
        else:
            return output, labels
        


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def self_report(self, step, mode='train'):
        if mode == "train":
            format_str = "[train-{}] step: [{} / {}] | loss = {:0.5f}".format(
                self._epoch, step, self._n_train_batches, self._train_loss.mean())
            format_str += self.metric_to_str(self._train_metrics)
        elif mode == "dev":
            format_str = "[predict-{}] step: [{} / {}] | loss = {:0.5f}".format(
                    self._epoch, step, self._n_dev_batches, self._dev_loss.mean())
            format_str += self.metric_to_str(self._dev_metrics)
        elif mode == "test":
            format_str = "[test] | test_exs = {} | step: [{} / {}]".format(
                    self._n_test_examples, step, self._n_test_batches)
            format_str += self.metric_to_str(self._dev_metrics)
        else:
            raise ValueError('mode = {} not supported.' % mode)
        return format_str

    def plain_metric_to_str(self, metrics):
        format_str = ''
        for k in metrics:
            format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k])
        return format_str

    def metric_to_str(self, metrics):
        format_str = ''
        for k in metrics:
            format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k].mean())
        return format_str

    def best_metric_to_str(self, metrics):
        format_str = '\n'
        for k in metrics:
            format_str += '{} = {:0.5f}\n'.format(k.upper(), metrics[k])
        return format_str

    def summary(self):
        start = "\n<<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        info = "Best epoch = {}; ".format(self._best_epoch) + self.best_metric_to_str(self._best_metrics)
        end = " <<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        return "\n".join([start, info, end])

    def _update_metrics(self, loss, metrics, batch_size, training=True):
        if training:
            if loss:
                self._train_loss.update(loss)
            for k in self._train_metrics:
                if not k in metrics:
                    continue
                self._train_metrics[k].update(metrics[k], batch_size)
        else:
            if loss:
                self._dev_loss.update(loss)
            for k in self._dev_metrics:
                if not k in metrics:
                    continue
                self._dev_metrics[k].update(metrics[k], batch_size)

    def _reset_metrics(self):
        self._train_loss.reset()
        self._dev_loss.reset()

        for k in self._train_metrics:
            self._train_metrics[k].reset()
        for k in self._dev_metrics:
            self._dev_metrics[k].reset()

    def _stop_condition(self, epoch, patience=10):
        """
        Checks have not exceeded max epochs and has not gone patience epochs without improvement.
        """
        no_improvement = epoch >= self._best_epoch + patience
        exceeded_max_epochs = epoch >= self.config['max_epochs']
        return False if exceeded_max_epochs or no_improvement else True


    def add_graph_loss(self, out_adj, features):
        # Graph regularization
        graph_loss = 0
        L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
        graph_loss += self.config['smoothness_ratio'] * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
        ones_vec = to_cuda(torch.ones(out_adj.size(-1)), self.device)
        graph_loss += -self.config['degree_ratio'] * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(out_adj, ones_vec.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)).squeeze() / out_adj.shape[-1]
        graph_loss += self.config['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        return graph_loss




def diff(X, Y, Z):
    assert X.shape == Y.shape
    diff_ = torch.sum(torch.pow(X - Y, 2))
    norm_ = torch.sum(torch.pow(Z, 2))
    diff_ = diff_ / torch.clamp(norm_, min=Constants.VERY_SMALL_NUMBER)
    return diff_


def batch_diff(X, Y, Z):
    assert X.shape == Y.shape
    diff_ = torch.sum(torch.pow(X - Y, 2), (1, 2)) # Shape: [batch_size]
    norm_ = torch.sum(torch.pow(Z, 2), (1, 2))
    diff_ = diff_ / torch.clamp(norm_, min=Constants.VERY_SMALL_NUMBER)
    return diff_

def SquaredFrobeniusNorm(X):
    return torch.sum(torch.pow(X, 2)) / int(np.prod(X.shape))

def batch_SquaredFrobeniusNorm(X):
    return torch.sum(torch.pow(X, 2), (1, 2)) / int(np.prod(X.shape[1:]))

