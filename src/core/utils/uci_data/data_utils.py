# The code below are borrowed from https://github.com/lucfra/LDS-GNN/blob/master/
import scipy.io
import os
import pickle
import sys
import math
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import kneighbors_graph
from ucimlrepo import fetch_ucirepo 

import torch
from ..generic_utils import *


class Config:
    """ Base class of a configuration instance; offers keyword initialization with easy defaults,
    pretty printing and grid search!
    """
    def __init__(self, **kwargs):
        self._version = 1
        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise AttributeError('This config does not include attribute: {}'.format(k) +
                                     '\n Available attributes with relative defaults are\n{}'.format(
                                         str(self.default_instance())))

    def __str__(self):
        _sting_kw = lambda k, v: '{}={}'.format(k, v)

        def _str_dict_pr(obj):
            return [_sting_kw(k, v) for k, v in obj.items()] if isinstance(obj, dict) else str(obj)

        return self.__class__.__name__ + '[' + '\n\t'.join(
            _sting_kw(k, _str_dict_pr(v)) for k, v in sorted(self.__dict__.items())) + ']\n'

    @classmethod
    def default_instance(cls):
        return cls()

    @classmethod
    def grid(cls, **kwargs):
        """Builds a mesh grid with given keyword arguments for this Config class.
        If the value is not a list, then it is considered fixed"""

        class MncDc:
            """This is because np.meshgrid does not always work properly..."""

            def __init__(self, a):
                self.a = a  # tuple!

            def __call__(self):
                return self.a

        sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
        for k, v in sin.items():
            copy_v = []
            for e in v:
                copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
            sin[k] = copy_v

        grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
        return [cls(**far.utils.merge_dicts(
            {k: v for k, v in kwargs.items() if not isinstance(v, list)},
            {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
        )) for vv in grd]



class ConfigData(Config):
    def __init__(self, **kwargs):
        self.seed = 0
        self.f1 = 'load_data_del_edges'
        self.dataset_name = 'cora'
        self.kwargs_f1 = {}
        self.f2 = 'reorganize_data_for_es'
        self.kwargs_f2 = {}
        super().__init__(**kwargs)

    def load(self):
        res = eval(self.f1)(seed=self.seed, dataset_name=self.dataset_name, **self.kwargs_f1)
        if self.f2:
            res = eval(self.f2)(res, **self.kwargs_f2, seed=self.seed)
        return res



class UCI(ConfigData):

    def __init__(self, **kwargs):
        self.seed = None
        self.n_train = None
        self.n_val = None
        self.ratio = None
        self.fake_samples = None
        self.fake_rate = None
        self.smote = None
        self.whole_minor = None
        self.noise_rate = None
        super().__init__(**kwargs)

    def load(self, data_index, data_dir=None, knn_size=None, epsilon=None, knn_metric='cosine', ratio=0.1):
        assert (knn_size is None) or (epsilon is None)
        

        abalone =['Abalone9-18','Abalone19','Abalone-17_vs_7-8-9-10','Abalone-19_vs_10-11-12-13']
        data_sets = ['pima','cleveland-0_vs_4','ecoli-0-1-3-7_vs_2-6','ecoli-0-1-4-6_vs_5','ecoli-0-1-4-7_vs_2-3-5-6','ecoli-0-1-4-7_vs_5-6' ,'ecoli-0-1_vs_2-3-5' ,'ecoli-0-1_vs_5','ecoli-0-2-3-4_vs_5'
                ,'ecoli-0-2-6-7_vs_3-5','ecoli-0-3-4_vs_5','ecoli-0-3-4-6_vs_5','ecoli-0-3-4-7_vs_5-6','ecoli-0-4-6_vs_5','ecoli-0-6-7_vs_3-5' ,'ecoli-0-6-7_vs_5','ecoli4'
                ,'glass-0-1-4-6_vs_2','glass-0-1-5_vs_2','glass-0-1-6_vs_2','glass-0-1-6_vs_5','glass-0-4_vs_5','glass-0-6_vs_5'
                ,'glass2','glass4','glass5','led7digit-0-2-4-5-6-7-8-9_vs_1' ,'page-blocks-1-3_vs_4','shuttle-c0-vs-c4','shuttle-c2-vs-c4','vowel0'
                ,'yeast-0-2-5-6_vs_3-7-8-9','yeast-0-2-5-7-9_vs_3-6-8','yeast-0-3-5-9_vs_7-8','yeast-0-5-6-7-9_vs_4','yeast-1-2-8-9_vs_7','yeast-1-4-5-8_vs_7','yeast-1_vs_7','yeast-2_vs_4','yeast-2_vs_8'
                ,'yeast4' ,'yeast5','yeast6'
                ,'wisconsin']
        
        def to_int(col):
            if col == "positive":
                return 1
            elif col == "negative":
                return 0
            
        def find_idx(data):
            idx = 0
            for i,dt in enumerate(data):
                if dt[0] == '@':
                    pass
                elif dt[0] != '@':
                    idx = i
                    break
            print(idx)
            return idx

        def find_last_col(df):
            last_col = ""
            for col in df:
                last_col = col
            return last_col
        ###
        

            
        #### 

        if self.dataset_name in data_sets:  
    
            with open(f"core/utils/uci_data/{self.dataset_name}.dat", 'r') as f:
                data = f.readlines()
            idx = find_idx(data)
            df = pd.DataFrame([row.strip().split(',') for row in data[idx:]])       
            last_col = find_last_col(df)
            df[last_col]=df[last_col].str.replace("negative","0")
            df[last_col]=df[last_col].str.replace("positive","1")
            features = df.iloc[:, :-1].astype(dtype='float').values
            y = df[last_col].astype(dtype='float').values  
            
        elif self.dataset_name in abalone:
            with open(f"core/utils/uci_data/{self.dataset_name}.dat", 'r') as f:
                data = f.readlines()
            idx = find_idx(data)
            df = pd.DataFrame([row.strip().split(',') for row in data[idx:]])  

            from sklearn.preprocessing import OneHotEncoder
            cat_f = df[0].values.reshape(-1,1)
            encoder = OneHotEncoder()
            encoder.fit(cat_f)
            tr_f = encoder.transform(cat_f).toarray()
            features = np.concatenate([tr_f, df.iloc[:, 1:-1].astype(dtype='float').values], axis=1)
            last_col = find_last_col(df)
            df[last_col]=df[last_col].str.replace("negative","0")
            df[last_col]=df[last_col].str.replace("positive","1")
            y = df[last_col].astype(dtype='int').values
            
        else:
            raise AttributeError('dataset not available')



        n = features.shape[0]

  
        print(f'-----------------------------------------------------------------------------------------')

        if self.ratio == 1 :
            ys = LabelBinarizer().fit_transform(y)
            ys = np.hstack([1-ys, ys]) 
            if len(np.where(y==0)[0]) < len(np.where(y==1)[0]):
                ys = 1-ys
            ys = np.argmax(ys, axis=1)
         
            for i in range(len(np.unique(ys))):
                print(f"Number of samples in original class {i}:", len(np.where(ys==i)[0]))
            print("Total number of samples in the original data:", len(ys))

            
        else:
            if len(np.where(y==0)[0]) < len(np.where(y==1)[0]):
                y = 1-y
            for i in range(len(np.unique(y))):
                globals()[f'idx_{i}'] = np.where(y == i)
            array_list = []    
            for i in range(len(np.unique(y))):
                print(f"Number of samples in original class {i}:", len(np.where(y==i)[0]))
                array_list.append(len(eval(f'idx_{i}')[0]))  #0 은 다수클래스 샘플수, 1는 소수클래스 샘플 수
            print("Total number of samples in the original data:", len(y))

            
            train, test = data_index
            train_neg = train[y[train]==0]
            train_pos = train[y[train]==1]
            
            test_neg = test[y[test]==0]
            test_pos = test[y[test]==1]
    
            num_samples = round(len(train_neg) * self.ratio)
            sampled_train_pos= train_pos[torch.randperm(len(train_pos))[:num_samples]]

            num_samples = round(len(test_neg) * self.ratio)
            sampled_test_pos= test_pos[torch.randperm(len(test_pos))[:num_samples]]                 

            train = np.concatenate([train_neg, sampled_train_pos])
            sampled_test_pos = np.atleast_1d(sampled_test_pos)
            test = np.concatenate([test_neg, sampled_test_pos])
            idx = np.concatenate([train, test])
            
       
            
            y= y[idx]
            features = features[idx]

            ys = LabelBinarizer().fit_transform(y)
            if ys.shape[1] == 1:
                ys = np.hstack([1-ys, ys])
                
                
            ys = np.argmax(ys, axis=1)
            
            for i in range(len(np.unique(ys))):
                print(f"Number of samples in class {i}:", len(np.where(ys==i)[0]))
            print("Total number of samples in the original data:", len(ys))
            
            train = np.arange(len(train))
            test = np.arange(len(train), len(idx))
            
        
        ###########################################################################################################
        
        
        from sklearn.model_selection import train_test_split
        
        if self.ratio == 1 :
            train, test = data_index
        
            
            train, val, y_train, y_val = train_test_split(train, train, random_state=self.seed,
                                                train_size= 0.75, test_size=0.25,
                                                stratify=y[train])
        else:
            train, val, y_train, y_val = train_test_split(train, train, random_state=self.seed,
                                                train_size= 0.75, test_size=0.25,
                                                stratify=y[train])            
        
        

        
        ### SMOTE   
        if self.smote == True : 
            from imblearn.over_sampling import SMOTE, BorderlineSMOTE
            if self.dataset_name =='ecoli-0-1-3-7_vs_2-6':
                smote = SMOTE(random_state=self.seed, k_neighbors=3)
            elif self.dataset_name == 'glass-0-1-6_vs_5' or self.dataset_name == 'glass-0-4_vs_5' or self.dataset_name=='glass-0-6_vs_5'or self.dataset_name=='glass5':
                smote = SMOTE(random_state=self.seed, k_neighbors=4)
            elif self.dataset_name =='shuttle-c2-vs-c4':
                smote = SMOTE(random_state=self.seed, k_neighbors=2)
            else:
                smote = SMOTE(random_state=self.seed)
  
            
            features_, ys_ = smote.fit_resample(features[train], ys[train])
            
        
            fake_feature, fake_lables = features_[len(features[train]):], ys_[len(features[train]):]
            make_idx = np.arange(len(features),len(features)+len(fake_feature))
            
            if self.whole_minor == True:
                fake_idx = np.where(ys_==i)[0]
            else:
                fake_idx = make_idx
            

            train = np.concatenate((train, make_idx))
            features = np.concatenate((features, fake_feature))
            ys = np.concatenate((ys,fake_lables))
            
            print(" ------- Training data after SMOTE -------")
            print(f"Number of samples : {features.shape[0]}")
            print(f'class 0 labels : {np.where(ys[train] == 0)[0].shape[0]}')
            print(f'class 1 labels : {np.where(ys[train] == 1)[0].shape[0]}') 
            
            

        from sklearn.preprocessing import scale
        features = scale(features)    
        
        if self.fake_samples == True:
            print(" ------- Training data before helper node generation -------")
            print(f'Number of samples : {features.shape[0]}')
            print(f'class 0 labels : {np.where(ys[train] == 0)[0].shape[0]}')
            print(f'class 1 labels : {np.where(ys[train] == 1)[0].shape[0]}') 
            
            major = np.where(ys[train] == 0)[0].shape[0]
            minor = np.where(ys[train] == 1)[0].shape[0]

         

            #########zero vetro init
            if self.fake_rate <= 1:
                if self.fake_rate == 1:
                    num_fake = major - minor
                elif self.fake_rate <1:
                    num_fake = math.ceil((major * self.fake_rate) - minor)
                    print('num fake', num_fake,"major", major ,"minor",minor, "self.fake_rate", self.fake_rate)
                if num_fake >0:
                    fake_feature = np.zeros((num_fake,features.shape[1]))
                    fake_lables = np.ones(num_fake)
                    fake_idx = np.arange(len(features), len(features)+num_fake)
                else: 
                    raise Exception("Data ratio mismatch")
                
            else:
                print("fake rate : ", self.fake_rate)
                maj_fake = math.ceil((major * self.fake_rate) - major)
                min_fake = (major + maj_fake) - minor
                num_fake = maj_fake + min_fake
                fake_feature = np.zeros((num_fake,features.shape[1]))
                min_lable = np.ones(min_fake)
                maj_lable = np.zeros(maj_fake)
                fake_lables = np.concatenate((min_lable, maj_lable))
                fake_idx = np.arange(n, n+num_fake)
         
                
     
            # fake_idx = np.where(ys_==i)[0]
            
            train = np.concatenate((train, fake_idx))
            features = np.concatenate((features, fake_feature))
            ys = np.concatenate((ys, fake_lables))
            
            
            print(" ------- Training data after helper node generation -------")
            print(f'Number of samples : {features.shape[0]}')
            print(f'class 0 labels : {np.where(ys[train] == 0)[0].shape[0]}')
            print(f'class 1 labels : {np.where(ys[train] == 1)[0].shape[0]}') 
            
            
        
 
   
        features = torch.Tensor(features)
        labels = torch.LongTensor(ys)
        idx_train = torch.LongTensor(train)
        idx_val = torch.LongTensor(val)
        idx_test = torch.LongTensor(test)
    
        
        if self.smote == True or self.fake_samples == True:
            fake_idx = torch.LongTensor(fake_idx)


        if not knn_size is None:
            print('[ Using KNN-graph as input graph: {} ]'.format(knn_size))
            adj = kneighbors_graph(features, knn_size, metric=knn_metric, include_self=True)
            adj_norm = normalize_sparse_adj(adj)
            adj_norm = torch.Tensor(adj_norm.todense())
            # adj_norm = torch.HalfTensor(adj_norm.todense())
            
        elif not epsilon is None:
            print('[ Using Epsilon-graph as input graph: {} ]'.format(epsilon))
            feature_norm = features.div(torch.norm(features, p=2, dim=-1, keepdim=True))
            attention = torch.mm(feature_norm, feature_norm.transpose(-1, -2))
            mask = (attention > epsilon).float()
            adj = attention * mask
            adj = (adj > 0).float()
            adj_norm = normalize_adj(adj)
            # adj_norm = adj / torch.clamp(torch.sum(adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
        else:
            adj_norm = None

        if  self.smote == True or self.fake_samples == True:
            return adj_norm, features, labels, idx_train, idx_val, idx_test, fake_idx
            
        else:
            return adj_norm, features, labels, idx_train, idx_val, idx_test

