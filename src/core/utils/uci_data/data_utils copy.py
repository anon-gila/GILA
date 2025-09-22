# The code below are borrowed from https://github.com/lucfra/LDS-GNN/blob/master/
import scipy.io
import os
import pickle
import math
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import kneighbors_graph

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
        self.n_train = None
        self.n_val = None
        self.ratio = None
        self.fake_samples = False
        self.fake_rate = None
        self.smote = None
        self.copy_node = None
        self.copy_rate = None
        super().__init__(**kwargs)

    def load(self, data_dir=None, knn_size=None, epsilon=None, knn_metric='cosine', ratio=0.1):
        assert (knn_size is None) or (epsilon is None)
        if self.dataset_name == 'iris':
            data = datasets.load_iris()
            scale_ = False
        elif self.dataset_name == 'wine':
            data = datasets.load_wine()
            scale_ = True
        elif self.dataset_name == 'breast_cancer':
            data = datasets.load_breast_cancer()
            scale_ = True
        elif self.dataset_name == 'digits':
            data = datasets.load_digits()
            scale_ = True
        elif self.dataset_name == 'fma':
            data = np.load(os.path.join(data_dir, 'fma.npz'))
            scale_ = False
        elif self.dataset_name == '20news10':
            scale_ = False
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.feature_extraction.text import TfidfTransformer
            data = pickle.load(open(os.path.join(data_dir, '20news10.pkl'), 'rb'))
            vectorizer = CountVectorizer(stop_words='english', min_df=0.05)
            X_counts = vectorizer.fit_transform(data.data).toarray()
            transformer = TfidfTransformer(smooth_idf=False)
            features = transformer.fit_transform(X_counts).todense()
            
        #### 
        elif self.dataset_name == 'pima':
            data = pd.read_csv(f'core/utils/uci_data/diabetes.csv')
            scale_ = True  
        elif self.dataset_name == 'phoneme' or self.dataset_name == 'CH':
            data = pd.read_csv(f'core/utils/uci_data/{self.dataset_name}.csv')
            scale_ = True  
        elif self.dataset_name == 'satimage':
            data = scipy.io.loadmat('core/utils/uci_data/satimage.mat')
            scale_ = True 
        ####
        
        else:
            raise AttributeError('dataset not available')
        
        #####
        if self.dataset_name == 'CH' or self.dataset_name == 'pima' or self.dataset_name == 'phoneme':
            from sklearn.preprocessing import scale
            features = data.iloc[:,:-1].values
            features = scale(features)
            y = data.iloc[:,-1].values
        elif self.dataset_name == 'satimage':
            from sklearn.preprocessing import scale
            features = data['X']
            features = scale(features)
            y = data['y']
        #####
        
        elif self.dataset_name != 'fma':
            from sklearn.preprocessing import scale
            if self.dataset_name != '20news10':
                if scale_:
                    features = scale(data.data)
                else:
                    features = data.data
            y = data.target
        else:
            features = data['X']
            y = data['y']
            
       
        n = features.shape[0]

        ###########################################################################################################
        # sampling
  
        print(f'-----------------------------------------------------------------------------------------')

        if self.ratio == 1 :
            ys = LabelBinarizer().fit_transform(y)
            ys = np.hstack([1-ys, ys]) 
            
            if len(np.where(y==0)[0]) < len(np.where(y==1)[0]):
                ys = 1-ys
            ys = np.argmax(ys, axis=1)
         
            for i in range(len(np.unique(ys))):
                print(f"Original Class {i} 데이터 수 :", len(np.where(ys==i)[0]))

            print('Original 총 데이터 수 : ', len(ys))

            
        else:
            
            if len(np.where(y==0)[0]) < len(np.where(y==1)[0]):
                y = 1-y
    
            for i in range(len(np.unique(y))):
                globals()[f'idx_{i}'] = np.where(y == i)
                

            array_list = []    
            for i in range(len(np.unique(y))):
                print(f"Original Class {i} 데이터 수 :", len(np.where(y==i)[0]))
                array_list.append(len(eval(f'idx_{i}')[0]))
            print('Original 총 데이터 수 : ', len(y))

            print("------- Ratio 변경중, Raio :", self.ratio, "-------")
            large_i = array_list.index(max(array_list))
            small_i = array_list.index(min(array_list))
            
            large = eval(f'idx_{large_i}')
            small = eval(f'idx_{small_i}')

            sampling_small = np.random.choice(small[0],math.ceil(len(large[0])*self.ratio))
            idx = np.concatenate([large[0], sampling_small])
            
       
            
            y= y[idx]
            features = features[idx]

            ys = LabelBinarizer().fit_transform(y)
            if ys.shape[1] == 1:
                ys = np.hstack([1-ys, ys])
                
                
            ys = np.argmax(ys, axis=1)
            
            for i in range(len(np.unique(ys))):
                print(f"Class {i} 데이터 수 :", len(np.where(ys==i)[0]))
            print('Original 총 데이터 수 : ', len(ys))
        
        ###########################################################################################################
        
        
        from sklearn.model_selection import train_test_split
        train, test, y_train, y_test = train_test_split(np.arange(features.shape[0]), ys, random_state=self.seed,
                                                        train_size=0.6, test_size=0.4,
                                                        stratify=ys)

        val, test, y_val, y_test = train_test_split(test, y_test, random_state=self.seed,
                                                    train_size= 0.5, test_size=0.5,
                                                    stratify=y_test)

        
        if self.fake_samples == True:
            print("------- fake_sample 생성중... -------")

            indices_of_0 = np.where(ys[train] == 0)[0]
            indices_of_1 = np.where(ys[train] == 1)[0]
        
            num_fake = math.floor(indices_of_0.shape[0] * self.fake_rate) - indices_of_1.shape[0]
            # num_fake = int(299-161)
            
            print("train data class 0 labels",indices_of_0.shape[0])
            print("train data class 1 labels",indices_of_1.shape[0])
            print("num_fake", num_fake, 'fake_rate', self.fake_rate)
            
            if num_fake> 0:
                fake_feature = np.zeros((num_fake,features.shape[1]))
                fake_lables = np.ones(num_fake)
                fake_idx = np.arange(n, n+num_fake)
                
            else:
                raise Exception("fake_rate을 확인해 주세요")
        
            train = np.concatenate((train, fake_idx))
            features = np.concatenate((features, fake_feature))
            ys = np.concatenate((ys, fake_lables))
            
            print(" ------- 생성 후 train data -------")
            print(f'feature 수 : {features.shape[0]}')
            print(f'class 0 labels : {np.where(ys[train] == 0)[0].shape[0]}')
            print(f'class 1 labels : {np.where(ys[train] == 1)[0].shape[0]}')
            
        if self.smote == True : 
            from imblearn.over_sampling import SMOTE
            print("-------------------------- SMOTE로 sample 생성중...")
        
            
            smote = SMOTE(random_state=self.seed)
            features_, ys_ = smote.fit_resample(features[train], ys[train])
            
            
            
            fake_feature, fake_lables = features_[len(features[train]):], ys_[len(features[train]):]
            fake_idx = np.arange(len(features),len(features)+len(fake_feature))
            
       

            train = np.concatenate((train, fake_idx))
            features = np.concatenate((features, fake_feature))
            ys = np.concatenate((ys,fake_lables))
            
            print(" ------- 생성 후 train data -------")
            print(f'feature 수 : {features.shape[0]}')
            print(f'class 0 labels : {np.where(ys[train] == 0)[0].shape[0]}')
            print(f'class 1 labels : {np.where(ys[train] == 1)[0].shape[0]}')    
            
        if self.copy_node == True:
            print("------- copy node 생성중... -------")

            indices_of_0 = np.where(ys[train] == 0)[0]
            indices_of_1 = np.where(ys[train] == 1)[0]
            
            num_fake = math.floor(indices_of_0.shape[0] * self.fake_rate) - indices_of_1.shape[0]
            # num_fake = int(299-161)
            
            print("train data class 0 labels",indices_of_0.shape[0])
            print("train data class 1 labels",indices_of_1.shape[0])

            
     
            fake_feature = features[indices_of_1]
            copy_minor = np.copy(fake_feature)
            
            for i in range((int(self.copy_rate) - 2)):
                print(i)
                fake_feature = np.concatenate((fake_feature, copy_minor))

            fake_lables = np.ones(len(fake_feature))
            fake_idx = np.arange(n, n+len(fake_feature))
                
 
        
            train = np.concatenate((train, fake_idx))
            features = np.concatenate((features, fake_feature))
            ys = np.concatenate((ys, fake_lables))
            
            print(" ------- 생성 후 train data -------")
            print(f'feature 수 : {features.shape[0]}')
            print(f'class 0 labels : {np.where(ys[train] == 0)[0].shape[0]}')
            print(f'class 1 labels : {np.where(ys[train] == 1)[0].shape[0]}')    
   
        features = torch.Tensor(features)
        labels = torch.LongTensor(ys)
        # if self.fake_samples == True:
        #     labels = torch.LongTensor(ys)
        # else:
        #     labels = torch.LongTensor(np.argmax(ys, axis=1))
            
        idx_train = torch.LongTensor(train)
        idx_val = torch.LongTensor(val)
        idx_test = torch.LongTensor(test)
        if self.copy_node == True or self.smote == True :
            fake_idx = torch.LongTensor(fake_idx)


        if not knn_size is None:
            print('[ Using KNN-graph as input graph: {} ]'.format(knn_size))
            adj = kneighbors_graph(features, knn_size, metric=knn_metric, include_self=True)
            adj_norm = normalize_sparse_adj(adj)
            adj_norm = torch.Tensor(adj_norm.todense())
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

        if self.copy_node == True or self.smote == True :
            return adj_norm, features, labels, idx_train, idx_val, idx_test, fake_idx
            
        else:
            return adj_norm, features, labels, idx_train, idx_val, idx_test

