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
from sklearn.preprocessing import scale
import torch
from core.utils.generic_utils import *

from ucimlrepo import fetch_ucirepo 


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
    # print(idx)
    return idx

def find_last_col(df):
    last_col = ""
    for col in df:
        last_col = col
    return last_col
###

def load_data(dataset_name):
    abalone =['Abalone9-18','Abalone19','Abalone-17_vs_7-8-9-10','Abalone-19_vs_10-11-12-13']
    data_sets = ['pima','cleveland-0_vs_4','ecoli-0-1-3-7_vs_2-6','ecoli-0-1-4-6_vs_5','ecoli-0-1-4-7_vs_2-3-5-6','ecoli-0-1-4-7_vs_5-6' ,'ecoli-0-1_vs_2-3-5' ,'ecoli-0-1_vs_5','ecoli-0-2-3-4_vs_5'
            ,'ecoli-0-2-6-7_vs_3-5','ecoli-0-3-4_vs_5','ecoli-0-3-4-6_vs_5','ecoli-0-3-4-7_vs_5-6','ecoli-0-4-6_vs_5','ecoli-0-6-7_vs_3-5' ,'ecoli-0-6-7_vs_5','ecoli4'
            ,'glass-0-1-4-6_vs_2','glass-0-1-5_vs_2','glass-0-1-6_vs_2','glass-0-1-6_vs_5','glass-0-4_vs_5','glass-0-6_vs_5'
            ,'glass2','glass4','glass5','led7digit-0-2-4-5-6-7-8-9_vs_1' ,'page-blocks-1-3_vs_4','shuttle-c0-vs-c4','shuttle-c2-vs-c4','vowel0'
            ,'yeast-0-2-5-6_vs_3-7-8-9','yeast-0-2-5-7-9_vs_3-6-8','yeast-0-3-5-9_vs_7-8','yeast-0-5-6-7-9_vs_4','yeast-1-2-8-9_vs_7','yeast-1-4-5-8_vs_7','yeast-1_vs_7','yeast-2_vs_4','yeast-2_vs_8'
            ,'yeast4' ,'yeast5','yeast6'
            ,'wisconsin']

    if dataset_name in data_sets:  
        with open(f"core/utils/uci_data/{dataset_name}.dat", 'r') as f:
            data = f.readlines()
        idx = find_idx(data)
        df = pd.DataFrame([row.strip().split(',') for row in data[idx:]])       
        last_col = find_last_col(df)
        df[last_col]=df[last_col].str.replace("negative","0")
        df[last_col]=df[last_col].str.replace("positive","1")
        features = df.iloc[:, :-1].astype(dtype='float').values
        y = df[last_col].astype(dtype='float').values  
        
    elif dataset_name in abalone:
        with open(f"core/utils/uci_data/{dataset_name}.dat", 'r') as f:
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

    # return features, y, df
    return features, y #####
