import math
import random
import time

import numpy as np
import scipy
import torch
from scipy.stats import ttest_ind
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import GaussianNB
from torch_scatter import scatter_add

# from utils.util_funcs import random_disassortative_splits, accuracy

pi = math.pi
# if torch.cuda.is_available():
#     device = 'cuda:0'
# else:
#     device = 'cpu'
# device = torch.device(device)
device = 'cpu'


def remove_self_loops(edge_index, edge_attr=None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    row, col = edge_index[0], edge_index[1]
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    edge_index = edge_index[:, mask]

    return edge_index, edge_attr


def edge_homophily(edge_index, labels, ignore_negative=False):
    """ gives edge homophily, i.e. proportion of edges that are intra-class
    compute homophily of classes in labels vector
    See Zhu et al. 2020 "Beyond Homophily ..."
    if ignore_negative = True, then only compute for edges where nodes both have
        nonnegative class labels (negative class labels are treated as missing
    """
    src_node, targ_node = edge_index[0].long(), edge_index[1].long()  # A.nonzero()
    matching = labels[src_node] == labels[targ_node]
    labeled_mask = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    
    print("labeled_mask", labeled_mask.shape, labeled_mask.sum())
    if ignore_negative:
        edge_hom = np.mean(matching[labeled_mask])
    else:
        edge_hom = torch.mean(matching.float())
    return edge_hom


def node_homophily(A, labels):
    """ average of homophily for each node
    """
    src_node = A.coalesce().indices()[0, :]
    targ_node = A.coalesce().indices()[1, :]
    edge_idx = torch.tensor(np.vstack((src_node, targ_node)), dtype=torch.long).contiguous()
    labels = torch.tensor(labels)
    num_nodes = A.shape[0]
    return node_homophily_edge_idx(edge_idx, labels, num_nodes)


def node_homophily_edge_idx(edge_index, labels, num_nodes):
    """ edge_idx is 2 x(number edges) """
    edge_index = remove_self_loops(edge_index)[0]
    hs = torch.zeros(num_nodes)
    degs = torch.bincount(edge_index[0, :]).float()
    matches = (labels[edge_index[0, :]] == labels[edge_index[1, :]]).float()
    hs = hs.scatter_add(0, edge_index[0, :], matches) / degs
    return hs[degs != 0].mean()


def compact_matrix_edge_idx(edge_idx, labels):
    """
     c x c compatibility matrix, where c is number of classes
     H[i,j] is proportion of endpoints that are class j 
     of edges incident to class i nodes 
     "Generalizing GNNs Beyond Homophily"
     treats negative labels as unlabeled
     """
    edge_index = remove_self_loops(edge_idx.to(device))[0]
    src_node, targ_node = edge_index[0, :], edge_index[1, :]
    labeled_nodes = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    label = labels.squeeze()
    c = label.max() + 1
    H = torch.zeros((c, c)).to(edge_index.to(device))
    src_label = label[src_node[labeled_nodes]]
    targ_label = label[targ_node[labeled_nodes]]
    for k in range(c):
        sum_idx = torch.where(src_label == k)[0]
        add_idx = targ_label[sum_idx]
        scatter_add(torch.ones_like(add_idx).to(H.dtype), add_idx, out=H[k, :], dim=-1)
    H = H / torch.sum(H, axis=1, keepdims=True)
    return H


def class_homophily(edge_index, label):
    """ 
    our measure \hat{h}
    treats negative labels as unlabeled 
    """
    label = label.squeeze()
    c = label.max() + 1
    H = compact_matrix_edge_idx(edge_index.to(device), label.to(device))
    nonzero_label = label[label >= 0]
    counts = nonzero_label.unique(return_counts=True)[1]
    proportions = counts.float() / nonzero_label.shape[0]
    val = 0
    for k in range(c):
        class_add = torch.clamp(H[k, k] - proportions[k], min=0)
        if not torch.isnan(class_add):
            # only add if not nan
            val += class_add
    val /= c - 1
    return val


def class_distribution(edge_index, labels):
    # edge_index = A.coalesce().indices()
    src_node, targ_node = edge_index[0, :], edge_index[1, :]
    deg = src_node.unique(return_counts=True)[1]

    # remove self-loop
    deg = deg - 1
    edge_index = remove_self_loops(edge_index)[0]
    src_node, targ_node = edge_index[0, :], edge_index[1, :]

    labels = labels.squeeze()
    p = labels.unique(return_counts=True)[1] / labels.shape[0]
    p_bar = torch.zeros(labels.max() + 1)
    pc = torch.zeros((labels.max() + 1, labels.max() + 1))
    for i in range(labels.max() + 1):
        p_bar[i] = torch.sum(deg[torch.where(labels == i)])

        for j in range(labels.max() + 1):
            pc[i, j] = torch.sum(labels[targ_node[torch.where(labels[src_node] == i)]] == j)
    p_bar, pc = p_bar / torch.sum(deg), pc / torch.sum(deg)
    p_bar[torch.where(p_bar == 0)], pc[torch.where(pc == 0)] = 1e-8, 1e-8
    return p, p_bar, pc


def adjusted_homo(edge_index, label):
    p, p_bar, pc = class_distribution(edge_index, label)
    edge_homo = edge_homophily(edge_index, label)
    adj_homo = (edge_homo - torch.sum(p_bar ** 2)) / (1 - torch.sum(p_bar ** 2))

    return adj_homo
