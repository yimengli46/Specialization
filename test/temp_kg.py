import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import bz2
import _pickle as cPickle
import scipy.sparse as sp
import matplotlib.pyplot as plt

with bz2.BZ2File('output/knowledge_graph/LVIS_relationships.pbz2', 'rb') as fp:
    LVIS_relationships = cPickle.load(fp)
    lvis_cat_relationships = LVIS_relationships['cat_id_relationship_dict']
    adjacency_cat_id = LVIS_relationships['adjacency_cat_id']

adjacency_cat_id = adjacency_cat_id.astype('float32')
adjacency_cat_id += 1e-5
adj = sp.coo_matrix(adjacency_cat_id)
rowsum = np.array(adj.sum(1))
d_inv_sqrt = np.power(rowsum, -0.5).flatten()
d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
A = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

A = A.tocsr().toarray()
