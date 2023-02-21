import torch
from copy import deepcopy
import time
from scipy.sparse import csr_matrix
from sparseprop import backend as sppb

def to_csr_2d(data):
    if isinstance(data, torch.nn.Parameter):
        data = data.data
    spa = csr_matrix(data, shape=data.shape)
    val = torch.Tensor(spa.data)
    idx_N = torch.Tensor(spa.indptr).int()
    idx_M = torch.Tensor(spa.indices).int()
    return val, (idx_N, idx_M)

def to_sparse_format_conv2d(dense_weight):
    OC, IC, K, _ = dense_weight.shape
    nnz = torch.sum(dense_weight != 0).item()
    W_val = torch.zeros(nnz).float()
    W_OC = torch.zeros(OC + 1).int()  
    W_IC = torch.zeros((IC + 1) * OC).type(torch.short)
    W_X = torch.zeros(nnz).type(torch.uint8)
    W_Y = torch.zeros(nnz).type(torch.uint8)
    sppb.sparsify_conv2d(OC, IC, K, dense_weight.data, W_OC, W_IC, W_X, W_Y, W_val)
    return W_val, (W_OC, W_IC, W_X, W_Y)