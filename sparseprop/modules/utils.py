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

def from_csr_2d(val, idx, shape):
    if isinstance(val, torch.nn.Parameter):
        val = val.data
    idx_N, idx_M = idx
    return torch.Tensor(csr_matrix((
        val, 
        idx_M, 
        idx_N
    ), shape=shape).toarray()).float()

def to_sparse_format_conv2d(dense_weight):
    if isinstance(dense_weight, torch.nn.Parameter):
        dense_weight = dense_weight.data
    OC, IC, K, _ = dense_weight.shape
    nnz = torch.sum(dense_weight != 0).item()
    W_val = torch.zeros(nnz).float()
    W_OC = torch.zeros(OC + 1).int()  
    W_IC = torch.zeros((IC + 1) * OC).type(torch.short)
    W_X = torch.zeros(nnz).type(torch.uint8)
    W_Y = torch.zeros(nnz).type(torch.uint8)
    sppb.sparsify_conv2d(OC, IC, K, dense_weight, W_OC, W_IC, W_X, W_Y, W_val)
    return W_val, (W_OC, W_IC, W_X, W_Y)

def from_sparse_format_conv2d(W_val, W_idx, shape):
    if isinstance(W_val, torch.nn.Parameter):
        W_val = W_val.data
    W_OC, W_IC, W_X, W_Y = W_idx
    OC, IC, K, _ = shape
    dense_weight = torch.zeros(*shape)
    sppb.densify_conv2d(OC, IC, K, dense_weight, W_OC, W_IC, W_X, W_Y, W_val)
    return dense_weight