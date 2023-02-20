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

@torch.enable_grad()
def run_and_choose(modules, input_shape, verbose=False):
    if len(modules) == 1:
        if verbose:
            print('only one option...')
        return modules[0]

    X_orig = torch.randn(*input_shape)
    Y_orig = None

    min_time = 1e10
    best_module = None
    for module in modules:
        module_copy = deepcopy(module)
        X = X_orig.clone()
        X.requires_grad_()
        X.retain_grad()

        temp = time.time()
        O = module_copy(X)
        fwd_time = time.time() - temp

        if Y_orig is None:
            Y_orig = torch.randn_like(O)
        Y = Y_orig.clone()

        L = torch.mean((O - Y) ** 2)
        temp = time.time()
        L.backward()
        bwd_time = time.time() - temp

        if verbose:
            print(f'module {module} took {fwd_time} fwd and {bwd_time} bwd')

        full_time = fwd_time + bwd_time
        if full_time < min_time:
            min_time = full_time
            best_module = module
    
    if verbose:
        print(f'going with {best_module} with full time of {min_time}')
    return best_module
