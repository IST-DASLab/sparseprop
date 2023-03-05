import torch
from scipy.sparse import csr_matrix

from sparseprop.modules.functions import SparseLinearFunction
from sparseprop.modules.utils import to_csr_2d

class SparseLinear(torch.nn.Module):
    def __init__(self, dense_weight, bias=None):
        super(SparseLinear, self).__init__()
        self.N, self.M = dense_weight.shape

        W_val, W_idx = to_csr_2d(dense_weight)
        self.W_val = torch.nn.Parameter(W_val)
        self.W_idx = W_idx

        assert bias is None or isinstance(bias, torch.nn.Parameter), f"bias is not a parameter but it's {type(bias)}"
        self.bias = bias
    
    @staticmethod
    def from_dense(module):
        return SparseLinear(
            dense_weight=module.weight.data,
            bias=None if module.bias is None else torch.nn.Parameter(module.bias.data.clone())
        )

    @property
    def weight(self):
        return self.W_val
    
    def forward(self, input):
        return SparseLinearFunction.apply(input, self.W_val, self.W_idx, self.bias, self.N)

    @torch.no_grad()
    def apply_further_mask(self, new_mask):
        """
            This function is used when we need to further sparsify a sparse module, e.g., gradual pruning.
        """

        indptr, indices = self.W_idx
        dense_weight = torch.Tensor(csr_matrix((
            self.W_val.data, 
            indices, 
            indptr
        ), shape=(self.N, self.M)).toarray()).float()

        dense_mask = torch.Tensor(csr_matrix((
            new_mask, 
            indices, 
            indptr
        ), shape=(self.N, self.M)).toarray()).float()
        
        W_val, W_idx = to_csr_2d(dense_weight * dense_mask)
        self.W_val = torch.nn.Parameter(W_val)
        self.W_idx = W_idx

    def __repr__(self):
        nnz = len(self.W_val)
        numel = self.N * self.M
        return f"SparseLinear([{self.N}, {self.M}], sp={1. - nnz/numel:.2f}, nnz={nnz})"
