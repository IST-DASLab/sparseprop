import torch
from sparseprop.modules.utils import to_sparse_format_conv2d
from sparseprop.modules.functions import SparseConvFunction
from sparseprop import backend as sppb
from copy import deepcopy

class SparseConv2d(torch.nn.Module):
    def __init__(self, dense_weight, bias=None, padding=0, stride=1, vectorizing_over_on=False):
        super(SparseConv2d, self).__init__()

        self.OC, self.IC, self.K, _ = dense_weight.shape
        self.padding = padding
        self.stride = stride
        self.set_vectorizing_over_on(vectorizing_over_on)

        W_val, W_idx = to_sparse_format_conv2d(dense_weight)
        
        self.W_val = torch.nn.Parameter(W_val)
        self.W_idx = W_idx
        
        assert bias is None or isinstance(bias, torch.nn.Parameter), f"bias is not a parameter but it's {type(bias)}"
        self.bias = bias

    @staticmethod
    def from_dense(conv, vectorizing_over_on=False):
        def bias_to_param():
            if conv.bias is None:
                return None
            return torch.nn.Parameter(conv.bias.data.clone())

        dense_weight = conv.weight.data
        stride = conv.stride[0]
        padding = conv.padding[0]

        return SparseConv2d(
            dense_weight,
            bias=bias_to_param(),
            padding=padding,
            stride=stride,
            vectorizing_over_on=vectorizing_over_on
        )

    def set_vectorizing_over_on(self, vectorizing_over_on):
        self.vectorizing_over_on = vectorizing_over_on
        self.vectorizing_bwd_over_on = vectorizing_over_on
        self.vectorizing_fwd_over_on = vectorizing_over_on and self.stride == 1 # stride 2 is not supported over on

    @property
    def weight(self):
        return self.W_val

    def forward(self, input):
        return SparseConvFunction.apply(input, self.W_val, self.W_idx, self.bias, self.OC, self.K, self.padding, self.stride, self.vectorizing_fwd_over_on, self.vectorizing_bwd_over_on)

    @torch.no_grad()
    def apply_further_mask(self, new_mask, input_shape=None, verbose=False):
        new_mask = (new_mask * 1).int()
        W_idx_OC, W_idx_IC, W_idx_X, W_idx_Y = self.W_idx
        new_nnz = torch.sum(new_mask).item()
        W_val_new = torch.zeros(new_nnz).float()
        W_idx_OC_new = torch.zeros_like(W_idx_OC).int()
        W_idx_IC_new = torch.zeros_like(W_idx_IC).type(torch.short)
        W_idx_X_new = torch.zeros(new_nnz).type(torch.uint8)
        W_idx_Y_new = torch.zeros(new_nnz).type(torch.uint8)
        sppb.further_sparsify_conv2d(
            self.OC,
            self.IC,
            W_idx_OC,
            W_idx_IC,
            W_idx_X,
            W_idx_Y,
            self.W_val.data,
            W_idx_OC_new,
            W_idx_IC_new,
            W_idx_X_new,
            W_idx_Y_new,
            W_val_new,
            new_mask
        )

        sp1 = deepcopy(self)
        sp1.W_val = torch.nn.Parameter(W_val_new)
        sp1.W_idx = W_idx_OC_new, W_idx_IC_new, W_idx_X_new, W_idx_Y_new

        sp2 = deepcopy(sp1)
        sp2.set_vectorizing_over_on(not sp1.vectorizing_over_on)

        sp = run_and_choose([sp1, sp2], input_shape=input_shape, verbose=verbose)
        self.W_val = torch.nn.Parameter(W_val_new)
        self.W_idx = W_idx_OC_new, W_idx_IC_new, W_idx_X_new, W_idx_Y_new
        self.set_vectorizing_over_on(sp.vectorizing_over_on)
    
    def __repr__(self):
        nnz = len(self.W_val)
        numel = self.OC * self.IC * self.K * self.K
        return f'SparseConv2d([{self.OC}, {self.IC}, {self.K}, {self.K}], sp={1. - nnz/numel:.2f}, nnz={nnz}, s={self.stride}, p={self.padding}, voo={self.vectorizing_over_on})'