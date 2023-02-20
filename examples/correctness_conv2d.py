import torch
from sparseprop.modules import SparseConv2d
from sparseprop.utils import error
import math
from copy import deepcopy

if __name__ == '__main__':
    torch.manual_seed(11)

    B = 256 # batch size
    IC = 512 # input channels
    OC = 512 # output channels
    M = 7 # input height
    N = 7 # input width
    K = 3 # kernel size
    stride = 1 # stride
    padding = 0 # padding
    vectorizing_over_on = False # as described in the paper
    sparsity = .9 # sparsity of the weights

    OM = math.ceil((M + 2 * padding - K + 1) / stride)
    ON = math.ceil((N + 2 * padding - K + 1) / stride)

    W = torch.randn(OC, IC, K, K)
    bias = torch.randn(OC)
    mask = torch.rand_like(W) > sparsity
    W *= mask

    Y_orig = torch.randn(B, OC, OM, ON)

    X_orig = torch.randn(B, IC, M, N)
    X_orig.requires_grad_()
    X_orig.retain_grad()

    torch_X = X_orig.clone()
    torch_X.retain_grad()
    torch_Y = Y_orig.clone()
    conv = torch.nn.Conv2d(IC, OC, K, stride=stride, padding=padding, bias=True)
    with torch.no_grad():
        conv.weight.mul_(0.)
        conv.weight.add_(W)
        conv.bias.mul_(0.)
        conv.bias.add_(bias)
    torch_O = conv(torch_X)
    torch.mean((torch_O - torch_Y) ** 2).backward()
    torch_X_grad = torch_X.grad
    torch_W_grad = conv.weight.grad[conv.weight != 0]

    our_X = X_orig.clone()
    our_X.retain_grad()
    our_Y = Y_orig.clone()
    spconv = SparseConv2d(W, bias=torch.nn.Parameter(deepcopy(bias)), padding=padding, stride=stride, vectorizing_over_on=vectorizing_over_on)
    our_O = spconv(our_X)
    torch.mean((our_O - our_Y) ** 2).backward()
    our_X_grad = our_X.grad
    our_W_grad = spconv.W_val.grad

    print('[Forward]\n O error:', error(our_O, torch_O))
    print('[Backward]\n X grad error:', error(our_X_grad, torch_X_grad), '\n W grad error:', error(our_W_grad, torch_W_grad))
    print('[Backward]\n bias grad error:', error(spconv.bias.grad, conv.bias.grad))