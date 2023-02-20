import torch
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from sparseprop.modules import attach_identity_and_time, SparseConv2d
import math
import os, sys

def DenseConv(W, padding, stride):
    OC, IC, K, _ = W.shape
    conv = torch.nn.Conv2d(IC, OC, K, stride=stride, padding=padding)
    with torch.no_grad():
        conv.weight.mul_(0.)
        conv.weight.add_(W)
    return conv

def SparseConv2dOverON(W, padding, stride):
    return SparseConv2d(W, padding=padding, stride=stride, vectorizing_over_on=True)

if __name__ == '__main__':
    B = 64 # batch size
    IC = 128 # input channels
    OC = 128 # output channels
    M = 32 # input height
    N = 32 # input width
    K = 3 # kernel size
    stride = 1 # stride
    padding = 0 # padding
    sparsities = [.8, .9, .95, .98, .99]
    reps = 3

    torch.manual_seed(10)
    tag = sys.argv[1] if len(sys.argv) > 1 else None

    OM = math.ceil((M + 2 * padding - K + 1) / stride)
    ON = math.ceil((N + 2 * padding - K + 1) / stride)

    module_fns = [DenseConv, SparseConv2d, SparseConv2dOverON]
    module_names = [m.__name__ for m in module_fns]

    forward_times = {m: [] for m in module_names}
    backward_times = {m: [] for m in module_names}

    for sparsity in sparsities:
        sp_forward_times = {m: [] for m in module_names}
        sp_backward_times = {m: [] for m in module_names}
        for _ in tqdm(range(reps)):
            W = torch.randn(OC, IC, K, K)
            mask = torch.rand_like(W) > sparsity
            W *= mask

            Y_orig = torch.randn(B, OC, OM, ON)

            X_orig = torch.randn(B, IC, M, N)
            X_orig.requires_grad_()
            X_orig.retain_grad()

            for module_name, module_fn in zip(module_names, module_fns):
                
                module = module_fn(W, padding=padding, stride=stride)
                X = X_orig.clone()
                Y = Y_orig.clone()

                bt, ft = attach_identity_and_time(module, X, Y, time_forward=True, time_backward=True)
                sp_forward_times[module_name].append(ft)
                sp_backward_times[module_name].append(bt)

        for mn in module_names:
            forward_times[mn].append(sum(sp_forward_times[mn]) / reps)
            backward_times[mn].append(sum(sp_backward_times[mn]) / reps)

    title = f'B{B}-IC{IC}-OC{OC}-M{M}-K{K}-S{stride}-P{padding}'
    if tag is not None:
        title += '-' + tag
    os.makedirs(f'plots/conv2d/{title}', exist_ok=False)

    for mn in module_names:
        plt.plot(sparsities, forward_times[mn], '-o', label=mn)
    plt.grid()
    plt.xlabel('sparsity')
    plt.ylabel('time')
    plt.ylim(bottom=0)
    plt.title(f'{title}-forward')
    plt.legend()
    plt.savefig(f'plots/conv2d/{title}/forward.jpg')
    plt.close()

    for mn in module_names:
        plt.plot(sparsities, backward_times[mn], '-o', label=mn)
    plt.grid()
    plt.xlabel('sparsity')
    plt.ylabel('time')
    plt.ylim(bottom=0)
    plt.title(f'{title}-backward')
    plt.legend()
    plt.savefig(f'plots/conv2d/{title}/backward.jpg')
    plt.close()