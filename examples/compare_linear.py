import torch
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from sparseprop.modules import SparseLinear, attach_identity_and_time
import os, sys

class SparseLinearTorch(torch.nn.Module):
    def __init__(self, W):
        super(SparseLinearTorch, self).__init__()
        self.W = W.clone()
        self.W = W.to_sparse()
        self.W.requires_grad_()
        self.W.retain_grad()

    def forward(self, X):
        return torch.sparse.mm(self.W, X.T).T

def DenseLinear(W):
    N, M = W.shape
    linear = torch.nn.Linear(M, N, bias=False)
    with torch.no_grad():
        linear.weight.mul_(0.)
        linear.weight.add_(W)
    return linear

if __name__ == '__main__':
    M = 512
    N = 256
    B = 128
    sparsities = [.8, .9, .95, .98, .99]
    reps = 3

    torch.manual_seed(11)
    tag = sys.argv[1] if len(sys.argv) > 1 else None

    module_fns = [DenseLinear, SparseLinear, SparseLinearTorch]
    module_names = [m.__name__ for m in module_fns]

    forward_times = {m: [] for m in module_names}
    backward_times = {m: [] for m in module_names}

    for sparsity in sparsities:
        sp_forward_times = {m: [] for m in module_names}
        sp_backward_times = {m: [] for m in module_names}
        for _ in tqdm(range(reps)):
            W = torch.randn(N, M)
            mask = torch.rand_like(W) > sparsity
            W *= mask

            Y_orig = torch.randn(B, N)

            X_orig = torch.randn(B, M)
            X_orig.requires_grad_()
            X_orig.retain_grad()

            for module_name, module_fn in zip(module_names, module_fns):
                
                module = module_fn(W)
                X = X_orig.clone()
                Y = Y_orig.clone()

                bt, ft = attach_identity_and_time(module, X, Y, time_forward=True, time_backward=True)
                sp_forward_times[module_name].append(ft)
                sp_backward_times[module_name].append(bt)

        for mn in module_names:
            forward_times[mn].append(sum(sp_forward_times[mn]) / reps)
            backward_times[mn].append(sum(sp_backward_times[mn]) / reps)

    title = f'B{B}-M{M}-N{N}'
    if tag is not None:
        title += '-' + tag
    os.makedirs(f'plots/linear/{title}', exist_ok=False)

    for mn in module_names:
        plt.plot(sparsities, forward_times[mn], '-o', label=mn)
    plt.grid()
    plt.xlabel('sparsity')
    plt.ylabel('time')
    plt.ylim(bottom=0)
    plt.title(f'{title}-forward')
    plt.legend()
    plt.savefig(f'plots/linear/{title}/forward.jpg')
    plt.close()

    for mn in module_names:
        plt.plot(sparsities, backward_times[mn], '-o', label=mn)
    plt.grid()
    plt.xlabel('sparsity')
    plt.ylabel('time')
    plt.ylim(bottom=0)
    plt.title(f'{title}-backward')
    plt.legend()
    plt.savefig(f'plots/linear/{title}/backward.jpg')
    plt.close()