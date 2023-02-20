import torch
import time

from sparseprop.modules.conv2d import SparseConv2d
from sparseprop.modules.linear import SparseLinear
from sparseprop.modules.utils import run_and_choose

def _sparsify_if_faster_linear(module, input_shape, include_dense, verbose):
    sp = SparseLinear(
        dense_weight=module.weight.data,
        bias=None if module.bias is None else torch.nn.Parameter(module.bias.data.clone())
    )
    
    if not include_dense:
        return sp

    assert input_shape is not None
    return run_and_choose([module, sp], input_shape, verbose=verbose)

def _sparsify_if_faster_conv2d(conv, input_shape, include_dense, verbose):
    def bias_to_param():
        if conv.bias is None:
            return None
        return torch.nn.Parameter(conv.bias.data.clone())

    dense_weight = conv.weight.data
    stride = conv.stride[0]
    padding = conv.padding[0]

    sp1 = SparseConv(
        dense_weight,
        bias=bias_to_param(),
        padding=padding,
        stride=stride,
        vectorizing_over_on=False
    )
    
    sp2 = SparseConv(
        dense_weight,
        bias=bias_to_param(),
        padding=padding,
        stride=stride,
        vectorizing_over_on=True
    )

    modules = []
    if include_dense:
        modules.append(conv)
    modules += [sp1, sp2]
    
    return run_and_choose(modules, input_shape, verbose=verbose)

def sparsify_if_faster(module, input_shape, include_dense=True, verbose=False):
    if isinstance(module, torch.nn.Linear):
        return _sparsify_if_faster_linear(module, input_shape, include_dense, verbose)
    else:
        assert isinstance(module, torch.nn.Conv2d)
        return _sparsify_if_faster_conv2d(module, input_shape, include_dense, verbose)


class TimingHook:
    def __init__(self, tag=None, verbose=False):
        self.clear()
        self._tag = tag
        self._verbose = verbose
        
    def __call__(self, module, inp, out):
        self.time = time.time()
        self.count += 1
        
        if isinstance(inp, tuple):
            inp = inp[0]
        if isinstance(out, tuple):
            out = out[0]

        if self._verbose:
            print(f"[Hook {self._tag}] inp: {inp.shape if inp is not None else inp}, out: {out.shape if out is not None else None}")
        
    def clear(self):
        self.time = None
        self.count = 0


def attach_identity_and_time(module, X, Y, time_forward=False, time_backward=True):
    if not time_backward:
        t = time.time()
        O = module(X)
        return time.time() - t
    else:
        identity = torch.nn.Identity()

        X.requires_grad_()
        X.retain_grad()

        module_hook = TimingHook()
        identity_hook = TimingHook()
        handles = [
            module.register_full_backward_hook(module_hook),
            identity.register_full_backward_hook(identity_hook)
        ]

        t = time.time()
        O_before_identity = module(X)
        forward_time = time.time() - t

        O = identity(O_before_identity)
        L = torch.mean((O - Y) ** 2)

        L.backward()

        backward_time = module_hook.time - identity_hook.time

        for handle in handles:
            handle.remove()
                
        if time_forward:
            return backward_time, forward_time
        else:
            return backward_time