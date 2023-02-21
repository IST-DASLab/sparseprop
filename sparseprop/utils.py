import torch
from copy import deepcopy

from sparseprop.modules.linear import SparseLinear
from sparseprop.modules.conv2d import SparseConv2d
from sparseprop.modules import sparsify_if_faster

@torch.no_grad()
def sparsity(module):
    if hasattr(module, 'weight'):
        return 1. - torch.mean((module.weight != 0).float()).item()
    else:
        return 0.

def swap_module(network, module_name, new_module):
    name_parts = module_name.split('.')
    parent = network
    for part in name_parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    
    last_part = name_parts[-1]
    if last_part.isdigit():
        parent[int(last_part)] = new_module
    else:
        setattr(parent, last_part, new_module)

class ShapeHook:
    def __init__(self):
        self.inshape = None
        self.outshape = None
        
    def __call__(self, module, inp, out):
        if isinstance(inp, tuple):
            inp = inp[0]
        if isinstance(out, tuple):
            out = out[0]
        
        self.inshape = inp.shape[1:]
        self.outshape = out.shape[1:]

def generate_intermediate_shapes(network, input_shape):
    hooks = {}
    handles = []
    for name, module in network.named_modules():
        if any([isinstance(module, c) for c in [torch.nn.Linear, torch.nn.Conv2d]]):
            hook = ShapeHook()
            handles.append(module.register_forward_hook(hook))
            hooks[name] = hook

    B = 1
    training_mode = network.training
    network.eval()
    with torch.no_grad():
        network(torch.randn(B, *input_shape))
    network.train(training_mode)
    
    inshapes = {name: hook.inshape for name, hook in hooks.items()}
    outshapes = {name: hook.outshape for name, hook in hooks.items()}

    for handle in handles:
        handle.remove()
    
    return inshapes, outshapes

def swap_modules_with_sparse(network, input_shape, inplace=False, skip_modules=None, verbose=False):
    # e.g., shapes_tag='resnet18', input_shape=(B, IC, M, N), skip_modules='input,conv1,conv2'

    if not inplace:
        network = deepcopy(network)
    if skip_modules is not None:
        skip_modules = skip_modules.split(',')

    B = input_shape[0]
    input_shape = input_shape[1:]
    inshapes, _ = generate_intermediate_shapes(network, input_shape)

    for name, module in network.named_modules():
        is_conv = isinstance(module, torch.nn.Conv2d)
        is_linear = isinstance(module, torch.nn.Linear)
        if not is_conv and not is_linear:
            continue
        
        found = False
        if skip_modules is not None:
            for sm in skip_modules:
                if name == sm.strip():
                    found = True
                    break
        if found:
            print(f'Skipped {name}.')
            continue
        
        sp = sparsity(module)
        new_module = None
        if sp > .8:
            new_module = sparsify_if_faster(
                module,
                (B, *inshapes[name]),
                verbose=verbose
            )

        if new_module is not None and new_module != module:
            swap_module(network, name, new_module)
            print(f'module {name} replaced with {str(new_module)}')
        else:
            print(f'keeping the module {name} dense...')

    return network

def error(pred, target):
    e = torch.mean((pred - target) ** 2) / torch.norm(target) ** 2
    return e.item()