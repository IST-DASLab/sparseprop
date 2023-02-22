# *SparseProp*

Official implementation of the paper *"SparseProp: Efficient Sparse Backpropagation for Faster Training of Neural Networks"*.

[Link to the paper](https://arxiv.org/abs/2302.04852)

This library provides fast PyTorch modules exploiting sparse backpropagation algorithms described in the paper.

## Installation
1. Make sure you have PyTorch installed (Refer to the [PyTorch website](https://pytorch.org)). A CPU version will suffice for our purpose.
2. Install *SparseProp*:
    ```
    pip install sparseprop
    ```

## Usage

#### Sparsifying a single layer
If you have a sparse *Linear* module called `linear`, you can easily convert is to a *SparseLinear* module using the `from_dense` method.
```
from sparseprop.modules import SparseLinear

sparse_linear = SparseLinear.from_dense(linear)
```

This will automatically store the parameters of the `linear` module in a sparse format and benefit from *SparseProp*'s efficient backend. You can treat `sparse_linear` as a normal PyTorch module, e.g., you can simply call `output = sparse_linear(input)`.

A similar interface exists for a sparse *Conv2d* module (called `conv`):
```
from sparseprop.modules import SparseConv2d

sparse_conv = SparseConv2d.from_dense(conv, vectorizing_over_on=False)
```

The only difference with the *Linear* case is that there is an additional boolean argument `vectorizing_over_on`. As described in the paper, we have two implementations for the convolution case, one performing the vectorization over the bactch size `B`, and the other over the output width `ON`. Using this argument you can specify which one of the two implementations to use. A quick rule of thumb is that if the input width and height are small (e.g., less than 32) then `vectorizing_over_on=False` is faster.

Alternatively, the `sparsify_conv2d_auto` method can automatically determine the correct value of `vectorizing_over_on`.

```
from sparseprop.modules import sparsify_conv2d_auto

sparse_conv = sparsify_conv2d_auto(conv, input_shape, verbose=True)
```

Notice that you will need to feed the `input_shape` to this method, which should look something like (`batch_size`, `input_channels`, `input_height`, `input_width`). This method will create two sparse modules, one with `vectorizing_over_on=False` and the other one with `vectorizing_over_on=True`, run a randomly generated batch through both, and return the faster module based on forward+backward time.

#### Sparsifying the whole network
As explained in the paper, we replace each *Linear* or *Conv2d* layer in a network with a sparse one, if the following conditions are met:
1. It is at least 80% sparse.
2. The sparse module is faster than the original dense one (in terms of forward+backward time).

This behavior is implemented in the `swap_modules_with_sparse` method in `sparseprop.utils`. For example, if you have a sparse (global or uniform) `model`:

```
from sparseprop.utils import swap_modules_with_sparse

sparse_model = swap_modules_with_sparse(model, input_shape, verbose=True)
```

Notice that you need to provide the `input_shape` to this method, which is easily accessible through your *DataLoader*. The `swap_modules_with_sparse` method will iterate through the network's layers and replace them with their sparse counterparts if the above two conditions are met.

## Examples
In the `examples` folder, you can find multiple python scripts, which will help you get started with *SparseProp*. In order to get persistent timings, we refer you to [this article](https://easyperf.net/blog/2019/08/02/Perf-measurement-environment-on-Linux). You can use your favorite command line tool in case you want to limit the number of CPU cores on which the code executes, e.g., `taskset` or `numactl`. Refer to the "Set cpu affinity" section in the same article.

- The files `correctness_linear.py` and `correctness_conv2d.py` will compare the output of the *SparseLinear* and *SparseConv2d* modules with PyTorch's *Linear* and *Conv2d*, respectively. You can tweak the parameters in the scripts to check the correctness in different cases.
- The files `compare_linear.py` and `compare_conv2d.py` will compare the running time of the *SparseLinear* and *SparseConv2d* modules with PyTorch's *Linear* and *Conv2d*, respectively. You will find the results in the `plots` directory. Again, feel free to tweak the parameters in the scripts to compare the runtime in different cases.
- COMING SOON: Fine-tuning 95% uniformly sparse ResNet18 on imagenette dataset
