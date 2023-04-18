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

__Check out our [tutorial notebook](https://github.com/IST-DASLab/sparseprop/blob/main/examples/notebook.ipynb) for a simple and step-by-step guide on how to use *SparseProp*.__

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

#### Correctness check
The files `correctness_linear.py` and `correctness_conv2d.py` will compare the output of the *SparseLinear* and *SparseConv2d* modules with PyTorch's *Linear* and *Conv2d*, respectively. You can tweak the parameters in the scripts to check the correctness in different cases.

#### Layer-wise performance comparison
The files `compare_linear.py` and `compare_conv2d.py` will compare the running time of the *SparseLinear* and *SparseConv2d* modules with PyTorch's *Linear* and *Conv2d*, respectively. You will find the results in the `plots` directory. Again, feel free to tweak the parameters in the scripts to compare the runtime in different cases.

#### Sparse fine-tuning of ResNet18 on imagenette

For this example to work, you will need to have the [*sparseml*](https://github.com/neuralmagic/sparseml) library installed, as we use it to conveniently load the imagenette dataset (`pip install sparseml`).

The file `finetune_resnet18_imagenette.py` finetunes a pretrained sparse ResNet18 model on the imagenette dataset, keeping the sparsity masks fixed. In the `examples/models/` folder, we have also included a 95% uniformly pruned ResNet18 checkpoint trained on imagenet (using the [AC/DC](https://arxiv.org/abs/2106.12379) method). You can use the following command to run this script on 4 cpu cores.

```
taskset -c 0,1,2,3 nice -n 5 python finetune_resnet18_imagenette.py --checkpoint-path=models/resnet18_ac_dc_500_epochs_sp\=0.95_uniform.pt --output-dir=results/resnet18_ac_dc_500_epochs_sp\=0.95_uniform/
```

Notice that "`0,1,2,3`" are the core numbers, so simply modify that in case your machine has less than 4 cores. Also "`nice -n 5`" gives a high priority to your process.

The most important arguments of this script are:
- `--checkpoint-path`: Path to the pretrained checkpoint.
- `--output-dir`: Path to a directory where you wish to write the results.
- `--run-dense`: You can use this argument to run this script without *SparseProp*.

For the complete list of arguments, refer to [here](https://github.com/IST-DASLab/sparseprop/blob/96a8f545461847effe863e4471d1cd80b33fc0a2/examples/finetune_resnet18_imagenette_95_uniform.py#L16).

In addition to the loss and accuracy metrics, this script also reports the time spent in each part of the process. The timings include:

- `avg_end_to_end_forward`: the average time spent in the forward pass, i.e., the `model(inputs)` line.
- `avg_end_to_end_backward`: the average time spent in the backward pass, i.e., the `loss.backward()` line.
- `avg_end_to_end_minibatch`: the average time spent processing a minibatch. This includes forward pass, backward pass, loss calculation, optimization step, etc. Note that loading the data into memory is not included.
- `avg_module_forward_sum`: the average time spent in the forward function of the modules *torch.nn.Linear*, *torch.nn.Conv2d*, *SparseLinear*, and *SparseConv2d*.
- `avg_module_backward_sum`: the average time spent in the backward function of the modules *torch.nn.Linear*, *torch.nn.Conv2d*, *SparseLinear*, and *SparseConv2d*.

## Todo
1. Include outputs of the example scripts in the README.
2. Prepare an example script for training a sparse model from scratch using gradual magnitude pruning. This will most likely be integrated into [this](https://github.com/IST-DASLab/ACDC) repository.
