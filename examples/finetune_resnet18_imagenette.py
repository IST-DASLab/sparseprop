import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sparseml.pytorch.datasets import ImagenetteDataset, ImagenetteSize
import numpy
import random

from argparse import ArgumentParser
from pprint import pformat
import os

from sparseprop.utils import sparsity, swap_modules_with_sparse
from utils import Logger, Timer, apply_to_all_modules_with_types, Finetuner

# arguments
parser = ArgumentParser()
parser.add_argument('-b','--batch-size', help='batch size for fine-tuning.', type=int, default=64)
parser.add_argument('-nw','--num-workers', help='number of workers for dataloaders.', type=int, default=4)
parser.add_argument('-rd', '--run-dense', help='set true to not use sparseprop and run everything dense.', action='store_true', default=False)
parser.add_argument('-s','--seed', help='manual seed.', type=int, default=10)
parser.add_argument('-e','--epochs', help='the number of epoch to train.', type=int, default=۱)
parser.add_argument('-sf','--save-frequency', help='how often save the model (in epochs).', type=int, default=2)
parser.add_argument('-lf','--log-frequency', help='how often to log (in batches).', type=int, default=20)
parser.add_argument('-cp','--checkpoint-path', dest='ckpt_path', help='path to the pretrained sparse resnet18 checkpoint to be fine-tuned.', type=str, required=True)
parser.add_argument('-od','--output-dir', dest='outdir', help='where to write the results. cannot already exist.', type=str, required=True)
parser.add_argument('-dd','--dataset-dir', help='where to store the dataset. we recommend /dev/shm/datasets/imagenette/. storing the data in /dev/shm/ will map it directly to memory, minimizing the data loading overhead.', type=str, default='/dev/shm/datasets/imagenette/')
args = parser.parse_args()


# set the seed everywhere
random.seed(args.seed)
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)


# make a directory to save the results in
os.makedirs(args.outdir, exist_ok=False)


# initialize the logger
logger = Logger(args.outdir)

# load the sparse model
model = resnet18()
ckpt = torch.load(args.ckpt_path, map_location='cpu')
model.load_state_dict(ckpt)


# print sparsity of each layer
logger.log("Sparsity per layer:")
logger.log(pformat(apply_to_all_modules_with_types(
    model,
    [torch.nn.Linear, torch.nn.Conv2d],
    lambda n, m: f'{sparsity(m):.3f}')
, indent=4))
logger.log('-' * 40)


# swap the last layer of the model to match the number of classes in imagenette
fc = model.fc
model.fc = torch.nn.Linear(
    fc.in_features,
    10, # number of classes in imagenette
    bias=fc.bias is not None
)


# load the datasets and prepare the dataloaders
train_dataset, test_dataset = [ImagenetteDataset(
    root=args.dataset_dir,
    train=train,
    dataset_size=ImagenetteSize.s320,
    image_size=224
) for train in [True, False]]

train_loader, test_loader = [DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True
) for dataset in [train_dataset, test_dataset]]

logger.log(f'Total number of training batches: {len(train_loader)}')


# replace modules with sparse ones if --run-dense not requested
if not args.run_dense:
    input_shape = next(iter(train_loader))[0].shape
    model = swap_modules_with_sparse(model, input_shape, inplace=True, verbose=True)


# loss and optim
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-4)


# initialize the finetuner
finetuner = Finetuner(
    model,
    optimizer,
    schedular=None,
    loss_fn=loss_fn,
    log_freq=args.log_frequency,
    save_freq=args.save_frequency,
    logger=logger
)


# finetune
finetuner.finetune(train_loader, test_loader, args.epochs)


