# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
from ...swag.swag_callback import add_model_specific_args as add_swag_args
import pytorch_lightning as pl
from .dataset import (
    augmentations_args,
    custom_dataset_args,
    dataset_args,
    linear_augmentations_args,
)
from .utils import additional_setup_linear, additional_setup_pretrain
from ..methods import METHODS
from ..utils.checkpointer import Checkpointer

try:
    from ..utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True


def parse_args_pretrain(parser) -> argparse.Namespace:
    """Parses dataset, augmentation, pytorch lightning, model specific and additional args.

    First adds shared args such as dataset, augmentation and pytorch lightning args, then pulls the
    model name from the command and proceeds to add model specific args from the desired class. If
    wandb is enabled, it adds checkpointer args. Finally, adds additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    # add shared arguments
    dataset_args(parser)
    augmentations_args(parser)
    custom_dataset_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # add method-specific arguments
    parser.add_argument("--method", default='dino', type=str)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add model specific args
    parser = METHODS[temp_args.method].add_model_specific_args(parser)
    parser = add_swag_args(parser)

    # add auto checkpoint/umap args
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--auto_umap", action="store_true")
    parser.add_argument("--logdir_init", default='./', type=str)
    parser.add_argument("--path", default= '/path/to/your/dir', type=str)
    parser.add_argument("--load_weights", action='store_true')
    parser.add_argument('--wandb_key', type=str, default=None,
                        help='The personal key for wandb ')
    temp_args, _ = parser.parse_known_args()

    # optionally add checkpointer and AutoUMAP args
    if temp_args.save_checkpoint:
        parser = Checkpointer.add_checkpointer_args(parser)

    if _umap_available and temp_args.auto_umap:
        parser = AutoUMAP.add_auto_umap_args(parser)

    # parse args
    args = parser.parse_args()
    args.gpus = 0
    # prepare arguments with additional setup
    additional_setup_pretrain(args)

    return args


def parse_args_linear() -> argparse.Namespace:
    """Parses feature extractor, dataset, pytorch lightning, linear eval specific and additional args.

    First adds and arg for the pretrained feature extractor, then adds dataset, pytorch lightning
    and linear eval specific args. If wandb is enabled, it adds checkpointer args. Finally, adds
    additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_feature_extractor", type=str)

    # add shared arguments
    dataset_args(parser)
    linear_augmentations_args(parser)
    custom_dataset_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # linear model
    parser = METHODS["linear"].add_model_specific_args(parser)

    # THIS LINE IS KEY TO PULL WANDB AND SAVE_CHECKPOINT
    parser.add_argument("--save_checkpoint", action="store_true")

    temp_args, _ = parser.parse_known_args()
    # parser.set_defaults(max_epochs = 250)
    # optionally add checkpointer
    if temp_args.save_checkpoint:
        parser = Checkpointer.add_checkpointer_args(parser)

    # parse args
    args = parser.parse_args()
    additional_setup_linear(args)

    return args
