import argparse
import os, sys
import time, copy
import tabulate

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

import utils.utils as utils



import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="training baselines")

parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")

parser.add_argument("--method", type=str, default="dnn",
                    choices=["dnn", "swag", "ll_swag", "vi", "ll_vi", "la", "ll_la"],
                    help="Learning Method")

parser.add_argument("--resume", type=str, default=None,
    help="path to load saved model to resume training (default: None)",)


## Data ---------------------------------------------------------
parser.add_argument(
    "--dataset", type=str, default="cifar10_c", choices=["cifar10_c", "cifar100_c", "imagenet_v2", "imagenet_c"],
                    help="dataset name")

parser.add_argument(
    "--data_dir",
    type=str,
    default='/data1/lsj9862/data/',
    help="path to datasets location",)

parser.add_argument("--severity", type=int, default=1,
            help="Set severity of corruption")

parser.add_argument("--batch_size", type=int, default=256,
            help="batch size")

parser.add_argument("--num_workers", type=int, default=4,
            help="number of workers")
#----------------------------------------------------------------

## Model ---------------------------------------------------------
parser.add_argument(
    "--model",
    type=str, default='resnet18', required=True,
    choices=['resnet18', 'resnet50', 'resnet101', 'resnet152',
            'resnet18-noBN',
            "vitb16-i21k"],
    help="model name (default : resnet18)")
#----------------------------------------------------------------

## bma or metrics -----------------------------------------------
parser.add_argument("--val_mc_num", type=int, default=1, help="number of MC sample in validation phase")
parser.add_argument("--eps", type=float, default=1e-8, help="small float to calculate nll")
parser.add_argument("--bma_num_models", type=int, default=30, help="Number of models for bma")
parser.add_argument("--num_bins", type=int, default=15, help="bin number for ece")
#----------------------------------------------------------------

"""
distributional shift에서는 validation set 없다는 가정 ==> ts 적용 불가
"""

args = parser.parse_args()
#----------------------------------------------------------------

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
utils.set_seed(args.seed)
print(f"Device : {args.device} / Seed : {args.seed}")
print("-"*30)

# Set BMA and Save Setting-----------------------------------------
if args.method == 'dnn':
    args.bma_num_models = 1

"""
excel로 save할 수 있도록 코딩
"""
print(f"Save Results on {args.save_path}")
print("-"*30)
#------------------------------------------------------------------




# Load Model
model, mean, variance, best_epoch = utils.load_best_model(args, model, swag_model, num_classes)



# Test


## Test ------------------------------------------------------------------------------------------------------
##### Get test nll, Entropy, ece, Reliability Diagram on best model


#### MAP
## Unscaled Results
res = utils.no_ts_map_estimation(args, te_loader, num_classes, model, mean, variance, criterion)
print(f"1) Unscaled Results:")
table = [["Accuracy", "NLL", "Ece" ],
        [format(res['accuracy'], '.2f'), format(res['nll'], '.4f'), format(res['ece'], '.4f')]]
print(tabulate.tabulate(table, tablefmt="simple", floatfmt="8.4f"))

## Temperature Scaled Results
if not args.no_ts:
    res, temperature = utils.ts_map_estimation(args, val_loader, te_loader, num_classes, model, mean, variance, criterion)
    print(f"2) Scaled Results:")
    table = [["Accuracy", "NLL", "Ece", "Temperature"],
            [format(res['accuracy'], '.2f'), format(res['nll'],'.4f'), format(res['ece'], '.4f'), format(temperature.item(), '.4f')]]
    print(tabulate.tabulate(table, tablefmt="simple", floatfmt="8.4f"))
else:
    temperature = None

   #### Bayesian Model Averaging
if args.method in ["swag", "ll_swag", "vi", "ll_vi"]:
    utils.set_seed(args.seed)
    bma_save_path = f"{args.save_path}/bma_models"
    os.makedirs(bma_save_path, exist_ok=True) 
    print(f"Start Bayesian Model Averaging with {args.bma_num_models} samples")
    utils.bma(args, tr_loader, val_loader, te_loader, num_classes, model, mean, variance, criterion, bma_save_path, temperature)
else:
    pass
