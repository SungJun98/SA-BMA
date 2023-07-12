import argparse
import os, sys
import time, copy
import tabulate

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import wandb

import utils.utils as utils
from utils.swag import swag, swag_utils

import warnings
warnings.filterwarnings('ignore')

# wandb
#############################################
## wandb 아이디 적기
#############################################
wandb.init(project=, entity=)
#############################################
#############################################

parser = argparse.ArgumentParser(description="training baselines")

parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")

parser.add_argument("--no_amp", action="store_true", default=False, help="Deactivate AMP")

parser.add_argument("--linear_probe", action="store_true", default=False,
        help = "When we do Linear Probing (Default : False)")

## Data ---------------------------------------------------------
parser.add_argument(
    "--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"],
                    help="dataset name")

parser.add_argument(
    "--data_path",
    type=str,
    default='/data1/lsj9862/data/cifar10',
    help="path to datasets location",)

parser.add_argument("--batch_size", type=int, default=256,
            help="batch size")

parser.add_argument("--num_workers", type=int, default=4,
            help="number of workers")

parser.add_argument("--use_validation", action='store_true', default=True,
            help ="Use validation for hyperparameter search (Default : True)")

parser.add_argument("--dat_per_cls", type=int, default=10,
            help="Number of data points per class in few-shot setting. -1 denotes deactivate few-shot setting (Default : -1)")
#----------------------------------------------------------------

## Model ---------------------------------------------------------
parser.add_argument(
    "--model",
    type=str, default='resnet18', required=True,
    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'resnet18-noBN',
            "vitb16-i21k"],
    help="model name (default : resnet18)")

parser.add_argument(
    "--pre_trained", action='store_true', default=True,
    help="Using pre-trained model from zoo"
    )

#######################################################################
## 결과 저장할 경로지정해주기
#######################################################################
parser.add_argument("--save_path",
            type=str, required=True,
            help="Path to save best model dict")
#######################################################################
#######################################################################
#----------------------------------------------------------------

## Optimizer Hyperparameter --------------------------------------
parser.add_argument("--optim", type=str, default="sgd",
                    choices=["sgd", "sam", "adam"],
                    help="Optimization options")

parser.add_argument("--lr_init", type=float, default=0.01,
                help="learning rate (Default : 0.01)")

parser.add_argument("--momentum", type=float, default=0.9,
                help="momentum (Default : 0.9)")

parser.add_argument("--epochs", type=int, default=300, metavar="N",
    help="number epochs to train (default : 300)")

parser.add_argument("--wd", type=float, default=5e-4, help="weight decay (default: 5e-4)")

parser.add_argument("--rho", type=float, default=0.05, help="size of pertubation ball for SAM / BSAM")
#----------------------------------------------------------------

## SWAG ---------------------------------------------------------
parser.add_argument("--swa_start", type=int, default=161, help="Start epoch of SWAG")
parser.add_argument("--swa_lr", type=float, default=0.05, help="Learning rate for SWAG")
parser.add_argument("--diag_only", action="store_true", default=False, help="Calculate only diagonal covariance")
parser.add_argument("--swa_c_epochs", type=int, default=1, help="Cycle to calculate SWAG statistics")
parser.add_argument("--max_num_models", type=int, default=20, help="Number of models to get SWAG statistics")

parser.add_argument("--swag_resume", type=str, default=None,
    help="path to load saved swag model to resume training (default: None)",)
#----------------------------------------------------------------

## bma or metrics -----------------------------------------------
parser.add_argument("--val_mc_num", type=int, default=1, help="number of MC sample in validation phase")
parser.add_argument("--eps", type=float, default=1e-8, help="small float to calculate nll")
parser.add_argument("--bma_num_models", type=int, default=30, help="Number of models for bma")
parser.add_argument("--num_bins", type=int, default=50, help="bin number for ece")
#----------------------------------------------------------------

args = parser.parse_args()
#----------------------------------------------------------------

# Set Device and Seed--------------------------------------------
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.batch_norm = True
args.aug = True

utils.set_seed(args.seed)

print(f"Device : {args.device} / Seed : {args.seed} / Augmentation {args.aug}")
print("-"*30)
#------------------------------------------------------------------

# Set BMA and Save Setting-----------------------------------------
args.save_path = utils.set_save_path(args)
print(f"Save Results on {args.save_path}")
print("-"*30)
#------------------------------------------------------------------

# wandb config-----------------------------------------------------
wandb.config.update(args)
wandb.run.name = utils.set_wandb_runname(args)
#------------------------------------------------------------------

# Load Data --------------------------------------------------------
tr_loader, val_loader, te_loader, num_classes = utils.get_dataset(dataset=args.dataset,
                                                        data_path=args.data_path,
                                                        dat_per_cls=args.dat_per_cls,
                                                        use_validation=args.use_validation,
                                                        batch_size=args.batch_size,
                                                        num_workers=args.num_workers,
                                                        seed=args.seed,
                                                        aug=args.aug,
                                                        )

if args.dat_per_cls >= 0:
    print(f"Load Data : {args.dataset}-{args.dat_per_cls}shot")
else:
    print(f"Load Data : {args.dataset}")
print("-"*30)
#------------------------------------------------------------------

# Define Model-----------------------------------------------------
model = utils.get_backbone(args.model, num_classes, args.device, args.pre_trained)
print("-"*30)
#-------------------------------------------------------------------


"""
코드 요청드리는 부분
0. wandb 정보 먼저 기입!

1. Pre-Train Your Loss에서 amp를 쓰는지 모르겠네요...한 번 확인해주시고 안 쓴다면 amp 적용해주세요!
  - amp 적용은 utils.set_scaler 함수를 한 번 보면 도움될 것

2. 지금 data, dnn model은 load를 한 상태입니다. 지금 run_ptl.py에서 해도 되고 여기에 제공드렸던 data, model load하는 부분을 pre-train your loss 코드에 붙이셔도 됩니다.
제 생각에는 pre-train your loss 코드 부분에 저희 model 로드하는 부분 + data 로드하는 부분을 붙이는게 더 편하지 않을까 생각됩니다. (물론 공저자분께서 편하신대로!)

3. 1) source task에서 prior 만드는 step, 2) downstream task에서 fine-tuning하는 step을 해주셔야 됩니다.

4. 1) prior 만드는 step에서 몇 epoch 하는지에 따라 SWAG 관련 hyperparameter도 바뀔겁니다. 따라서 몇 epoch하는지 먼저 정해지면 그 때 저에게 말씀해주세요!
  - epoch뿐만 아니라 다른 hyperparameter도 논문에 있는거 바탕으로 저에게 한 번 말씀해주시면 좋을 것 같습니다.
  - 논문에 기재된걸 확인해봐야될 것 같아요. 그 부분 확인 한 번 부탁드립니다.

5. 2) downstream task에서 fine-tuning하는건 우선 prior 만드는 step 처리하고 얘기하시죠!
"""