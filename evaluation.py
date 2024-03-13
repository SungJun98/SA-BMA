import argparse
import os, sys
import time, copy
import tabulate

import torch
import torch.nn.functional as F

import utils.utils as utils
from utils.swag import swag, swag_utils
from utils.vi import vi_utils
from utils.la import la_utils
from utils.sabma import sabma_utils
from utils import temperature_scaling as ts
import utils.data.data as data

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

def save_to_csv_accumulated(df, filename):
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, index=False)




parser = argparse.ArgumentParser(description="training baselines")

parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")

parser.add_argument("--method", type=str, default="dnn",
                    choices=["dnn", "swag", "ll_swag", "vi", "ll_vi", "la", "ll_la", "sabma", "ptl"],
                    help="Learning Method")

parser.add_argument("--optim", type=str, default="sgd",
                    choices=["sgd", "sgld", "sam", "fsam", "bsam", "sabma"],
                    help="Learning Method")

parser.add_argument("--load_path", type=str, default=None,
                    help="Path to test")

parser.add_argument("--save_path", type=str, default=None,
                    help="Path to save result")

## Data ---------------------------------------------------------
parser.add_argument(
    "--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"],
                    help="dataset name")

parser.add_argument(
    "--data_path",
    type=str,
    default='/data1/lsj9862/data',
    help="path to datasets location",)

parser.add_argument("--batch_size", type=int, default=2048,
            help="batch size")

parser.add_argument("--num_workers", type=int, default=4,
            help="number of workers")

parser.add_argument("--use_validation", action='store_true', default=True,
            help ="Use validation for hyperparameter search (Default : False)")

parser.add_argument("--dat_per_cls", type=int, default=10,
            help="Number of data points per class in few-shot setting. -1 denotes deactivate few-shot setting (Default : 10)")

parser.add_argument("--no_aug", action="store_true", default=False,
            help="Deactivate augmentation")
#----------------------------------------------------------------

## Model ---------------------------------------------------------
parser.add_argument(
    "--model",
    type=str, default='resnet18', required=True,
    choices=['resnet18', 'resnet50', 'resnet101',
            'resnet18-noBN', "vitb16-i1k", "vitb16-i21k"],
    help="model name (default : resnet18)")

parser.add_argument(
    "--pre_trained", action='store_true', default=True,
    help="Using pre-trained model from zoo"
    )
#----------------------------------------------------------------

## SABMA---------------------------------------------------------
parser.add_argument("--tr_layer", type=str, default="nl_ll",
        help="Traning layer of SABMA")
#----------------------------------------------------------------

## bma or metrics -----------------------------------------------
parser.add_argument("--eps", type=float, default=1e-8, help="small float to calculate nll")
parser.add_argument("--bma_num_models", type=int, default=30, help="Number of models for bma")
parser.add_argument("--num_bins", type=int, default=15, help="bin number for ece")
parser.add_argument("--no_save_bma", action='store_true', default=False,
            help="Deactivate saving model samples in BMA")
#----------------------------------------------------------------

## OOD test -----------------------------------------------------
parser.add_argument("--corrupt_option",
    default=['brightness.npy','contrast.npy','defocus_blur.npy','elastic_transform.npy','fog.npy',
    'frost.npy','gaussian_blur.npy','gaussian_noise.npy','glass_blur.npy','impulse_noise.npy','jpeg_compression.npy',
    'motion_blur.npy','pixelate.npy','saturate.npy','shot_noise.npy','snow.npy','spatter.npy','speckle_noise.npy','zoom_blur.npy'],
    help='corruption option of CIFAR10/100-C'
        )
parser.add_argument("--severity",
    default=1,
    type=int,
    help='Severity of corruptness in CIFAR10/100-C (1 to 5)')
#----------------------------------------------------------------

args = parser.parse_args()
#----------------------------------------------------------------

# Set Device and Seed--------------------------------------------
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model.split("-")[-1] == "noBN":
    args.batch_norm = False
    args.aug = False
else:
    args.batch_norm = True
    args.aug = True

if args.no_aug:
    args.aug = False

utils.set_seed(args.seed)

print(f"Device : {args.device} / Seed : {args.seed}")
print("-"*30)
#------------------------------------------------------------------

# Set BMA and Save Setting-----------------------------------------
if args.method == 'dnn':
    args.bma_num_models = 1
args.ignore_wandb = True
#------------------------------------------------------------------


# Load Data --------------------------------------------------------
data_path_ood = args.data_path
args.data_path = os.path.join(args.data_path, args.dataset)
tr_loader, val_loader, _, num_classes = utils.get_dataset(dataset=args.dataset,
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
model = utils.get_backbone(args.model, num_classes, args.device, True)

swag_model=None
if args.method == "swag":
    swag_model = swag.SWAG(copy.deepcopy(model),
                        no_cov_mat=False,
                        max_num_models=5,
                        last_layer=False).to(args.device)
    print("Preparing SWAG model")

print("-"*30)
#-------------------------------------------------------------------

# Set Criterion-----------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()
print("Set Criterion as Cross Entropy")
print("-"*30)
#-------------------------------------------------------------------


method = args.method
if args.method == 'ptl':
    args.method = 'dnn'

## Test ------------------------------------------------------------------------------------------------------
##### Get test nll, Entropy, ece, Reliability Diagram on best model
## Load Distributional shifted data
if args.model in ['vitb16-i1k', 'vitb16-i21k']:
    is_backbone_vit = True
else:
    is_backbone_vit = False

if args.dataset == 'cifar10':
    ood_loader = data.corrupted_cifar10(data_path=data_path_ood,
                            corrupt_option=args.corrupt_option,
                            severity=args.severity,
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers,
                            is_vit=is_backbone_vit)
elif args.dataset == 'cifar100':
    ood_loader = data.corrupted_cifar100(data_path=data_path_ood,
                            corrupt_option=args.corrupt_option,
                            severity=args.severity,
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers,
                            is_vit=is_backbone_vit)


### Load Best Model
if args.method != 'sabma':
    state_dict_path = f'{args.load_path}/{method}-{args.optim}_best_val.pt'
    checkpoint = torch.load(state_dict_path)
else:
    model = torch.load(f'{args.load_path}/{args.method}-{args.optim}_best_val_model.pt')
    

    
mean = None; variance = None
if args.method in ["swag", "ll_swag"]:
    swag_model.load_state_dict(checkpoint["state_dict"])
    model = swag_model
    
elif args.method in ["vi", "ll_vi"]:
    model = utils.get_backbone(args.model, num_classes, args.device, True)
    if args.method == "ll_vi":
        vi_utils.make_ll_vi(args, model)
    vi_utils.load_vi(model, checkpoint)
    mean = torch.load(f'{args.load_path}/{args.method}-{args.optim}_best_val_mean.pt')
    variance = torch.load(f'{args.load_path}/{args.method}-{args.optim}_best_val_variance.pt')
    
elif args.method == 'dnn':
    if method == 'dnn':
        model.load_state_dict(checkpoint["state_dict"])
    elif method == 'ptl':
        for key in list(checkpoint.keys()):
            if 'backbone.' in key:
                new_key = key.replace('backbone.', '')
                checkpoint[new_key] = checkpoint.pop(key)
            elif 'classifier.' in key:
                if 'vitb16-i21k' in args.model:
                    new_key = key.replace('classifier', 'head')
                    checkpoint[new_key] = checkpoint.pop(key)
                elif 'vitb16-i1k' in args.model:
                    new_key = key.replace('classifier', 'heads.head')
                    checkpoint[new_key] = checkpoint.pop(key)
                elif 'resnet' in args.model:
                    new_key = key.replace('classifier', 'fc')
                    checkpoint[new_key] = checkpoint.pop(key)
                else:
                    raise NotImplementedError("No code to load this backbone")
        model.load_state_dict(checkpoint)
else:
    pass
model.to(args.device)        
print("Load Best Validation Model (Lowest Loss)")


if args.method != 'sabma':
    #### MAP
    res = utils.no_ts_map_estimation(args, ood_loader, num_classes, model, mean, variance, criterion)
    table = [["OOD Accuracy", "OOD NLL", "OOD ECE"],
            [format(res['accuracy'], '.2f'), format(res['nll'], '.4f'), format(res['ece'], '.4f')]]
    print(tabulate.tabulate(table, tablefmt="simple", floatfmt="8.4f"))

    #### Bayesian Model Averaging
    if args.no_save_bma:
        bma_save_path  = None
    else:
        bma_save_path = f"{args.save_path}/bma_models"
        os.makedirs(bma_save_path, exist_ok=True)
    
    if args.method in ["swag", "ll_swag", "vi", "ll_vi"]:
        utils.set_seed(args.seed)
        print(f"Start Bayesian Model Averaging with {args.bma_num_models} samples")
        res, bma_accuracy, bma_nll, bma_ece, bma_accuracy_ts, bma_nll_ts, bma_ece_ts, temperature = utils.bma(args, tr_loader, val_loader, ood_loader, num_classes, model, mean, variance, criterion, bma_save_path, temperature=None)
    else:
        pass


else:
    ### BMA prediction
    if args.no_save_bma:
        bma_save_path  = None
    else:
        bma_save_path = f"{args.save_path}/bma_models"
        os.makedirs(bma_save_path, exist_ok=True)

    ## BMA result w/o Ts
    res = sabma_utils.bma_sabma(ood_loader, model, args.bma_num_models,
                        num_classes, criterion, args.device,
                        bma_save_path=bma_save_path, eps=args.eps, num_bins=args.num_bins,
                        validation=False, tr_layer=args.tr_layer)
    
    bma_logits = res["logits"]
    bma_predictions = res["predictions"]
    bma_targets = res["targets"]

    bma_accuracy = res["accuracy"]
    bma_nll = res["nll"]
    unc = utils.calibration_curve(bma_predictions, bma_targets, args.num_bins)
    bma_ece = res['ece']

    print("[BMA w/o TS Results]\n")
    tab_name = ["# of Models", "BMA Accuracy", "BMA NLL", "BMA ECE"]
    tab_contents = [args.bma_num_models, format(bma_accuracy, '.2f'), format(bma_nll, '.4f'), format(bma_ece, '.4f')]
    table = [tab_name, tab_contents]
    print(tabulate.tabulate(table, tablefmt="simple"))
    print("-"*30)

if args.corrupt_option == ['brightness.npy','contrast.npy','defocus_blur.npy','elastic_transform.npy','fog.npy',
    'frost.npy','gaussian_blur.npy','gaussian_noise.npy','glass_blur.npy','impulse_noise.npy','jpeg_compression.npy',
    'motion_blur.npy','pixelate.npy','saturate.npy','shot_noise.npy','snow.npy','spatter.npy','speckle_noise.npy','zoom_blur.npy']:
    corr = 'all'
else:
    corr = args.corrupt_option
  
if method == 'ptl':
    args.method = 'ptl'      

result_df = pd.DataFrame({"method" : [args.method],
                "optim" : [args.optim],
                "seed" : [args.seed],
                "dataset" : [args.dataset],
                "dat_per_cls" : [args.dat_per_cls],
                "corrupt_option" : [corr],
                "severity" : [args.severity],
                "OOD Accuracy" : [res['accuracy']],
                "OOD NLL" : [res['nll']],
                "OOD ECE" : [res['ece']],
                })
   
save_to_csv_accumulated(result_df, args.save_path)