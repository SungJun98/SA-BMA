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

parser.add_argument("--method", type=str, default="vi",
                    help="Learning Method")

parser.add_argument("--optim", type=str, default="bsam",
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

parser.add_argument("--batch_size", type=int, default=256,
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

## bma or metrics -----------------------------------------------
parser.add_argument("--eps", type=float, default=1e-8, help="small float to calculate nll")
parser.add_argument("--bma_num_models", type=int, default=32, help="Number of models for bma")
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
print("-"*30)
#-------------------------------------------------------------------

# Set Criterion-----------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()
print("Set Criterion as Cross Entropy")
print("-"*30)
#-------------------------------------------------------------------


## Test ------------------------------------------------------------------------------------------------------
##### Get test nll, Entropy, ece, Reliability Diagram on best model
if args.dataset == 'cifar10':
    ood_loader = data.corrupted_cifar10(data_path=data_path_ood,
                            corrupt_option=args.corrupt_option,
                            severity=args.severity,
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers,
                            resize=(not args.no_aug))
elif args.dataset == 'cifar100':
    ood_loader = data.corrupted_cifar100(data_path=data_path_ood,
                            corrupt_option=args.corrupt_option,
                            severity=args.severity,
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers,
                            resize=(not args.no_aug))


### Load Best Model
bma_load_paths = sorted(os.listdir(args.load_path))
bma_load_paths.pop(-1) # remove "performance" folder
bma_predictions = np.zeros((len(ood_loader.dataset), num_classes))

for i, path in enumerate(bma_load_paths):
    model_num = path.split(".")[0]
    model_num = model_num.split("_")[-1]
    
    bma_sample = torch.load(f"{args.load_path}/{path}").to(args.device)
    # model = bma_sample
    
    res = utils.eval(ood_loader, bma_sample, criterion, args.device)
    print(f"Sample {i+1}/{args.bma_num_models}. OOD Accuracy : {res['accuracy']:.2f}%. ECE : {res['ece']:.4f}. NLL : {res['nll']:.4f}")
    
    ## 결과 누적
    bma_predictions += res["predictions"]
    
    ens_accuracy = np.mean(np.argmax(bma_predictions, axis=1) == res["targets"]) * 100
    ens_nll = -np.mean(np.log(bma_predictions[np.arange(bma_predictions.shape[0]), res["targets"]] / (i + 1) + args.eps))        
    print(f"Ensemble {i+1}/{args.bma_num_models}. Accuracy: {ens_accuracy:.2f}% NLL: {ens_nll:.4f}")
    
    bma_predictions /= args.bma_num_models

bma_loss = criterion(torch.tensor(bma_predictions), torch.tensor(res['targets'])).item()
bma_accuracy = np.mean(np.argmax(bma_predictions, axis=1) == res["targets"]) * 100
bma_nll = -np.mean(np.log(bma_predictions[np.arange(bma_predictions.shape[0]), res["targets"]] + args.eps))        
unc = utils.calibration_curve(bma_predictions, res["targets"], args.num_bins)
bma_ece = unc['ece']


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
  

result_df = pd.DataFrame({"method" : [args.method],
                "optim" : [args.optim],
                "seed" : [args.seed],
                "dataset" : [args.dataset],
                "dat_per_cls" : [args.dat_per_cls],
                "corrupt_option" : [corr],
                "severity" : [args.severity],
                "OOD Accuracy" : [bma_accuracy],
                "OOD NLL" : [bma_nll],
                "OOD ECE" : [bma_ece],
                })
   
save_to_csv_accumulated(result_df, args.save_path)