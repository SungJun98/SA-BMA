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
                    choices=["emcmc"],
                    help="Learning Method")

parser.add_argument("--optim", type=str, default="sgd",
                    choices=["sgld"],
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

parser.add_argument("--dat_per_cls", type=int, default=-1,
            help="Number of data points per class in few-shot setting. -1 denotes deactivate few-shot setting (Default : -1)")

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
## Load Distributional shifted data
# if args.model == 'vitb16-i21k':
#     is_backbone_vit = True
# else:
#     is_backbone_vit = False

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
print("Load Best Validation Model (Lowest Loss)")

bma_load_paths = sorted(os.listdir(args.load_path))

bma_logits = np.zeros((len(ood_loader.dataset), num_classes))
bma_predictions = np.zeros((len(ood_loader.dataset), num_classes))

for path in bma_load_paths:
    try:
        bma_sample = torch.load(f"{args.load_path}/{path}", map_location=args.device)
    except:
        pass
    model.load_state_dict(bma_sample)
    model.to(args.device)        

    #### Bayesian Model Averaging
    loss_sum = 0.0
    num_objects_total = len(ood_loader.dataset)

    logits = list()
    preds = list()
    targets = list()

    model.eval()
    offset = 0
    with torch.no_grad():
        for _, (input, target) in enumerate(ood_loader):
            input, target = input.to(args.device), target.to(args.device)
            pred = model(input)
            loss = criterion(pred, target)
            loss_sum += loss.item() * input.size(0)
            
            logits.append(pred.cpu().numpy())
            preds.append(F.softmax(pred, dim=1).cpu().numpy())
            targets.append(target.cpu().numpy())
            offset += input.size(0)
    
    logits = np.vstack(logits)
    preds = np.vstack(preds)
    targets = np.concatenate(targets)

    bma_logits += logits
    bma_predictions += preds

bma_predictions /= len(bma_load_paths)

bma_accuracy = np.mean(np.argmax(bma_predictions, axis=1) == targets) * 100
bma_nll = -np.mean(np.log(bma_predictions[np.arange(bma_predictions.shape[0]), targets] + args.eps))
bma_unc = utils.calibration_curve(bma_predictions, targets, args.num_bins)
bma_ece = bma_unc['ece']


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