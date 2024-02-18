import argparse
import os, sys
import time, copy
import tabulate

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import pickle

import utils.utils as utils
from utils.swag import swag_utils

import warnings
warnings.filterwarnings('ignore')



parser = argparse.ArgumentParser(description="Run BMA")

parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")

parser.add_argument("--method", default='swag', choices=['swag', 'last-swag'])

## Data ---------------------------------------------------------------------
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

parser.add_argument("--use_validation", action='store_true',
            help ="Use validation for hyperparameter search (Default : False)")

parser.add_argument("--dat_per_cls", type=int, default=-1,
            help="Number of data points per class in few-shot setting. -1 denotes deactivate few-shot setting (Default : -1)")
#----------------------------------------------------------------

## Model ---------------------------------------------------------
parser.add_argument(
    "--model",
    type=str, default='resnet18-noBN',
    choices=['resnet18-noBN'],
    help="model name (default : resnet18-noBN)")

parser.add_argument(
    "--pre_trained", action='store_true', default=False,
    help="Using pre-trained model from zoo"
    )

parser.add_argument("--load_path",
            type=str, default="/data2/lsj9862/best_result/bma_num_plot/swag-sgd_best_val_model.pt", 
            help="Path for model")

parser.add_argument("--bma_load_path",
            type=str, default="/data2/lsj9862/best_result/bma_num_plot/bma_models",
            help="Path for sampled model")

parser.add_argument("--performance_path", type=str, default="/data2/lsj9862/best_result/bma_num_plot/performance/performance_final.pt",
            help="Path for performance (model_num, accuracy, ece, nll, flat_idx, rand_idx, sharp_idx, tr_cum_eign, tr_max_eign)")
#----------------------------------------------------------------

parser.add_argument("--num_bins", type=int, default=15, help="bin number for ece")
parser.add_argument("--eps", type=float, default=1e-8, help="small float to calculate nll")

args = parser.parse_args()
# ---------------------------------------------------------------



# Set Device and Seed--------------------------------------------
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model.split("-")[-1] == "noBN":
    args.batch_norm = False
    args.aug = False
else:
    args.batch_norm = True
    args.aug = True

utils.set_seed(args.seed)

print(f"Device : {args.device} / Seed : {args.seed} / Augmentation {args.aug}")
print("-"*30)
#------------------------------------------------------------------


# Set Save path ---------------------------------------------------
args.save_path = os.path.dirname(args.performance_path)
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

if args.method == "swag":
    args.tr_layer = "full_layer"
elif args.method == "last_swag":
    args.tr_layer = "last_layer"
    
if args.tr_layer == "last_layer":
    for name, _ in model.named_modules():
        tr_layer_name = name
elif args.tr_layer == "last_block":
    raise NotImplementedError("need code for last block")
else:
    tr_layer_name = None
#-------------------------------------------------------------------


# Load flat/rand/shapr indices -------------------------------------
perf_dict = torch.load(args.performance_path)    
flat_indices = perf_dict['flat_idx']
rand_indices = perf_dict['rand_idx']
sharp_indices = perf_dict['sharp_idx']
#-------------------------------------------------------------------

# Load BMA models paths --------------------------------------------
flat_bma_paths = list(); rand_bma_paths = list(); sharp_bma_paths = list()
for flat_idx in flat_indices:
    flat_bma_paths.append(f'bma_model-{flat_idx}.pt')
for rand_idx in rand_indices:
    rand_bma_paths.append(f'bma_model-{rand_idx}.pt')
for sharp_idx in sharp_indices:
    sharp_bma_paths.append(f'bma_model-{sharp_idx}.pt')
#-------------------------------------------------------------------

## Run BMA -------------------------------------------------------
for type_, bma_paths in enumerate([flat_bma_paths, rand_bma_paths, sharp_bma_paths]):
    if type_==0:
        name = "flat"
    elif type_==1:
        name = "rand"
    elif type_==2:
        name = "sharp"
    
    num_list = []; acc_list = [] ; nll_list = [] ; ece_list = []
    for i in range(len(bma_paths)+1):
        if i ==0:
            pass
        else:
            bma_paths.pop(0)

        bma_models = bma_paths

        bma_predictions = np.zeros((len(te_loader.dataset), num_classes))
        with torch.no_grad():      
            for j, bma_model in enumerate(bma_models):
                # get sampled model
                bma_sample = torch.load(f"{args.bma_load_path}/{bma_model}")
                bma_state_dict = utils.list_to_state_dict(model, bma_sample, args.tr_layer, tr_layer_name)
                model.load_state_dict(bma_state_dict, strict=False)

                model.eval()
                if args.batch_norm:
                    swag_utils.bn_update(tr_loader, model, verbose=False, subset=1.0)
                res = swag_utils.predict(te_loader, model)

                predictions = res["predictions"]
                targets = res["targets"]

                accuracy = np.mean(np.argmax(predictions, axis=1) == targets)
                nll = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), targets] + args.eps))
                print(
                    "SWAG Sample %d/%d. Accuracy: %.2f%% NLL: %.4f"
                    % (j + 1, len(bma_models), accuracy * 100, nll)
                )

                bma_predictions += predictions

                ens_accuracy = np.mean(np.argmax(bma_predictions, axis=1) == targets)
                ens_nll = -np.mean(
                    np.log(
                        bma_predictions[np.arange(bma_predictions.shape[0]), targets] / (i + 1)
                        + 1e-8
                    )
                )
                print(
                    "Ensemble %d/%d. Accuracy: %.2f%% NLL: %.4f"
                    % (j + 1, len(bma_models), ens_accuracy * 100, ens_nll)
                )

            bma_predictions /= len(bma_models)

            bma_accuracy = np.mean(np.argmax(bma_predictions, axis=1) == targets)
            bma_nll = -np.mean(
                np.log(bma_predictions[np.arange(bma_predictions.shape[0]), targets] + 1e-8)
            )

            unc = utils.calibration_curve(bma_predictions, targets, num_bins=args.num_bins)
            ece = unc["ece"]

            print(f"bma Accuracy using {len(bma_models)} models : {bma_accuracy * 100:.2f}% / NLL : {bma_nll:.4f} / ECE {ece:.4f}")
            
            num_list.append(len(bma_models))
            acc_list.append(bma_accuracy)
            nll_list.append(bma_nll)
            ece_list.append(ece)

    dict = {
        'num_models' : num_list,
        'acc' : acc_list,
        'nll' : nll_list,
        'ece' : ece_list
    }

    torch.save(dict, f"{args.save_path}/{name}_bma_num.pt")
    print(f"Save {i}-th model result")
    print("-"*30)
    
print("Complete all BMA!!")