import argparse
import os, sys
import time, copy

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import pickle

import utils, data #, losses

from baselines.sam.sam import SAM, FSAM
from baselines.swag import swag, swag_utils

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description="Run BMA")

parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")

parser.add_argument("--method", type=str, default="dnn",
                    choices=["dnn", "swag", "sabtl"],
                    help="Learning Method")

## Data ---------------------------------------------------------------------
parser.add_argument(
    "--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"],
                    help="dataset name")

parser.add_argument(
    "--data_path",
    type=str,
    default='/data1/lsj9862/cifar10',
    help="path to datasets location",)

parser.add_argument("--batch_size", type=int, default=64,
            help="batch size")

parser.add_argument("--num_workers", type=int, default=4,
            help="number of workers")

parser.add_argument("--use_validation", action='store_true',
            help ="Use validation for hyperparameter search (Default : False)")
#----------------------------------------------------------------

## Model ---------------------------------------------------------
parser.add_argument(
    "--model",
    type=str, default='mlp', required=True,
    choices=['mlp', 'resnet18', 'resnet50', 'wideresnet28x10', 'wideresnet40x10',
            'resnet18-noBN', 'resnet50-noBN', 'wideresnet28x10-noBN', 'wideresnet40x10-noBN'],
    help="model name (default : mlp)")

parser.add_argument("--bma_load_path",
            type=str, default=None, required=True,
            help="Path for sampled model")

parser.add_argument("--idx_path", type=str, default=None, required=True,
            help="Path for idx pickle file")
#----------------------------------------------------------------

parser.add_argument("--num_bins", type=int, default=50, help="bin number for ece")


parser.add_argument("--mode", type=str, default="flat", required=True,
                choices=["flat", "sharp", "rand"])

args = parser.parse_args()
# ---------------------------------------------------------------



args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {args.device}")


if args.model.split("-")[-1] == "noBN":
    args.batch_norm = False
else:
    args.batch_norm = True

utils.set_seed(args.seed)


# Load Data ------------------------------------------------------
if args.dataset == 'cifar10':
    tr_loader, val_loader, te_loader, num_classes = data.get_cifar10(args.data_path, args.batch_size,
                                                                    args.num_workers,
                                                                    use_validation = args.use_validation)
elif args.dataset == 'cifar100':
    tr_loader, val_loader, te_loader, num_classes = data.get_cifar100(args.data_path, args.batch_size,
                                                                    args.num_workers,
                                                                    use_validation = args.use_validation)

if not args.use_validation:
    val_loader = te_loader

print(f"Load Data : {args.dataset}")
#----------------------------------------------------------------


# Define Model------------------------------------------------------
model = utils.get_backbone(args.model, num_classes, args.device)
#----------------------------------------------------------------




with open(f"{args.idx_path}/{args.mode}_idx_list.pickle", 'rb') as f:
    sorted_idx = pickle.load(f)


bma_paths = list()
for idx in sorted_idx:
    bma_paths.append(f'bma_model-{idx}.pt')

## Run BMA -------------------------------------------------------
num_list = []; acc_list = [] ; nll_list = [] ; ece_list = []
for i in range(len(bma_paths)+1):
    if i ==0:
        pass
    else:
        bma_paths.pop(0)

    bma_models =bma_paths

    bma_predictions = np.zeros((len(te_loader.dataset), num_classes))
    with torch.no_grad():      
        for j, bma_model in enumerate(bma_models):
            # get sampled model
            bma_sample = torch.load(f"{args.bma_load_path}/{bma_model}")
            bma_state_dict = utils.list_to_state_dict(model, bma_sample)
            model.load_state_dict(bma_state_dict)

            if args.batch_norm:
                swag_utils.bn_update(tr_loader, model, verbose=False, subset=1.0)
            res = swag_utils.predict(te_loader, model, verbose=False)

            predictions = res["predictions"]
            targets = res["targets"]

            accuracy = np.mean(np.argmax(predictions, axis=1) == targets)
            nll = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), targets] + 1e-8))
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


    with open(f'{args.idx_path}/{args.mode}_bma_num.pickle', 'wb') as f:
        pickle.dump(dict, f)
    print(f"Save {i}-th model result")
    print("-"*30)
    
print("Complete all BMA!!")