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

parser.add_argument("--optim", type=str, default="sgd",
                    choices=["sgd", "sam", "fsam", "bsam"],
                    help="Optimization options")

## Data ---------------------------------------------------------------------
parser.add_argument(
    "--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"],
                    help="dataset name")

parser.add_argument(
    "--data_path",
    type=str,
    default='/data1/lsj9862/cifar10',
    help="path to datasets location",)

parser.add_argument("--batch_size", type=int, default=256,
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

parser.add_argument("--model_path", type=str, default=None, required=True,
            help="Path to model state dict")
#----------------------------------------------------------------

parser.add_argument("--num_bins", type=int, default=50, help="bin number for ece")


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
if args.model == "mlp":
    from models import mlp
    model = mlp.MLP(output_size=num_classes).to(args.device)
elif args.model in ["resnet18", "resnet18-noBN"]:
    from torchvision.models import resnet18
    model = resnet18(pretrained=False, num_classes=num_classes).to(args.device)
elif args.model in ["resnet50", "resnet50-noBN"]:
    from torchvision.models import resnet50
    model = resnet50(pretrained=False, num_classes=num_classes).to(args.device)
elif args.model in ["wideresnet40x10", "wideresnet40x10-noBN"]:
    from models import wide_resnet
    model_cfg = getattr(wide_resnet, "WideResNet40x10")
    model = model_cfg.base(num_classes=num_classes).to(args.device)
elif args.model in ["wideresnet28x10", "wideresnet28x10-noBN"]:
    from models import wide_resnet
    model_cfg = getattr(wide_resnet, "WideResNet28x10")
    model = model_cfg.base(num_classes=num_classes).to(args.device)

print(f"Preparing model {args.model}")
#----------------------------------------------------------------

# %%
#%%

## Run BMA -------------------------------------------------------
bma_models = os.listdir(args.bma_load_path)


bma_predictions = np.zeros((len(te_loader.dataset), num_classes))
with torch.no_grad():
    
    # Load Model State dict
    model.load_state_dict(torch.load(args.model_path))
    model_state_dict = model.state_dict()
    
    for i, bma_model in enumerate(bma_models):
        # get sampled model
        bma_sample = torch.load(f"{args.bma_load_path}/{bma_model}")
        bma_state_dict = utils.list_to_state_dict(model, bma_sample)

        model_state_dict.update(bma_state_dict)
        model.load_state_dict(model_state_dict)

        if args.batch_norm:
            swag_utils.bn_update(tr_loader, model, verbose=False, subset=1.0)
        res = swag_utils.predict(te_loader, model, verbose=False)

        predictions = res["predictions"]
        targets = res["targets"]

        accuracy = np.mean(np.argmax(predictions, axis=1) == targets)
        nll = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), targets] + 1e-8))
        print(
            "SWAG Sample %d/%d. Accuracy: %.2f%% NLL: %.4f"
            % (i + 1, len(bma_models), accuracy * 100, nll)
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
            % (i + 1, len(bma_models), ens_accuracy * 100, ens_nll)
        )

    bma_predictions /= len(bma_models)

    bma_accuracy = np.mean(np.argmax(bma_predictions, axis=1) == targets)
    bma_nll = -np.mean(
        np.log(bma_predictions[np.arange(bma_predictions.shape[0]), targets] + 1e-8)
    )

    unc = utils.calibration_curve(bma_predictions, targets, num_bins=args.num_bins)
    ece = unc["ece"]

    # utils.save_reliability_diagram(args.method, args.optim, './', unc, True)


print(f"bma Accuracy using {len(bma_models)} models : {bma_accuracy * 100:.2f}% / NLL : {bma_nll:.4f} / ECE {ece:.4f}")