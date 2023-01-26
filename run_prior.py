'''
Laplace Redux로 run prior를 하는 것도 코드에 추가해야될 듯
'''

import argparse
import os, sys
import time, copy
import tabulate

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import pickle
import wandb

import utils
import data
from models import mlp

from baselines.sam.sam import SAM, FSAM
from baselines.swag import swag, swag_utils

import warnings
warnings.filterwarnings('ignore')

# wandb
wandb.init(project="SA-BTL")

parser = argparse.ArgumentParser(description="SGD/SAM/BSAM training")

parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")

parser.add_argument("--method", type=str, default="swag",
                    choices=["swag", "redux"],
                    help="Method to make prior")

parser.add_argument("--print_epoch", type=int, default=10, help="Printing epoch")

## Data ---------------------------------------------------------
parser.add_argument(
    "--dataset", type=str, default="mnist-source", choices=["mnist-source", "imagenet"],
                    help="dataset name")

parser.add_argument(
    "--data_path",
    type=str,
    default='./',
    required=True,
    help="path to datasets location (default: None)",)

parser.add_argument("--batch_size", type=int, default = 64,
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
    choices=['mlp', 'resnet18', 'resnet50'],
    help="model name (default : resnet18)")

parser.add_argument("--save_path",
            type=str, default=None,
            help="Path to save best model dict")
#----------------------------------------------------------------

## Learning Hyperparameter --------------------------------------
parser.add_argument("--lr_init", type=float, default=0.01,
                help="learning rate (Default : 0.01)")

parser.add_argument("--momentum", type=float, default=0.9,
                help="momentum (Default : 0.9)")

parser.add_argument("--nesterov", action='store_true',  help="Nesterov (Default : False)")

parser.add_argument("--epochs", type=int, default=400, metavar="N",
    help="number epochs to train (default : 400)")

parser.add_argument("--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)")

parser.add_argument("--scheduler", type=str, default=None, choices=[None, "step_lr", "cos_anneal", "cyclic_lr"])

parser.add_argument("--t_max", type=int, default=200, help="T_max for Cosine Annealing Learning Rate Scheduler")
#----------------------------------------------------------------

## SWAG ---------------------------------------------------------
parser.add_argument("--swa_start", type=int, default=161, help="Start epoch of SWAG")
parser.add_argument("--swa_lr", type=float, default=0.05, help="Learning rate for SWAG")
parser.add_argument("--diag_only", action="store_true", help="Calculate only diagonal covariance")
parser.add_argument("--swa_c_epochs", type=int, default=1, help="Cycle to calculate SWAG statistics")
parser.add_argument("--max_num_models", type=int, default=20, help="Number of models to get SWAG statistics")
parser.add_argument("--swa_c_batches", type=int, default=None, help="???")
#----------------------------------------------------------------


## bma or metrics -----------------------------------------------
parser.add_argument("--eps", type=float, default=1e-8, help="small float to calculate nll")

parser.add_argument("--bma_num_models", type=int, default=50, help="Number of models for bma")

parser.add_argument("--num_bins", type=int, default=50, help="bin number for ece")

parser.add_argument("--metrics_step", action='store_true',
                    help="Calculate loss, accuracy, nll, ece, and auroc for every evaluation step")
#----------------------------------------------------------------

## Transfer Learning --------------------------------------------
parser.add_argument("--load", type=str, default=None,
    help="path to load saved model for transfer learning (default: None)",)
#----------------------------------------------------------------

args = parser.parse_args()


# Set Device and Seed--------------------------------------------
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {args.device}")

utils.set_seed(args.seed)
#----------------------------------------------------------------


# wandb config---------------------------------------------------
wandb.config.update(args)
if args.load is not None:
    wandb.run.name = f"{args.dataset}_run_prior"
else:
    raise "You need to give load path as argument"
    
#----------------------------------------------------------------

# Load Data ------------------------------------------------------
if args.dataset == 'mnist-source':
    tr_loader, val_loader, te_loader, num_classes = data.get_mnist_source(args.batch_size,
                                                                        args.num_workers,
                                                                        use_validation = args.use_validation,
                                                                        seed = args.seed)
elif args.dataset == "imagenet":
    raise "Add Code for ImageNet"


if not args.use_validation:
    val_loader = te_loader

print(f"Load Data : {args.dataset}")
#----------------------------------------------------------------

# Load Model------------------------------------------------------
if args.model == "mlp":
    model = mlp.MLP(output_size=num_classes).to(args.device)
elif args.model == "resnet18":
    from torchvision.models import resnet18
    model = resnet18(pretrained=False, num_classes=num_classes).to(args.device)
elif args.model == "resnet50":
    from torchvision.models import resnet18
    model = resnet18(pretrained=True, num_classes=num_classes).to(args.device)
print(f"Preparing model {args.model}")



swag_model = swag.SWAG(copy.deepcopy(model), no_cov_mat=args.diag_only, max_num_models=args.max_num_models).to(args.device) 
print("Preparing SWAG model")
#-------------------------------------------------------------------

# Set Criterion------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()
#-------------------------------------------------------------------


# Set Optimizer & Scheduler--------------------------------------
## Optimizer
optimizer = torch.optim.SGD(model.parameters(),
                    lr=args.lr_init, weight_decay=args.wd,
                    momentum=args.momentum, nesterov=args.nesterov)
#-------------------------------------------------------------------

    
## Scheduler-------------------------------------------------------
if args.scheduler == "step_lr":
    from utils import StepLR
    scheduler = StepLR(optimizer, args.lr_init, args.epochs)
elif args.scheduler == "cos_anneal":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)    
# Cyclic lr은 SWAG 내부에서
#-------------------------------------------------------------------



## Load Pre-Trained Model ----------------------------------------------------------
if args.load is not None:
    print(f"Load pre-trained model from {args.load}")
    checkpoint = torch.load(args.load)
    model.load_state_dict(checkpoint["state_dict"])


## Training -------------------------------------------------------------------------
print("Start Run Prior")

## print setting
if args.method == "swag":
    columns = ["epoch", "method", "lr", "tr_loss", "tr_acc", "val_loss", "val_acc", "val_nll", "val_ece", "swa_val_loss", "swa_val_acc", "swa_nll", "swa_ece", "time"]

    swag_res = {"loss": None, "accuracy": None, "nll" : None, "ece" : None}
    if args.swa_c_epochs is not None and args.swa_c_batches is not None:
        raise RuntimeError("One of swa_c_epochs or swa_c_batches must be None!")

elif args.method == "redux":
    raise "Add code for Redux"


best_val_loss=9999 ; best_val_acc=0 ; best_epoch=0 ; cnt=0
for epoch in range(int(args.epochs)):
    time_ep = time.time()

    # lr scheduling
    if args.scheduler == "cyclic_lr":
        if args.method == "swag":
            lr = swag_utils.schedule(epoch, args.lr_init, args.epochs, True, args.swa_start, args.swa_lr)
            swag_utils.adjust_learning_rate(optimizer, lr)
        else:
            lr = swag_utils.schedule(epoch, args.lr_init, args.epochs, False, None, None)
    else:
        lr = optimizer.param_groups[0]['lr']

    # train
    tr_res = utils.train_sgd(tr_loader, model, criterion, optimizer, args.device)

    # eval
    if args.metrics_step:
        val_res = utils.eval_metrics(val_loader, model, criterion, args.device, args.num_bins, args.eps)
    else:
        val_res = utils.eval(val_loader, model, criterion, args.device)


    if (args.method=="swag") and ((epoch + 1) > args.swa_start) and ((epoch + 1 - args.swa_start) % args.swa_c_epochs == 0):
        swag_model.collect_model(model)
        swag_model.sample(0.0)
        swag_utils.bn_update(tr_loader, swag_model)
        if args.metrics_step:
            swag_res = utils.eval_metrics(val_loader, swag_model, criterion, args.device, args.num_bins, args.eps)
        else:
            swag_res = utils.eval(val_loader, swag_model, criterion, args.device)

    time_ep = time.time() - time_ep

    if args.method == "swag":
        values = [epoch + 1, f"{args.method}", lr, tr_res["loss"], tr_res["accuracy"],
            val_res["loss"], val_res["accuracy"], val_res["nll"], val_res["ece"],
            swag_res["loss"], swag_res["accuracy"], swag_res["nll"], swag_res["ece"],
                time_ep]
    else:
        values = [epoch + 1, f"{args.method}", lr, tr_res["loss"], tr_res["accuracy"],
            val_res["loss"], val_res["accuracy"], val_res["nll"], val_res["ece"],
            time_ep]
    
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % args.print_epoch == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)


    ## wandb
    if args.method == "swag":
        wandb.log({"Train Loss ": tr_res["loss"], "Train Accuracy" : tr_res["accuracy"],
            "Validation loss" : val_res["loss"], "Validation Accuracy" : val_res["accuracy"],
            "SWAG Validation loss" : swag_res["loss"], "SWAG Validation Accuracy" : swag_res["accuracy"],
            "SWAG Validation nll" : swag_res["nll"], "SWAG Validation ece" : swag_res["ece"],
            "lr" : lr,})
    elif args.method == "redux":
        raise "Add Code for Columns of table"

    if (args.method == "swag") and ((epoch + 1) > args.swa_start) and ((epoch + 1 - args.swa_start) % args.swa_c_epochs == 0):
        # if swag_res['accuracy'] > best_val_acc:
        if swag_res['loss'] < best_val_loss:
            cnt = 1
            best_val_loss = swag_res["loss"]
            best_val_acc = swag_res['accuracy']
            best_epoch = epoch + 1

            # save state_dict
            os.makedirs(args.save_path, exist_ok=True)
            torch.save(model.state_dict(), f"{args.save_path}/{args.method}_best_val_model.pt")
            utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}_best_val.pt",
                                epoch = epoch,
                                state_dict = swag_model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                )

            # Save Mean, variance, Covariance matrix
            mean, variance, cov_mat_sqrt = swag_model.generate_mean_var_covar()
            
            mean = swag_utils.flatten(mean)             # flatten
            variance = swag_utils.flatten(variance)     # flatten
            cov_mat = torch.cat([layer for layer in cov_mat_sqrt], dim=1)   # [max_num_model, num_of_params]
                
            torch.save(mean,f'{args.save_path}/{args.method}_best_val_mean.pt')
            torch.save(variance, f'{args.save_path}/{args.method}_best_val_variance.pt')
            torch.save(cov_mat, f'{args.save_path}/{args.method}_best_val_covmat.pt')
        else:
            cnt = cnt + 1
    
    else:
        # Early Stopping & Save Best Model
        if val_res["loss"] < best_val_loss:
        # if te_res["accuracy"] > best_val_acc:
            cnt = 1
            best_val_loss = val_res["loss"]
            best_val_acc = val_res["accuracy"]
            best_epoch = epoch + 1

            # save state_dict
            os.makedirs(args.save_path,exist_ok=True)
            utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}_best_val.pt",
                                epoch = epoch,
                                state_dict = model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                )

    if args.scheduler in ["cos_anneal", "step_lr"]:
        scheduler.step()



##### Get accuracy, nll, ece on best model
# Load Best Model
print("Load Best Validation Model (Lowest Loss)")
state_dict_path = f'{args.save_path}/{args.method}_best_val.pt'
checkpoint = torch.load(state_dict_path)
if args.method == "swag":
    swag_model.load_state_dict(checkpoint["state_dict"])
    swag_model.to(args.device)
else:
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device)


### bma prediction
if args.method == "swag":
    bma_res = utils.bma(tr_loader, te_loader, swag_model, args.bma_num_models, num_classes, eps=args.eps)
    bma_predictions = bma_res["predictions"]
    bma_targets = bma_res["targets"]

    # Acc
    bma_accuracy = bma_res["bma_accuracy"] * 100
    wandb.run.summary['bma accuracy'] = bma_accuracy
    print(f"bma accuracy : {bma_accuracy:8.4f}")
    
    # nll
    bma_nll = bma_res["nll"]
    wandb.run.summary['bma nll'] = bma_nll
    print(f"bma nll : {bma_nll:8.4f}")

    # ece
    unc = utils.calibration_curve(bma_predictions, bma_targets, args.num_bins)
    bma_ece = unc["ece"]
    wandb.run.summary['bma ece'] = bma_ece
    print(f"bma ece : {bma_ece:8.4f}")


if args.method == "swag":
    # MAP solution
    res = swag_utils.predict(te_loader, swag_model)
else:
    res = swag_utils.predict(te_loader, model)

predictions = res["predictions"]
targets = res["targets"]

# Acc
te_accuracy = np.mean(np.argmax(predictions, axis=1) == targets) * 100
wandb.run.summary['Test accuracy'] = te_accuracy
wandb.run.summary['Best epoch'] = checkpoint["epoch"] + 1
print(f"Best Test Accuracy : {te_accuracy:8.4f}% on epoch {best_epoch}")

# nll
te_nll = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), targets] + args.eps))
wandb.run.summary['Test nll'] = te_nll
print(f"Test nll: {te_nll:8.4f}")

# ece
unc = utils.calibration_curve(predictions, targets, args.num_bins)
te_ece = unc["ece"]
wandb.run.summary["Test ece"]  = te_ece
print(f"Test ece : {te_ece:8.4f}")