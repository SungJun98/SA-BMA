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

import utils.utils as utils
from utils.swag import swag, swag_utils

import warnings
warnings.filterwarnings('ignore')

# wandb
wandb.init(project="SA-BTL", entity='sungjun98')

parser = argparse.ArgumentParser(description="training baselines")

parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")

parser.add_argument("--method", type=str, default="dnn",
                    choices=["dnn", "swag", "last_swag", "vi", "la"],
                    help="Learning Method")

parser.add_argument("--no_amp", action="store_true", default=False, help="Deactivate AMP")

parser.add_argument("--print_epoch", type=int, default=10, help="Printing epoch")

parser.add_argument("--resume", type=str, default=None,
    help="path to load saved model to resume training (default: None)",)

parser.add_argument("--linear_probe", action="store_true", default=False,
        help = "When we do Linear Probing (Default : False)")

parser.add_argument("--tol", type=int, default=30,
        help="tolerance for early stopping (Default : 30)")

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
            help ="Use validation for hyperparameter search (Default : False)")

parser.add_argument("--dat_per_cls", type=int, default=-1,
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
    "--pre_trained", action='store_true', default=False,
    help="Using pre-trained model from zoo"
    )

parser.add_argument("--save_path",
            type=str, default="/data2/lsj9862/exp_result/",
            help="Path to save best model dict")
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

# Scheduler
parser.add_argument("--scheduler", type=str, default='constant', choices=['constant', "step_lr", "cos_anneal", "swag_lr", "cos_decay"])

parser.add_argument("--lr_min", type=float, default=1e-8,
                help="Min learning rate. (Cosine Annealing Warmup Restarts)")

parser.add_argument("--warmup_t", type=int, default=10,
                help="Linear warmup step size. (Cosine Annealing Warmup Restarts)")

parser.add_argument("--warmup_lr_init", type=float, default=1e-7,
                help="Linear warmup initial learning rate. (Cosine Annealing Warmup Restarts)")
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

## VI ---------------------------------------------------------
parser.add_argument("--vi_prior_mu", type=float, default=0.0,
                help="Set prior mean for variational ineference (Default: 0.0)")
parser.add_argument("--vi_prior_sigma", type=float, default=1.0,
                help="Set prior variance for variational inference (Default: 1.0)")
parser.add_argument("--vi_posterior_mu_init", type=float, default=0.0,
                help="Set posterior mean initialization for variatoinal inference (Default: 0.0)")
parser.add_argument("--vi_posterior_rho_init", type=float, default=-3.0,
                help="Set perturbation on posterior mean for variational inference (Default: -3.0)")
parser.add_argument("--vi_type", type=str, default="Reparameterization", choices=["Reparameterization", "Flipout"],
                help="Set type of variational inference (Default: Reparameterization)")
parser.add_argument("--vi_moped_delta", type=float, default=0.5,
                help="Set initial perturbation factor for weight in MOPED framework (Default: 0.5)")
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

# Set BMA and Save Setting-----------------------------------------
if args.method == 'dnn':
    args.bma_num_models = 1

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
if args.linear_probe or args.method == "last_swag":
    utils.freeze_fe(model)

swag_model=None
if args.method == "swag":
    swag_model = swag.SWAG(copy.deepcopy(model),
                        no_cov_mat=args.diag_only,
                        max_num_models=args.max_num_models,
                        last_layer=False).to(args.device)
    print("Preparing SWAG model")
elif args.method == "last_swag":
    swag_model = swag.SWAG(copy.deepcopy(model),
                        no_cov_mat=args.diag_only,
                        max_num_models=args.max_num_models,
                        last_layer=True).to(args.device)
    print("Preparing Last-SWAG model")
elif args.method == "vi":
    from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
    const_bnn_prior_parameters = {
        "prior_mu": args.vi_prior_mu,
        "prior_sigma": args.vi_prior_sigma,
        "posterior_mu_init": args.vi_posterior_mu_init,
        "posterior_rho_init": args.vi_posterior_rho_init,
        "type": args.vi_type,
        "moped_enable": True,
        "moped_delta": args.vi_moped_delta,
    }
    dnn_to_bnn(model, const_bnn_prior_parameters)
    model.to(args.device)
    print(f"Preparing Model for {args.vi_type} VI with MOPED ")
elif args.method == "la":
    raise ValueError("Add code for Laplace Approximation")

print("-"*30)
#-------------------------------------------------------------------

# Set Criterion-----------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()
print("Set Criterion as Cross Entropy")
print("-"*30)
#-------------------------------------------------------------------

# Set Optimizer--------------------------------------
optimizer = utils.get_optimizer(args, model)
print(f"Set {args.optim} optimizer with lr_init {args.lr_init} / wd {args.wd} / momentum {args.momentum}")
print("-"*30)
#----------------------------------------------------------------
    
## Set Scheduler----------------------------------------------------
if args.scheduler not in ["constant", "swag_lr"]:
    scheduler = utils.get_scheduler(args, optimizer)
print(f"Set {args.scheduler} lr scheduler")
print("-"*30)
#-------------------------------------------------------------------


## Resume ---------------------------------------------------------------------------
start_epoch = 1

if not args.pre_trained and args.resume is not None:
    print(f"Resume training from {args.resume}")
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint["state_dict"])
    if args.method == 'last_swag':
        swag_model.base.load_state_dict(checkpoint["state_dict"], strict=False)
    utils.freeze_fe(model)
        
else:
    if args.resume is not None:
        print(f"Resume training from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["state_dict"])

    if args.method == "swag" and args.swag_resume is not None:
        print(f"Resume swag training from {args.swag_resume}")
        checkpoint = torch.load(args.swag_resume)
        start_epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        # if args.scheduler != "swag_lr":
        #     scheduler = scheduler.state_dict()
        swag_model = swag.SWAG(copy.deepcopy(model), no_cov_mat=args.diag_only, max_num_models=args.max_num_models).to(args.device) 
        swag_model.load_state_dict(checkpoint["state_dict"])
#------------------------------------------------------------------------------------


## Set AMP --------------------------------------------------------------------------
scaler, first_step_scaler, second_step_scaler = utils.get_scaler(args)
if args.resume:
    raise ValueError("Add code to load scalar from resume")
print("-"*30)
#------------------------------------------------------------------------------------


## Training -------------------------------------------------------------------------
print(f"Start training {args.method} with {args.optim} optimizer from {start_epoch} epoch!")

## print setting
columns = ["epoch", "method", "lr", "tr_loss", "tr_acc", "val_loss", "val_acc", "val_nll", "val_ece", "time"]
if args.method in ["swag", "last_swag"]:
    columns = columns[:-1] + ["swag_val_loss", "swag_val_acc", "swag_val_nll", "swag_val_ece"] + columns[-1:]
    swag_res = {"loss": None, "accuracy": None, "nll" : None, "ece" : None}

    if args.swa_c_epochs is None:
        raise RuntimeError("swa_c_epochs must not be None!")
       
    print(f"Running SWAG...")


best_val_loss=9999 ; best_val_acc=0 ; best_epoch=0 ; cnt=0; swag_cnt = 0; best_swag_val_loss=9999
for epoch in range(start_epoch, int(args.epochs)+1):
    time_ep = time.time()

    ## lr scheduling
    if args.scheduler == "swag_lr":
        if args.method in ["swag", "last_swag"]:
            lr = swag_utils.schedule(epoch, args.lr_init, args.epochs, True, args.swa_start, args.swa_lr)
        else:
            lr = swag_utils.schedule(epoch, args.lr_init, args.epochs, False, None, None)
        swag_utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = optimizer.param_groups[0]['lr']
    
    ## train
    if args.method in ["vi"]:
        tr_res = utils.train_vi(tr_loader, model, criterion, optimizer, args.device, scaler, args.batch_size)
    else:
        if args.optim in ["sgd", "adam"]:
            tr_res = utils.train_sgd(tr_loader, model, criterion, optimizer, args.device, scaler)
        elif args.optim == "sam":
            tr_res = utils.train_sam(tr_loader, model, criterion, optimizer, args.device, first_step_scaler, second_step_scaler)

    ## valid
    if args.method in ["vi"]:
        val_res = utils.eval_vi(val_loader, model, num_classes, criterion, args.val_mc_num, args.num_bins, args.eps)
    else:
        val_res = utils.eval(val_loader, model, criterion, args.device, args.num_bins, args.eps)

    ## swag valid
    if (args.method in ["swag", "last_swag"]) and ((epoch + 1) > args.swa_start) and ((epoch + 1 - args.swa_start) % args.swa_c_epochs == 0):

        swag_model.collect_model(model)
        swag_model.sample(0.0)
        
        if args.batch_norm == True:
            swag_utils.bn_update(tr_loader, swag_model)
            
        swag_res = utils.eval(val_loader, swag_model, criterion, args.device, args.num_bins, args.eps)

    time_ep = time.time() - time_ep

    ## print result
    if args.method in ["swag", "last_swag"]:
        values = [epoch, f"{args.method}-{args.optim}", lr, tr_res["loss"], tr_res["accuracy"],
            val_res["loss"], val_res["accuracy"], val_res["nll"], val_res["ece"],
            swag_res["loss"], swag_res["accuracy"], swag_res["nll"], swag_res["ece"],
                time_ep]
    else:
        values = [epoch, f"{args.method}-{args.optim}", lr, tr_res["loss"], tr_res["accuracy"],
            val_res["loss"], val_res["accuracy"], val_res["nll"], val_res["ece"],
            time_ep]
    
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % args.print_epoch == 1:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)


    ## wandb
    if args.method in ["swag", "last_swag"]:
        wandb.log({"Train Loss ": tr_res["loss"], "Train Accuracy" : tr_res["accuracy"],
            "Validation loss" : val_res["loss"], "Validation Accuracy" : val_res["accuracy"],
            "SWAG Validation loss" : swag_res["loss"], "SWAG Validation Accuracy" : swag_res["accuracy"],
            "SWAG Validation nll" : swag_res["nll"], "SWAG Validation ece" : swag_res["ece"],
            "lr" : lr,})
    else:
        wandb.log({"Train Loss ": tr_res["loss"], "Train Accuracy" : tr_res["accuracy"],
            "Validation loss" : val_res["loss"], "Validation Accuracy" : val_res["accuracy"],
            "Validation nll" : val_res["nll"], "Validation ece" : val_res["ece"],
            "lr" : lr,})


    ## Save best model (Early Stopping)
    if (args.method in ["swag", "last_swag"]) and (swag_res['loss'] is not None):
        if swag_res['loss'] < best_swag_val_loss:
            swag_cnt = 0
            best_val_loss = val_res["loss"]
            best_val_acc = val_res['accuracy']
            best_swag_val_loss = swag_res["loss"]
            best_swag_val_acc = swag_res['accuracy']
            best_epoch = epoch

            # save state_dict
            os.makedirs(args.save_path, exist_ok=True)
            utils.save_best_swag_model(args, best_epoch, model, swag_model, optimizer, scaler, first_step_scaler, second_step_scaler)
        else:
            swag_cnt += 1

    else:
        if val_res["loss"] < best_val_loss:
            cnt = 0
            best_val_loss = val_res["loss"]
            best_val_acc = val_res["accuracy"]
            best_epoch = epoch

            # save state_dict
            os.makedirs(args.save_path, exist_ok=True)
            if args.method == "vi":
                utils.save_best_vi_model(args, best_epoch, model, optimizer, scaler, first_step_scaler, second_step_scaler)
            elif args.method == "dnn":
                utils.save_best_dnn_model(args, best_epoch, model, optimizer, scaler, first_step_scaler, second_step_scaler)
        else:
            cnt +=1
    
    ## Early Stopping
    if cnt == args.tol and args.method in ['dnn', "vi"]:
        break
    elif swag_cnt == args.tol and args.method in ['swag', 'last_swag']:
        break

    if args.scheduler in ["cos_decay", "step_lr"]:
        scheduler.step(epoch)
#------------------------------------------------------------------------------------------------------------






## Test ------------------------------------------------------------------------------------------------------
"""
이 부분도 bma.py로 따로 파놓자
"""
##### Get test nll, Entropy, ece, Reliability Diagram on best model
# Load Best Model
print("Load Best Validation Model (Lowest Loss)")
state_dict_path = f'{args.save_path}/{args.method}-{args.optim}_best_val.pt'
checkpoint = torch.load(state_dict_path)
if args.method in ["swag", "last_swag"]:
    swag_model.load_state_dict(checkpoint["state_dict"])
    swag_model.to(args.device)
else:
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device)


### BMA prediction
if args.method in ["swag", "last_swag", "vi"]:
    bma_save_path = f"{args.save_path}/bma_models"
    os.makedirs(bma_save_path, exist_ok=True)
    
    if args.method in ["swag", "last_swag"]:
        model = swag_model
    
    bma_res = utils.bma(tr_loader, te_loader, args.method, model, args.bma_num_models, num_classes, bma_save_path=bma_save_path, eps=args.eps, batch_norm=args.batch_norm)
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

    # Save Reliability Diagram 
    utils.save_reliability_diagram(args.method, args.optim, args.save_path, unc, True)



if args.method in ["swag", "last_swag"]:
    sample = swag_model.sample(0)
    if args.batch_norm:
        swag_utils.bn_update(tr_loader, swag_model, verbose=False, subset=1.0)
    res = utils.eval(te_loader, swag_model, criterion, args.device)
else:
    res = utils.eval(te_loader, model, criterion, args.device)

wandb.run.summary['Best epoch'] = checkpoint["epoch"]
# Acc
wandb.run.summary['test accuracy'] = res['accuracy']
print(f"Best test accuracy : {res['accuracy']:8.4f}% on epoch {checkpoint['epoch']}")

# nll
wandb.run.summary['test nll'] = res['nll']
print(f"test nll: {res['nll']:8.4f}")

# ece
wandb.run.summary["test ece"]  = res['ece']
print(f"test ece : {res['ece']:8.4f}")

# Save Reliability Diagram 
utils.save_reliability_diagram(args.method, args.optim, args.save_path, res['unc'], False)