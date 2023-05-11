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

import utils, data

from baselines.sam.sam import SAM
from baselines.swag import swag, swag_utils

import sabtl

import warnings
warnings.filterwarnings('ignore')

# wandb
wandb.init(project="SA-BTL", entity='sungjun98')

parser = argparse.ArgumentParser(description="Run SABTL")

parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")

parser.add_argument("--method", type=str, default="sabtl",
                    choices=["sabtl"],
                    help="Learning Method")

parser.add_argument("--print_epoch", type=int, default=10, help="Printing epoch")

parser.add_argument("--resume", type=str, default=None,
    help="path to load saved model to resume training (default: None)",)

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

parser.add_argument(
    "--pre_trained", action='store_true', default=False,
    help="Using pre-trained model from zoo"
    )

parser.add_argument("--save_path",
            type=str, default="/data2/lsj9862/exp_result/",
            help="Path to save best model dict")
#----------------------------------------------------------------

## Optimizer Hyperparameter --------------------------------------
parser.add_argument("--optim", type=str, default="bsam",
                    choices=["sgd", "sam", "bsam"],
                    help="Optimization options")

parser.add_argument("--lr_init", type=float, default=0.01,
                help="learning rate (Default : 0.01)")

parser.add_argument("--momentum", type=float, default=0.9,
                help="momentum (Default : 0.9)")

parser.add_argument("--epochs", type=int, default=300, metavar="N",
    help="number epochs to train (default : 300)")

parser.add_argument("--wd", type=float, default=5e-4, help="weight decay (default: 5e-4)")

parser.add_argument("--rho", type=float, default=0.05, help="size of pertubation ball for SAM / BSAM")

parser.add_argument("--scheduler", type=str, default='constant', choices=['constant', "step_lr", "cos_anneal", "swag_lr"])

parser.add_argument("--t_max", type=int, default=300, help="T_max for Cosine Annealing Learning Rate Scheduler")
#----------------------------------------------------------------


## SABTL ---------------------------------------------------------
parser.add_argument("--swa_lr", type=float, default=0.05, help="Learning rate for SWAG")
parser.add_argument("--src_bnn", type=str, default="swag", choices=["swag", "la", "vi"],
        help="Type of pre-trained BNN model")
# parser.add_argument("--z_scale", type=float, default=1e-2, help="Sampling scale for weight")
parser.add_argument("--diag_only", action="store_true", default=False, help="Consider only diagonal variance")
parser.add_argument("--low_rank", type=int, default=20, help="Low-rank component")

parser.add_argument("--mean_path", type=str, default="/home/lsj9862/BayesianSAM/exp_result/resnet18-noBN/swag-sgd_best_val_mean.pt",
    help="path to load saved mean of swag model for transfer learning (default: None)")
parser.add_argument("--var_path", type=str, default="/home/lsj9862/BayesianSAM/exp_result/resnet18-noBN/swag-sgd_best_val_variance.pt",
    help="path to load saved variance of swag model for transfer learning (default: None)")
parser.add_argument("--covmat_path", type=str, default="/home/lsj9862/BayesianSAM/exp_result/resnet18-noBN/swag-sgd_best_val_covmat.pt",
    help="path to load saved covariance matrix of swag model for transfer learning (default: None)")
#----------------------------------------------------------------

## bma or metrics -----------------------------------------------
parser.add_argument("--eps", type=float, default=1e-8, help="small float to calculate nll")
parser.add_argument("--bma_num_models", type=int, default=30, help="Number of models for bma")
parser.add_argument("--num_bins", type=int, default=50, help="bin number for ece")
#----------------------------------------------------------------

args = parser.parse_args()
#----------------------------------------------------------------

# Set Device and Seed--------------------------------------------
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {args.device}")

args.last_layer = True

if args.model.split("-")[-1] == "noBN":
    args.batch_norm = False
else:
    args.batch_norm = True

utils.set_seed(args.seed)
#----------------------------------------------------------------

# Set BMA and Save Setting--------------------------------------------
args.save_path = utils.set_save_path(args)
print(f"Save Results on {args.save_path}")
#---------------------------------------------------------------

# wandb config---------------------------------------------------
wandb.config.update(args)
wandb.run.name = utils.set_wandb_runname(args)
#----------------------------------------------------------------

# Load Data ------------------------------------------------------
tr_loader, val_loader, te_loader, num_classes = utils.get_dataset(args.dataset,
                                                            args.data_path,
                                                            args.batch_size,
                                                            args.num_workers,
                                                            args.use_validation)
print(f"Load Data : {args.dataset}")
#----------------------------------------------------------------

# Define Model------------------------------------------------------
model = utils.get_backbone(args.model, num_classes, args.device, args.pre_trained)

w_mean = torch.load(args.mean_path)
w_var = torch.load(args.var_path) 
w_covmat = torch.load(args.covmat_path)
sabtl_model = sabtl.SABTL(copy.deepcopy(model),
                        src_bnn=args.src_bnn,
                        w_mean = w_mean,
                        diag_only=args.diag_only,
                        w_var=w_var,
                        low_rank=args.low_rank,
                        w_cov_sqrt=w_covmat,
                        ).to(args.device)
#----------------------------------------------------------------

# Set Criterion------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()
#-------------------------------------------------------------------

# Set Optimizer--------------------------------------
## Optimizer
if args.optim == "sgd":
    optimizer = torch.optim.SGD(sabtl_model.bnn_param.values(),
                        lr=args.lr_init, weight_decay=args.wd,
                        momentum=args.momentum)
    
elif args.optim == "sam":
    base_optimizer = torch.optim.SGD
    optimizer = SAM(sabtl_model.bnn_param.values(), base_optimizer, rho=args.rho, lr=args.lr_init, momentum=args.momentum,
                    weight_decay=args.wd)
    
elif args.optim == "bsam":
    base_optimizer = torch.optim.SGD
    optimizer = sabtl.BSAM(sabtl_model.bnn_param.values(), base_optimizer, rho=args.rho, lr=args.lr_init, momentum=args.momentum,
                    weight_decay=args.wd)
#----------------------------------------------------------------

## Set Scheduler-------------------------------------------------------
if args.scheduler == "step_lr":
    from utils import StepLR
    scheduler = StepLR(optimizer, args.lr_init, args.epochs)
elif args.scheduler == "cos_anneal":
    if args.optim == "sgd":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)    
    elif args.optim in ["sam", "bsam"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=args.t_max)
#-------------------------------------------------------------------

## Resume ---------------------------------------------------------------------------
"""
나중에 필요하면 채우기
"""
start_epoch = 0
#------------------------------------------------------------------------------------

## Set AMP --------------------------------------------------------------------------
if args.optim == "sgd":
    scaler = torch.cuda.amp.GradScaler()

elif args.optim in ["sam", "bsam"]:
    first_step_scaler = torch.cuda.amp.GradScaler(2 ** 8)
    second_step_scaler = torch.cuda.amp.GradScaler(2 ** 8)

print(f"Set AMP Scaler for {args.optim}")
#------------------------------------------------------------------------------------

## Training -------------------------------------------------------------------------
print(f"Start training SABTL with {args.optim} optimizer from {start_epoch} epoch!")

## print setting
columns = ["epoch", "method", "lr",
        "tr_loss", "tr_acc",
        "val_loss(MAP)", "val_acc(MAP)", "val_nll(MAP)", "val_ece(MAP)",
        "time"]

best_val_loss=9999 ; best_val_acc=0 ; best_epoch=0 ; cnt=0
print("Start Training!!")
for epoch in range(start_epoch, int(args.epochs)):
    time_ep = time.time()

    ## lr scheduling
    if args.scheduler == "swag_lr":
        lr = swag_utils.schedule(epoch, args.lr_init, args.epochs, False, None, None)
        swag_utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = optimizer.param_groups[0]['lr']
        
    ## train
    if args.optim == "sgd":
        tr_res = utils.train_sabtl_sgd(tr_loader, sabtl_model, criterion, optimizer, args.device, scaler)
    elif args.optim == "sam":
        tr_res = utils.train_sabtl_sam(tr_loader, sabtl_model, criterion, optimizer, args.device, first_step_scaler, second_step_scaler)
    elif args.optim == "bsam":
        tr_res = utils.train_sabtl_bsam(tr_loader, sabtl_model, criterion, optimizer, args.device, first_step_scaler, second_step_scaler)
        
    # validation / test
    params, _, _ = sabtl_model.sample(0.0)
    params = utils.format_weights(params, sabtl_model)
    val_res = utils.eval_sabtl(val_loader, sabtl_model, params, criterion, args.device, args.num_bins, args.eps)

    time_ep = time.time() - time_ep

    values = [epoch + 1, f"sabtl-{args.optim}", lr, tr_res["loss"], tr_res["accuracy"],
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
    wandb.log({"Train Loss ": tr_res["loss"], "Train Accuracy" : tr_res["accuracy"],
        "Validation loss (MAP)" : val_res["loss"], "Validation Accuracy (MAP)" : val_res["accuracy"],
        "Validation nll (MAP)" : val_res["nll"], "Validation ece (MAP)" : val_res["ece"],
        "lr" : lr,
        "max(mean)" : torch.max(sabtl_model.bnn_param['mean']),
        "min(mean)" : torch.min(sabtl_model.bnn_param['mean']),
        "max(std)" : torch.max(torch.exp(sabtl_model.bnn_param['log_std'])),
        "min(std)" : torch.min(torch.exp(sabtl_model.bnn_param['log_std'])),
        })
    if not args.diag_only:
        wandb.log({"max(cov_sqrt)" : torch.max(sabtl_model.bnn_param['cov_sqrt']),
            "min(cov_sqrt)" : torch.min(sabtl_model.bnn_param['cov_sqrt']),})

    # Save best model (Early Stopping)
    if val_res['loss'] < best_val_loss: #### 지금 loss scale이...
    # if (val_res['accuracy'] > best_val_acc) or (val_res['loss'] < best_val_loss): #### 지금 loss scale이...
        best_val_loss = val_res['loss']
        best_val_acc = val_res['accuracy']
        best_epoch = epoch + 1

        # save state_dict
        os.makedirs(args.save_path, exist_ok=True)
        if args.optim == "sgd":
                utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                epoch = epoch,
                                state_dict =sabtl_model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                # scheduler = scheduler.state_dict(),
                                scaler = scaler.state_dict()
                                )
        elif args.optim in ["sam", "bsam"]:
            utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                epoch = epoch,
                                state_dict = sabtl_model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                # scheduler = scheduler.state_dict(),
                                first_step_scaler = first_step_scaler.state_dict(),
                                second_step_scaler = second_step_scaler.state_dict()
                                )
        # Save Mean, variance, Covariance matrix
        # mean, variance, cov_mat_sqrt = swag_model.generate_mean_var_covar()
        mean = sabtl_model.get_mean_vector()
        variance = sabtl_model.get_variance_vector()
        cov_mat_list = sabtl_model.get_covariance_matrix()
            
        torch.save(mean,f'{args.save_path}/{args.method}-{args.optim}_best_val_mean.pt')
        torch.save(variance, f'{args.save_path}/{args.method}-{args.optim}_best_val_variance.pt')
        torch.save(cov_mat_list, f'{args.save_path}/{args.method}-{args.optim}_best_val_covmat.pt')
    
        
    ## Scheduler step
    if args.scheduler in ["cos_anneal", "step_lr"]:
        scheduler.step()
#------------------------------------------------------------------------------------------------------------


## Test ------------------------------------------------------------------------------------------------------
##### Get test nll, Entropy, ece, Reliability Diagram on best model
# Load Best Model
print("Load Best Validation Model (Lowest Loss)")
state_dict_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt"
checkpoint = torch.load(state_dict_path)

sabtl_model.load_state_dict(checkpoint["state_dict"])
sabtl_model.to(args.device)

### BMA prediction
bma_save_path = f"{args.save_path}/bma_models"
os.makedirs(bma_save_path, exist_ok=True)

bma_res = utils.bma_sabtl(te_loader, sabtl_model, args.bma_num_models,
                    num_classes, criterion, args.device,
                    bma_save_path=bma_save_path, eps=args.eps)

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

 # Save ece for reliability diagram
os.makedirs(f'{args.save_path}/unc_result', exist_ok=True)
with open(f"{args.save_path}/unc_result/{args.method}-{args.optim}_bma_uncertainty.pkl", 'wb') as f:
    pickle.dump(unc, f)

# Save Reliability Diagram 
utils.save_reliability_diagram(args.method, args.optim, args.save_path, unc, True)




### MAP Prediction
params, _, _ = sabtl_model.sample(0)
params = utils.format_weights(params, sabtl_model)

res = utils.eval_sabtl(te_loader, sabtl_model, params, criterion, args.device, args.num_bins, args.eps)

wandb.run.summary['Best epoch'] = checkpoint["epoch"] + 1
# Acc
wandb.run.summary['test accuracy'] = res['accuracy']
print(f"Best test accuracy : {res['accuracy']:8.4f}% on epoch {checkpoint['epoch'] + 1}")

# nll
wandb.run.summary['test nll'] = res['nll']
print(f"test nll: {res['nll']:8.4f}")

# ece
unc = utils.calibration_curve(res['predictions'], res['targets'], args.num_bins)
te_ece = unc["ece"]
wandb.run.summary["test ece"]  = te_ece
print(f"test ece : {te_ece:8.4f}")

# Save ece for reliability diagram
os.makedirs(f'{args.save_path}/unc_result', exist_ok=True)
with open(f"{args.save_path}/unc_result/{args.method}-{args.optim}_uncertainty.pkl", 'wb') as f:
    pickle.dump(unc, f)

# Save Reliability Diagram 
utils.save_reliability_diagram(args.method, args.optim, args.save_path, unc, False)