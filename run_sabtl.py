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
from utils.sabtl import sabtl, sabtl_utils

import warnings
warnings.filterwarnings('ignore')

# wandb
wandb.init(project="SA-BTL", entity='sungjun98')

parser = argparse.ArgumentParser(description="Run SABTL")

parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")

parser.add_argument("--method", type=str, default="sabtl",
                    choices=["sabtl"],
                    help="Learning Method")

parser.add_argument("--no_amp", action="store_true", default=False, help="Deactivate AMP")

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

parser.add_argument("--dat_per_cls", type=int, default=-1,
            help="Number of data points per class in few-shot setting. -1 denotes deactivate few-shot setting (Default : -1)")

parser.add_argument("--fe_dat", type=str, default=None, choices=[None, "resnet18-noBN", "vitb16-i21k"],
            help = "Use Feature Extracted from Feature Extractor (Default : None)")
#----------------------------------------------------------------

## Model ---------------------------------------------------------
parser.add_argument(
    "--model",
    type=str, default='resnet18', required=True,
    choices=['resnet18', 'resnet50', 'wideresnet28x10', 'wideresnet40x10',
            'resnet18-noBN', 'resnet50-noBN', 'wideresnet28x10-noBN', 'wideresnet40x10-noBN',
            "vitb16-i21k"],
    help="model name (default : resnet18)")

parser.add_argument(
    "--pre_trained", action='store_true', default=False,
    help="Using pre-trained model from zoo"
    )

parser.add_argument("--save_path",
            type=str, default="/data2/lsj9862/exp_result",
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

parser.add_argument("--eta", type=float, default=1.0, help="Eta to calculate Inverse of Fisher Information Matrix")

parser.add_argument("--scheduler", type=str, default='constant', choices=['constant', "step_lr", "cos_anneal", "swag_lr", "cos_decay"])

parser.add_argument("--lr_min", type=float, default=1e-8,
                help="Min learning rate. (Cosine Annealing Warmup Restarts)")

parser.add_argument("--warmup_t", type=int, default=10,
                help="Linear warmup step size. (Cosine Annealing Warmup Restarts)")

parser.add_argument("--warmup_lr_init", type=float, default=1e-7,
                help="Linear warmup initial learning rate (Cosine Annealing Warmup Restarts)")
#----------------------------------------------------------------


## SABTL ---------------------------------------------------------
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
parser.add_argument("--val_mc_num", type=int, default=5, help="Number of models for Mc integration in validation phase")
parser.add_argument("--eps", type=float, default=1e-8, help="small float to calculate nll")
parser.add_argument("--bma_num_models", type=int, default=30, help="Number of models for bma in test phase")
parser.add_argument("--num_bins", type=int, default=50, help="bin number for ece")
#----------------------------------------------------------------

args = parser.parse_args()
#----------------------------------------------------------------

# Set Device and Seed--------------------------------------------
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {args.device}")

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
                                                            data_path = args.data_path,
                                                            batch_size = args.batch_size,
                                                            num_workers = args.num_workers,
                                                            use_validation = args.use_validation,
                                                            fe_dat = args.fe_dat,
                                                            dat_per_cls = args.dat_per_cls,
                                                            seed = args.seed)
print(f"Load Data : {args.dataset} feature extracted from {args.fe_dat}")
#----------------------------------------------------------------

# Define Model------------------------------------------------------
model = utils.get_backbone(args.model, num_classes, args.device, args.pre_trained, args.fe_dat)   


w_mean = torch.load(args.mean_path)
w_var = torch.load(args.var_path) 
w_covmat = torch.load(args.covmat_path)
sabtl_model = sabtl.SABTL(copy.deepcopy(model),
                        src_bnn=args.src_bnn,
                        pre_trained=args.pre_trained,
                        fe_dat=args.fe_dat,
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
optimizer = sabtl_utils.get_optimizer(args, sabtl_model)
#----------------------------------------------------------------

## Set Scheduler-------------------------------------------------------
if args.scheduler not in ["constant", "swag_lr"]:
    scheduler = utils.get_scheduler(args, optimizer)
#-------------------------------------------------------------------

## Resume ---------------------------------------------------------------------------
"""
나중에 필요하면 채우기
"""
start_epoch = 0
#------------------------------------------------------------------------------------

## Set AMP --------------------------------------------------------------------------
scaler, first_step_scaler, second_step_scaler = utils.get_scaler(args)
#------------------------------------------------------------------------------------

## Training -------------------------------------------------------------------------
print(f"Start training SABTL with {args.optim} optimizer from {start_epoch} epoch!")

## print setting
columns = ["epoch", "method", "lr",
        "tr_loss", "tr_acc",
        "val_loss(MAP)", "val_acc(MAP)", "val_nll(MAP)", "val_ece(MAP)",
        "time"]

best_val_loss=9999 ; best_val_acc=0 ; best_epoch=0 ; duration=0
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
        tr_res = sabtl_utils.train_sabtl_sgd(tr_loader, sabtl_model, criterion, optimizer, args.device, scaler)
    elif args.optim == "sam":
        tr_res = sabtl_utils.train_sabtl_sam(tr_loader, sabtl_model, criterion, optimizer, args.device, first_step_scaler, second_step_scaler)
    elif args.optim == "bsam":
        tr_res = sabtl_utils.train_sabtl_bsam(tr_loader, sabtl_model, criterion, optimizer, args.device, args.eta, first_step_scaler, second_step_scaler)
        
    # validation / test
    if args.val_mc_num ==1:
        params, _, _ = sabtl_model.sample(0.0)
        params = utils.format_weights(params, sabtl_model)
        val_res = sabtl_utils.eval_sabtl(val_loader, sabtl_model, params, criterion, args.device, args.num_bins, args.eps)
    else:
        """
        근데 mc prediction의 acc가 낮고, loss가 높게 나온다..
        """
        val_res = sabtl_utils.bma_sabtl(val_loader, sabtl_model, args.val_mc_num,
                            num_classes, criterion, args.device,
                            bma_save_path=None, eps=1e-8, num_bins=50,
                            validation=True,
                            )
    
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
        "mean(mean)" : torch.mean(sabtl_model.bnn_param['mean']),
        "std(mean)" : torch.std(sabtl_model.bnn_param['mean']),
        "min(mean)" : torch.min(sabtl_model.bnn_param['mean']),
        "max(std)" : torch.max(torch.exp(sabtl_model.bnn_param['log_std'])),
        "mean(std)" : torch.mean(torch.exp(sabtl_model.bnn_param['log_std'])),
        "std(std)" : torch.std(torch.exp(sabtl_model.bnn_param['log_std'])),
        "min(std)" : torch.min(torch.exp(sabtl_model.bnn_param['log_std'])),
        },
        step=epoch)
    if not args.diag_only:
        wandb.log({"max(cov_sqrt)" : torch.max(sabtl_model.bnn_param['cov_sqrt']),
            "mean(cov_sqrt)" : torch.mean(sabtl_model.bnn_param['cov_sqrt']),
            "std(cov_sqrt)" : torch.std(sabtl_model.bnn_param['cov_sqrt']),
            "min(cov_sqrt)" : torch.min(sabtl_model.bnn_param['cov_sqrt']),},
            step=epoch)

    # Save best model (Early Stopping)
    if (val_res['loss'] < best_val_loss):
    # if (val_res['accuracy'] > best_val_acc) or (val_res['loss'] < best_val_loss):
        best_val_loss = val_res['loss']
        best_val_acc = val_res['accuracy']
        best_epoch = epoch + 1

        # save state_dict
        os.makedirs(args.save_path, exist_ok=True)
        if args.optim == "sgd":
            if not args.no_amp:
                utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                epoch = epoch,
                                state_dict =sabtl_model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                # scheduler = scheduler.state_dict(),
                                scaler = scaler.state_dict(),
                                )
            else:
                utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                epoch = epoch,
                                state_dict =sabtl_model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                # scheduler = scheduler.state_dict(),
                                )
        elif args.optim in ["sam", "bsam"]:
            if not args.no_amp:
                utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                    epoch = epoch,
                                    state_dict = sabtl_model.state_dict(),
                                    optimizer = optimizer.state_dict(),
                                    # scheduler = scheduler.state_dict(),
                                    first_step_scaler = first_step_scaler.state_dict(),
                                    second_step_scaler = second_step_scaler.state_dict()
                                    )
            else:
                utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                    epoch = epoch,
                                    state_dict = sabtl_model.state_dict(),
                                    optimizer = optimizer.state_dict(),
                                    # scheduler = scheduler.state_dict(),
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
    # if args.scheduler == "cos_anneal":
    #     scheduler.step()
    if args.scheduler in ["cos_anneal", "step_lr", "cos_decay"]:
        scheduler.step(epoch)
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

bma_res = sabtl_utils.bma_sabtl(te_loader, sabtl_model, args.bma_num_models,
                    num_classes, criterion, args.device,
                    bma_save_path=bma_save_path, eps=args.eps, num_bins=args.num_bins)

bma_predictions = bma_res["predictions"]
bma_targets = bma_res["targets"]

# Acc
bma_accuracy = bma_res["accuracy"]
wandb.run.summary['bma accuracy'] = bma_accuracy
print(f"bma accuracy : {bma_accuracy:8.4f}")

# nll
bma_nll = bma_res["nll"]
wandb.run.summary['bma nll'] = bma_nll
print(f"bma nll : {bma_nll:8.4f}")       

# ece
unc = utils.calibration_curve(bma_predictions, bma_targets, args.num_bins)
# bma_ece = unc["ece"]
bma_ece = bma_res['ece']
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

res = sabtl_utils.eval_sabtl(te_loader, sabtl_model, params, criterion, args.device, args.num_bins, args.eps)

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