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

import utils, data #, losses

from baselines.sam.sam import SAM
from baselines.swag import swag, swag_utils

import warnings
warnings.filterwarnings('ignore')

# wandb
wandb.init(project="SA-BTL", entity='sungjun98')

parser = argparse.ArgumentParser(description="training baselines")

parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")

parser.add_argument("--method", type=str, default="dnn",
                    choices=["dnn", "swag", "last_swag", "sabtl"],
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
parser.add_argument("--optim", type=str, default="sgd",
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

parser.add_argument("--scheduler", type=str, default='constant', choices=['constant', "step_lr", "cos_anneal", "swag_lr", "cos_decay"])

parser.add_argument("--t_max", type=int, default=300, help="T_max (Cosine Annealing)")

# parser.add_argument("--first_cycle_steps", type=int, default=120,
#                 help="First cycle step size. (Cosine Annealing Warmup Restarts)")

# parser.add_argument("--cycle_mult", type=float, default=1.,
#                 help="Cycle steps magnification. (Cosine Annealing Warmup Restarts)")

# parser.add_argument("--min_lr", type=float, default=0.001,
#                 help="Min learning rate. (Cosine Annealing Warmup Restarts)")

# parser.add_argument("--warmup_steps", type=int, default=30,
#                 help="Linear warmup step size. (Cosine Annealing Warmup Restarts)")

# parser.add_argument("--decay_ratio", type=float, default=0.75,
#                 help="Decrease rate of max learning rate by cycle. (Cosine Annealing Warmup Restarts)")

parser.add_argument("--t_initial", type=int, default=100,
                help="First cycle step size. (Cosine Annealing Warmup Restarts)")

parser.add_argument("--lr_min", type=float, default=1e-5,
                help="Min learning rate. (Cosine Annealing Warmup Restarts)")

parser.add_argument("--cycle_mul", type=float, default=0.9,
                help="Decrease rate of max learning rate by cycle. (Cosine Annealing Warmup Restarts)")

parser.add_argument("--cycle_decay", type=float, default=1.,
                help="Decrease rate of cycle length. (Cosine Annealing Warmup Restarts)")

parser.add_argument("--warmup_t", type=int, default=0,
                help="Linear warmup step size. (Cosine Annealing Warmup Restarts)")


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


## bma or metrics -----------------------------------------------
parser.add_argument("--eps", type=float, default=1e-8, help="small float to calculate nll")
parser.add_argument("--bma_num_models", type=int, default=30, help="Number of models for bma")
parser.add_argument("--num_bins", type=int, default=50, help="bin number for ece")
#----------------------------------------------------------------

## Transfer Learning --------------------------------------------
parser.add_argument("--load", type=str, default=None,
    help="path to load saved mosdel for transfer learning (default: None)",)
parser.add_argument("--swag_mean_load", type=str, default=None,
    help="path to load saved mean of swag model for transfer learning (default: None)")
parser.add_argument("--swag_var_load", type=str, default=None,
    help="path to load saved variance of swag model for transfer learning (default: None)")
parser.add_argument("--swag_covmat_load", type=str, default=None,
    help="path to load saved covariance matrix of swag model for transfer learning (default: None)")
#----------------------------------------------------------------


args = parser.parse_args()
#----------------------------------------------------------------

# Set Device and Seed--------------------------------------------
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {args.device}")

if args.method in ["last_swag", "sabtl"]:
    args.last_layer = True 
    # args.swa_start = 0
else:
    args.last_layer = False
    
if args.model.split("-")[-1] == "noBN":
    args.batch_norm = False
else:
    args.batch_norm = True

utils.set_seed(args.seed)
#----------------------------------------------------------------

# Set BMA and Save Setting--------------------------------------------
if args.method == 'dnn':
    args.bma_num_models = 1

args.save_path = utils.set_save_path(args)
print(f"Save Results on {args.save_path}")
#----------------------------------------------------------------

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
swag_model=None
if args.method in ["swag", "last_swag"]:
    swag_model = swag.SWAG(copy.deepcopy(model),
                        no_cov_mat=args.diag_only,
                        max_num_models=args.max_num_models,
                        last_layer=args.last_layer).to(args.device)
    print("Preparing SWAG model")
elif args.method == "sabtl":
    raise RuntimeError("Add code for SABTL")
#-------------------------------------------------------------------

# Set Criterion------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()
#-------------------------------------------------------------------

# Set Optimizer-------------------------------------- (utils로)
if args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(),
                        lr=args.lr_init, weight_decay=args.wd,
                        momentum=args.momentum)
    
elif args.optim == "sam":
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, lr=args.lr_init, momentum=args.momentum,
                    weight_decay=args.wd)
    
elif args.optim == "bsam":
    raise RuntimeError("You have to add BSAM optimizer code!!")
#----------------------------------------------------------------

    
## Set Scheduler------------------------------------------------------- (utils로)
if args.scheduler == "step_lr":
    # from scheduler import StepLR
    # scheduler = StepLR(optimizer, args.lr_init, args.epochs)
    from timm.scheduler.step_lr import StepLRScheduler
    scheduler = StepLRScheduler(optimizer, decay_rate=0.2, )
    
elif args.scheduler == "cos_anneal":
    if args.optim == "sgd":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)    
    elif args.optim in ["sam", "bsam"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=args.t_max)
        
elif args.scheduler == "cos_decay":
    from timm.scheduler.cosine_lr import CosineLRScheduler
    if args.optim == 'sgd':
        scheduler = CosineLRScheduler(optimizer = optimizer,
                                    t_initial= args.t_initial,
                                    lr_min=args.lr_min,
                                    cycle_mul=args.cycle_mul,
                                    cycle_decay=args.cycle_decay,
                                    cycle_limit=args.epochs//args.t_initial,
                                    warmup_t=args.warmup_t,
                                    warmup_lr_init=args.warmup_lr_init,
                                        )
    elif args.optim in ["sam", "bsam"]:
        scheduler = CosineLRScheduler(optimizer = optimizer.base_optimizer,
                                    t_initial= args.t_initial,
                                    lr_min=args.lr_min,
                                    cycle_mul=args.cycle_mul,
                                    cycle_decay=args.cycle_decay,
                                    cycle_limit=args.epochs//args.t_initial,
                                    warmup_t=args.warmup_t,
                                    warmup_lr_init=args.warmup_lr_init,
                                        )
#-------------------------------------------------------------------


## Resume ---------------------------------------------------------------------------
start_epoch = 0

if args.last_layer and not args.pre_trained:
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
if not args.no_amp:
    if args.optim == "sgd":
        scaler = torch.cuda.amp.GradScaler()
        first_step_scaler = None
        second_step_scaler = None

    elif args.optim in ["sam", "bsam"]:
        scaler = None
        first_step_scaler = torch.cuda.amp.GradScaler(2 ** 8)
        second_step_scaler = torch.cuda.amp.GradScaler(2 ** 8)

    print(f"Set AMP Scaler for {args.optim}")

else:
    scaler = None
    first_step_scaler = None
    second_step_scaler = None
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


best_val_loss=9999 ; best_val_acc=0 ; best_epoch=0 ; cnt=0; best_swag_val_loss=9999
print("Start Training!!")
for epoch in range(start_epoch, int(args.epochs)):
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
    if args.optim == "sgd":
        tr_res = utils.train_sgd(tr_loader, model, criterion, optimizer, args.device, scaler)
    elif args.optim == "sam":
        tr_res = utils.train_sam(tr_loader, model, criterion, optimizer, args.device, first_step_scaler, second_step_scaler)
    elif args.optim == "bsam":
        raise RuntimeError("Put utils.train_sabtl in here!!")

    ## valid
    val_res = utils.eval(val_loader, model, criterion, args.device, args.num_bins, args.eps)

    ## swag valid
    if (args.method in ["swag", "last_swag"]) and ((epoch + 1) > args.swa_start) and ((epoch + 1 - args.swa_start) % args.swa_c_epochs == 0):

        swag_model.collect_model(model)
        swag_model.sample(0.0)
        
        if (args.batch_norm == True) and (not args.last_layer):
            swag_utils.bn_update(tr_loader, swag_model)
            
        swag_res = utils.eval(val_loader, swag_model, criterion, args.device, args.num_bins, args.eps)

    time_ep = time.time() - time_ep

    ## print result
    if args.method in ["swag", "last_swag"]:
        values = [epoch + 1, f"{args.method}-{args.optim}", lr, tr_res["loss"], tr_res["accuracy"],
            val_res["loss"], val_res["accuracy"], val_res["nll"], val_res["ece"],
            swag_res["loss"], swag_res["accuracy"], swag_res["nll"], swag_res["ece"],
                time_ep]
    else:
        values = [epoch + 1, f"{args.method}-{args.optim}", lr, tr_res["loss"], tr_res["accuracy"],
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
            best_val_loss = val_res["loss"]
            best_val_acc = val_res['accuracy']
            best_swag_val_loss = swag_res["loss"]
            best_swag_val_acc = swag_res['accuracy']
            best_epoch = epoch + 1

            # save state_dict
            os.makedirs(args.save_path, exist_ok=True)
            if args.optim == "sgd":
                if not args.no_amp:
                    utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                        epoch = epoch,
                                        state_dict = swag_model.state_dict(),
                                        optimizer = optimizer.state_dict(),
                                        # scheduler = scheduler.state_dict(),
                                        scaler = scaler.state_dict()
                                        )
                else:
                    utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                        epoch = epoch,
                                        state_dict = swag_model.state_dict(),
                                        optimizer = optimizer.state_dict(),
                                        # scheduler = scheduler.state_dict(),
                                        )
            elif args.optim in ["sam", "bsam"]:
                if not args.no_amp:
                    utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                        epoch = epoch,
                                        state_dict = swag_model.state_dict(),
                                        optimizer = optimizer.state_dict(),
                                        # scheduler = scheduler.state_dict(),
                                        first_step_scaler = first_step_scaler.state_dict(),
                                        second_step_scaler = second_step_scaler.state_dict()
                                        )
                else:
                    utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                        epoch = epoch,
                                        state_dict = swag_model.state_dict(),
                                        optimizer = optimizer.state_dict(),
                                        # scheduler = scheduler.state_dict(),
                                        )
            torch.save(model.state_dict(),f'{args.save_path}/{args.method}-{args.optim}_best_val_model.pt')
            
            # Save Mean, variance, Covariance matrix
            # mean, variance, cov_mat_sqrt = swag_model.generate_mean_var_covar()
            mean = swag_model.get_mean_vector()
            variance = swag_model.get_variance_vector()
            cov_mat_list = swag_model.get_covariance_matrix()
                
            torch.save(mean,f'{args.save_path}/{args.method}-{args.optim}_best_val_mean.pt')
            torch.save(variance, f'{args.save_path}/{args.method}-{args.optim}_best_val_variance.pt')
            torch.save(cov_mat_list, f'{args.save_path}/{args.method}-{args.optim}_best_val_covmat.pt')
    
    else:
        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            best_val_acc = val_res["accuracy"]
            best_epoch = epoch + 1

            # save state_dict
            os.makedirs(args.save_path,exist_ok=True)
            if args.optim == "sgd":
                if not args.no_amp:
                    utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                    epoch = epoch,
                                    state_dict = model.state_dict(),
                                    optimizer = optimizer.state_dict(),
                                    # scheduler = scheduler.state_dict(),
                                    scaler = scaler.state_dict()
                                    )
                else:
                    utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                    epoch = epoch,
                                    state_dict = model.state_dict(),
                                    optimizer = optimizer.state_dict(),
                                    # scheduler = scheduler.state_dict(),
                                    )
            elif args.optim == "sam":
                if not args.no_amp:
                    utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                    epoch = epoch,
                                    state_dict = model.state_dict(),
                                    optimizer = optimizer.state_dict(),
                                    # scheduler = scheduler.state_dict(),
                                    first_step_scaler = first_step_scaler.state_dict(),
                                    second_step_scaler = second_step_scaler.state_dict()
                                    )
                else:
                    utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                    epoch = epoch,
                                    state_dict = model.state_dict(),
                                    optimizer = optimizer.state_dict(),
                                    # scheduler = scheduler.state_dict(),
                                    )
    
    if args.scheduler in ["cos_anneal", "step_lr"]:
        scheduler.step()
#------------------------------------------------------------------------------------------------------------




## Test ------------------------------------------------------------------------------------------------------
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
if args.method in ["swag", "last_swag"]:
    bma_save_path = f"{args.save_path}/bma_models"
    os.makedirs(bma_save_path, exist_ok=True)
    
    bma_res = utils.bma(tr_loader, te_loader, swag_model, args.bma_num_models, num_classes, bma_save_path=bma_save_path, eps=args.eps, batch_norm=args.batch_norm)
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
if args.method in ["swag", "last_swag"]:
    sample = swag_model.sample(0)
    if args.batch_norm:
        swag_utils.bn_update(tr_loader, swag_model, verbose=False, subset=1.0)
    res = swag_utils.predict(te_loader, swag_model)
else:
    res = swag_utils.predict(te_loader, model)

predictions = res["predictions"]
targets = res["targets"]

wandb.run.summary['Best epoch'] = checkpoint["epoch"] + 1
# Acc
te_accuracy = np.mean(np.argmax(predictions, axis=1) == targets) * 100
wandb.run.summary['test accuracy'] = te_accuracy
print(f"Best test accuracy : {te_accuracy:8.4f}% on epoch {checkpoint['epoch'] + 1}")

# nll
te_nll = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), targets] + args.eps))
wandb.run.summary['test nll'] = te_nll
print(f"test nll: {te_nll:8.4f}")

# ece
unc = utils.calibration_curve(predictions, targets, args.num_bins)
te_ece = unc["ece"]
wandb.run.summary["test ece"]  = te_ece
print(f"test ece : {te_ece:8.4f}")

# Save ece for reliability diagram
os.makedirs(f'{args.save_path}/unc_result', exist_ok=True)
with open(f"{args.save_path}/unc_result/{args.method}-{args.optim}_uncertainty.pkl", 'wb') as f:
    pickle.dump(unc, f)

# Save Reliability Diagram 
utils.save_reliability_diagram(args.method, args.optim, args.save_path, unc, False)