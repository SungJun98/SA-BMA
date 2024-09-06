import argparse
import os, sys
import time, copy
import tabulate

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import wandb

import utils.utils as utils
from utils.sam import sam, sam_utils
from utils.swag import swag, swag_utils
from utils.sabma import sabma, sabma_utils
from utils import temperature_scaling as ts
import utils.data.data as data

from utils.vi import vi_utils
from bayesian_torch.models.dnn_to_bnn import get_kl_loss

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="training sabma with VI from scratch")

parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")

parser.add_argument("--method", type=str, default="sabma",
                    choices=["sabma"],
                    help="Learning Method")

parser.add_argument("--no_amp", action="store_true", default=True, help="Deactivate AMP")

parser.add_argument("--print_epoch", type=int, default=10, help="Printing epoch")

parser.add_argument("--ignore_wandb", action="store_true", default=False, help="Deactivate wandb")

parser.add_argument("--resume", type=str, default=None,
    help="path to load saved model to resume training (default: None)",)

parser.add_argument("--tol", type=int, default=30,
        help="tolerance for early stopping (Default : 30)")

## Data ---------------------------------------------------------
parser.add_argument(
    "--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100",],
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

parser.add_argument("--no_aug", action="store_true", default=True,
            help="Deactivate augmentation")
#----------------------------------------------------------------

## Model ---------------------------------------------------------
parser.add_argument(
    "--model",
    type=str, default='resnet18',
    choices=['resnet18', 'vitb16-i1k'],
    help="model name (default : resnet18)")

parser.add_argument(
    "--pre_trained", action='store_true', default=False,
    help="Using pre-trained model from zoo"
    )

parser.add_argument("--save_path",
            type=str, default="/data2/lsj9862/exp_result/scratch",
            help="Path to save best model dict")
#----------------------------------------------------------------

## Optimizer Hyperparameter --------------------------------------
parser.add_argument("--enc_optim", type=str, default="sgd",
                    choices=["sgd", "sam"],
                    help="Optimization options")

parser.add_argument("--lr_init", type=float, default=0.01,
                help="learning rate (Default : 0.01)")

parser.add_argument("--momentum", type=float, default=0.9,
                help="momentum (Default : 0.9)")

parser.add_argument("--epochs", type=int, default=300, metavar="N",
    help="number epochs to train (default : 300)")

parser.add_argument("--wd", type=float, default=5e-4, help="weight decay (default: 5e-4)")

parser.add_argument("--rho", type=float, default=0.05, help="size of pertubation ball for SAM / FSAM")

# Scheduler
parser.add_argument("--scheduler", type=str, default='cos_decay', choices=['constant', "step_lr", "cos_anneal", "swag_lr", "cos_decay"])

parser.add_argument("--lr_min", type=float, default=1e-8,
                help="Min learning rate. (Cosine Annealing Warmup Restarts)")

parser.add_argument("--warmup_t", type=int, default=10,
                help="Linear warmup step size. (Cosine Annealing Warmup Restarts)")

parser.add_argument("--warmup_lr_init", type=float, default=1e-7,
                help="Linear warmup initial learning rate. (Cosine Annealing Warmup Restarts)")
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
parser.add_argument("--vi_type", type=str, default="Flipout", choices=["Reparameterization", "Flipout"],
                help="Set type of variational inference (Default: Reparameterization)")
parser.add_argument("--vi_moped_delta", type=float, default=0.05,
                help="Set initial perturbation factor for weight in MOPED framework (Default: 0.05)")
parser.add_argument("--kl_beta", type=float, default=1.0,
                help="Hyperparameter to adjust kld term on vi loss function (Default: 1.0)")
#----------------------------------------------------------------

## bma or metrics -----------------------------------------------
parser.add_argument("--val_mc_num", type=int, default=1, help="number of MC sample in validation phase")
parser.add_argument("--eps", type=float, default=1e-8, help="small float to calculate nll")
parser.add_argument("--bma_num_models", type=int, default=30, help="Number of models for bma")
parser.add_argument("--num_bins", type=int, default=15, help="bin number for ece")
parser.add_argument("--no_save_bma", action='store_true', default=True, ## change to default=False to save bma models
            help="Deactivate saving model samples in BMA")
#----------------------------------------------------------------

args = parser.parse_args()
#----------------------------------------------------------------

if not args.ignore_wandb:
    wandb.init(project="SA-BTL", entity='sungjun98')

# Set Device and Seed--------------------------------------------
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.batch_norm = True
args.aug = False

utils.set_seed(args.seed)

print(f"Device : {args.device} / Seed : {args.seed} / Augmentation : {args.aug}")
print("-"*30)
#------------------------------------------------------------------

# Set BMA and Save Setting-----------------------------------------
args.save_path = f"{args.save_path}/seed_{args.seed}/{args.dataset}/"
args.save_path += f"scratch_{args.model}/vi-sabma-{args.enc_optim}/"
args.save_path += f"{args.scheduler}/{args.lr_init}_{args.wd}_{args.vi_moped_delta}_{args.kl_beta}"  
args.save_path += f"{args.rho}"
print(f"Save Results on {args.save_path}")
print("-"*30)
#------------------------------------------------------------------

# wandb config-----------------------------------------------------
if not args.ignore_wandb:
    wandb.config.update(args)
    run_name = f"seed{args.seed}_vi-sabma-{args.enc_optim}_scratch-{args.model}_{args.dataset}_{args.lr_init}_{args.wd}_{args.vi_moped_delta}_{args.kl_beta}_{args.rho}"
    wandb.run.name = run_name
#------------------------------------------------------------------

# Load Data --------------------------------------------------------
if args.dataset in ['mnist', 'cifar10', 'cifar100']:
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
model = utils.get_backbone(args.model, num_classes, args.device, args.pre_trained)

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

tab_name = ["Model", "# of SGD Params", "# of SABMA Params", "# of Total Params"]
enc_params = list();enc_num_params = 0
sabma_params = list(); sabma_num_params = 0
num_params = 0
for name, param in model.named_parameters():
    num_params += param.numel()
    if ('la' in name) or ('bn' in name) or ('fc' in name):
    # if ('fc' in name):
        sabma_num_params += param.numel()
        sabma_params.append(param)
    else:
        enc_num_params += param.numel()
        enc_params.append(param)
tab_contents= [args.model, enc_num_params, sabma_num_params, num_params]
table = [tab_name, tab_contents]
print(tabulate.tabulate(table, tablefmt="simple"))

print("-"*30)
#-------------------------------------------------------------------

# Set Criterion-----------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()
print("Set Criterion as Cross Entropy")
print("-"*30)
#-------------------------------------------------------------------

# Set Optimizer--------------------------------------
if args.enc_optim == 'sgd':
    enc_optim = torch.optim.SGD(enc_params,
                            lr=args.lr_init, weight_decay=args.wd,
                            momentum=args.momentum)
elif args.enc_optim == 'sam':
    base_optimizer = torch.optim.SGD
    enc_optim = sam.SAM(enc_params, base_optimizer,
                        rho=args.rho,
                        lr=args.lr_init,
                        momentum=args.momentum,
                        weight_decay=args.wd)

base_optimizer = torch.optim.SGD
# sabma_optim = sabma.SABMA_optim_scratch(sabma_params, base_optimizer,
#                         rho=args.rho, lr=args.lr_init, momentum=args.momentum,
#                         weight_decay=args.wd)
sabma_optim = sam.SAM(sabma_params, base_optimizer,
                        rho=args.rho,
                        lr=args.lr_init,
                        momentum=args.momentum,
                        weight_decay=args.wd)
print("-"*30)
#----------------------------------------------------------------
    
## Set Scheduler----------------------------------------------------
from timm.scheduler.cosine_lr import CosineLRScheduler
if args.enc_optim in ["sgd", "adam"]:
    enc_scheduler = CosineLRScheduler(optimizer = enc_optim,
                                t_initial= args.epochs,
                                lr_min=args.lr_min,
                                cycle_mul=1,
                                cycle_decay=1,
                                cycle_limit=1,
                                warmup_t=args.warmup_t,
                                warmup_lr_init=args.warmup_lr_init,
                                    )
elif args.enc_optim in ["sam", "fsam", "sabma"]:
    enc_scheduler = CosineLRScheduler(optimizer = enc_optim.base_optimizer,
                                t_initial= args.epochs,
                                lr_min=args.lr_min,
                                cycle_mul=1,
                                cycle_decay=1,
                                cycle_limit=1,
                                warmup_t=args.warmup_t,
                                warmup_lr_init=args.warmup_lr_init,
                                    )
    
sabma_scheduler = CosineLRScheduler(optimizer = sabma_optim.base_optimizer,
                        t_initial= args.epochs,
                        lr_min=args.lr_min,
                        cycle_mul=1,
                        cycle_decay=1,
                        cycle_limit=1,
                        warmup_t=args.warmup_t,
                        warmup_lr_init=args.warmup_lr_init,
                            )  
    
    
print(f"Set cos decay lr scheduler")
print("-"*30)
#-------------------------------------------------------------------

## Training -------------------------------------------------------------------------
## print setting
columns = ["epoch", "method", "lr", "tr_loss", "tr_acc", "val_loss", "val_acc", "val_nll", "val_ece", "time"]

best_val_loss=9999 ; best_val_acc=0 ; best_epoch=0 ; cnt=0
for epoch in range(1, int(args.epochs)+1):
    time_ep = time.time()
    ## lr scheduling
    lr = enc_optim.param_groups[0]['lr']

    ## train ----------------------------------------------------------
    loss_sum = 0.0
    correct = 0.0

    num_objects_current = 0
    num_batches = len(tr_loader)

    model.train()
    for batch_idx, batch in enumerate(tr_loader):
        try:
            X = batch["img"].to(args.device)
            y = batch["label"].to(args.device)
        except:
            X, y = batch[0].to(args.device), batch[1].to(args.device)
            
        enc_optim.zero_grad()
        sabma_optim.zero_grad()
            
        ## first forward & backward
        pred = model(X)
        kl = get_kl_loss(model)
        loss = criterion(pred, y)
        loss += args.kl_beta * kl / args.batch_size    
        loss.backward()
        
        if args.enc_optim in ['sgd', 'adam']:
            enc_optim.step()
        else:
            enc_optim.first_step(zero_grad=True, amp=False)
        sabma_optim.first_step(zero_grad=True, amp=False)
    
        ## second forward-backward pass
        pred = model(X)
        kl = get_kl_loss(model)
        loss = criterion(pred, y)
        loss += args.kl_beta * kl / args.batch_size
        loss.backward()
        if args.enc_optim in ['sam']:
            enc_optim.second_step(zero_grad=True, amp=False)   
        sabma_optim.second_step(zero_grad=True, amp=False)   
    
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss_sum += loss.data.item() * X.size(0)
        num_objects_current += X.size(0)
    
    tr_res = dict()
    tr_res['loss'] = loss_sum / num_objects_current
    tr_res['accuracy'] = correct / num_objects_current * 100
    # ----------------------------------------------------------------
        
    ## valid ----------------------------------------------------------
    loss_sum = 0.0
    num_objects_total = len(val_loader.dataset)

    preds = list()
    targets = list()

    model.eval()
    offset = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                input = batch["img"].to(args.device)
                target = batch["label"].to(args.device)
            except:
                input, target = batch[0].to(args.device), batch[1].to(args.device)
        
            pred = model(input)
            kl = get_kl_loss(model)
            loss = criterion(pred, target)
            loss += args.kl_beta * kl / args.batch_size
            loss_sum += loss.item() * input.size(0)
            
            preds.append(F.softmax(pred, dim=1).cpu().numpy())
            targets.append(target.cpu().numpy())
            offset += input.size(0)
    
    preds = np.vstack(preds)
    targets = np.concatenate(targets)

    accuracy = np.mean(np.argmax(preds, axis=1) == targets)
    nll = -np.mean(np.log(preds[np.arange(preds.shape[0]), targets] + args.eps))
    unc = utils.calibration_curve(preds, targets, args.num_bins)
        
    val_res = dict()
    val_res["loss"] = loss_sum / num_objects_total
    val_res["predictions"] = preds
    val_res["targets"] = targets
    val_res["accuracy"] = accuracy * 100.0
    val_res["nll"] = nll
    val_res["ece"] = unc['ece']
    # ----------------------------------------------------------------
    
    ## print result
    time_ep = time.time() - time_ep
    values = [epoch, f"{args.method}({args.enc_optim})", lr, tr_res["loss"], tr_res["accuracy"],
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
    if not args.ignore_wandb:
        wandb.log({"Train Loss ": tr_res["loss"], "Train Accuracy" : tr_res["accuracy"],
            "Validation loss" : val_res["loss"], "Validation Accuracy" : val_res["accuracy"],
            "Validation nll" : val_res["nll"], "Validation ece" : val_res["ece"],
            "lr" : lr,})

    ## Save best model (Early Stopping)
    if val_res["loss"] < best_val_loss:
        cnt = 0
        best_val_loss = val_res["loss"]
        best_val_acc = val_res["accuracy"]
        best_epoch = epoch

        # save state_dict
        os.makedirs(args.save_path, exist_ok=True)
        utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.enc_optim}_best_val.pt",
                              epoch = best_epoch,
                              state_dict = model.state_dict(),
                            )
        
        mean = vi_utils.get_vi_mean_vector(model)
        torch.save(mean,f'{args.save_path}/{args.method}-{args.enc_optim}_best_val_mean.pt')

        variance = vi_utils.get_vi_variance_vector(model)
        torch.save(variance, f'{args.save_path}/{args.method}-{args.enc_optim}_best_val_variance.pt')

    else:
        cnt +=1
        
    ## Early Stopping
    if cnt == args.tol:
        break
    
    if args.scheduler in ["cos_decay", "step_lr"]:
        enc_scheduler.step(epoch)
        sabma_scheduler.step(epoch)
#------------------------------------------------------------------------------------------------------------



## Test ------------------------------------------------------------------------------------------------------
##### Get test nll, Entropy, ece, Reliability Diagram on best model
### Load Best Model
print("Load Best Validation Model (Lowest Loss)")
state_dict_path = f'{args.save_path}/{args.method}-{args.enc_optim}_best_val.pt'
checkpoint = torch.load(state_dict_path)
if not args.ignore_wandb:
    wandb.run.summary['Best epoch'] = checkpoint["epoch"]
mean = None; variance = None

model = utils.get_backbone(args.model, num_classes, args.device, args.pre_trained)
vi_utils.load_vi(model, checkpoint)
mean = torch.load(f'{args.save_path}/{args.method}-{args.enc_optim}_best_val_mean.pt')
variance = torch.load(f'{args.save_path}/{args.method}-{args.enc_optim}_best_val_variance.pt')
    
model.to(args.device) 


#### MAP
res = vi_utils.bma_vi(None, te_loader, mean, variance, model, args.method, criterion, num_classes, temperature=None, bma_num_models=1,  bma_save_path=None, num_bins=args.num_bins, eps=args.eps)
print(f"1) Unscaled Results:")
table = [["Best Epoch", "Test Accuracy", "Test NLL", "Test Ece"],
        [best_epoch, format(res['accuracy'], '.2f'), format(res['nll'], '.4f'), format(res['ece'], '.4f')]]
print(tabulate.tabulate(table, tablefmt="simple", floatfmt="8.4f"))

if not args.ignore_wandb:
    wandb.run.summary['test accuracy'] = res['accuracy']
    wandb.run.summary['test nll'] = res['nll']
    wandb.run.summary["test ece"]  = res['ece']

#### Bayesian Model Averaging
utils.set_seed(args.seed)
bma_save_path = f"{args.save_path}/bma_models"
os.makedirs(bma_save_path, exist_ok=True) 
print(f"Start Bayesian Model Averaging with {args.bma_num_models} samples")
if args.no_save_bma:
        bma_save_path = None
        
bma_res = vi_utils.bma_vi(val_loader, te_loader, mean, variance, model, 'vi', criterion, num_classes, None, args.bma_num_models, bma_save_path, args.num_bins, args.eps)
bma_accuracy = bma_res['accuracy']
bma_nll = bma_res['nll']
bma_ece = bma_res['ece']

print(f"BMA Results:")
table = [["Num BMA models", "Test Accuracy", "Test NLL", "Test Ece"],
        [args.bma_num_models, format(bma_accuracy, '.4f'), format(bma_nll, '.4f'), format(bma_ece, '.4f')]]
print(tabulate.tabulate(table, tablefmt="simple"))

if not args.ignore_wandb:
    wandb.run.summary['bma accuracy'] = bma_accuracy
    wandb.run.summary['bma nll'] = bma_nll
    wandb.run.summary['bma ece'] = bma_ece