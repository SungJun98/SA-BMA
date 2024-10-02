import argparse
import os, time, tabulate
import wandb
from tqdm import tqdm
import numpy as np

import torch

import utils.utils as utils
from utils.mcmc import mcmc_utils
from utils import temperature_scaling as ts

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="training baselines (MCMC)")

parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")

parser.add_argument("--method", type=str, default="mcmc",
                    choices=["mcmc", "emcmc"],
                    help="Learning Method")

parser.add_argument("--no_amp", action="store_true", default=False, help="Deactivate AMP")

parser.add_argument("--print_epoch", type=int, default=10, help="Printing epoch")

parser.add_argument("--ignore_wandb", action="store_true", default=False, help="Deactivate wandb")
parser.add_argument("--wd_project", default=None, type=str, help="name of wandb project")
parser.add_argument("--wd_entity", default=None, type=str, help="entity of wandb")

parser.add_argument("--resume", type=str, default=None,
    help="path to load saved model to resume training (default: None)",)

parser.add_argument("--linear_probe", action="store_true", default=False,
        help = "When we do Linear Probing (Default : False)")

parser.add_argument("--tol", type=int, default=30,
        help="tolerance for early stopping (Default : 30)")

## Data ---------------------------------------------------------
parser.add_argument("--dataset",
                    type=str, default="cifar10", choices=["cifar10", "cifar100",
                                        "eurosat", "dtd", "oxford_flowers",
                                        "oxford_pets", "food101", "ucf101", 'fgvc_aircraft',
                                        'mnist'],
                    help="dataset name")

parser.add_argument(
    "--data_path",
    type=str,
    default=None,
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
    type=str, default='resnet18',
    choices=['mlp', 'resnet18', 'resnet50', 'resnet101',
            'resnet18-noBN',
            'vitb16-i1k', "vitb16-i21k"],
    help="model name (default : resnet18)")

parser.add_argument(
    "--pre_trained", action='store_true', default=False,
    help="Using pre-trained model from zoo"
    )

parser.add_argument("--save_path",
            type=str, default=None,
            help="Path to save best model dict")
#----------------------------------------------------------------

## Optimizer Hyperparameter --------------------------------------
parser.add_argument("--optim", type=str, default="sgld",
                    choices=["sgld"],
                    help="Optimization options")

parser.add_argument("--lr_init", type=float, default=0.1,
                help="learning rate (Default : 0.1)")

parser.add_argument("--lr_end", type=float, default=1e-4,
                help="learning rate (Default : 1e-4)")

parser.add_argument('--decay_scheme', type=str, default='cyclical')


parser.add_argument("--epochs", type=int, default=300, metavar="N",
    help="number epochs to train (default : 300)")

parser.add_argument("--wd", type=float, default=5e-4, help="weight decay (default: 5e-4)")

parser.add_argument("--momentum", type=float, default=0.9,
                help="momentum (Default : 0.9)")
#----------------------------------------------------------------

## MCMC ---------------------------------------------------------
parser.add_argument("--n_cycle", type=int, default=4,
                help='number of cycle w.r.t. lr and posterior sampling')
parser.add_argument("--temp", type=float, default=1e-2,
                help='temperature')
#----------------------------------------------------------------

## E-MCMC ---------------------------------------------------------
parser.add_argument("--eta", type=float, default=4e-4,
                help='reguarization for entropy')
#----------------------------------------------------------------

## bma or metrics -----------------------------------------------
parser.add_argument("--bma_num_models", type=int, default=12, help="Number of models for bma")
parser.add_argument("--val_mc_num", type=int, default=1, help="number of MC sample in validation phase")
parser.add_argument("--eps", type=float, default=1e-8, help="small float to calculate nll")
parser.add_argument("--num_bins", type=int, default=15, help="bin number for ece")
#----------------------------------------------------------------

args = parser.parse_args()
args.scheduler = None
args.rho = 0.0
#----------------------------------------------------------------

if not args.ignore_wandb:
    wandb.init(project=args.wd_preoject, entity=args.wd_entity)

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

print(f"Device : {args.device} / Seed : {args.seed} / Augmentation : {args.aug}")
print("-"*30)
#------------------------------------------------------------------

# Set BMA and Save Setting-----------------------------------------
if args.method == 'dnn':
    args.bma_num_models = 1

args.save_path = utils.set_save_path(args)
os.makedirs(args.save_path, exist_ok=True) 
print(f"Save Results on {args.save_path}")
print("-"*30)
#------------------------------------------------------------------

# wandb config-----------------------------------------------------
if not args.ignore_wandb:
    wandb.config.update(args)
    wandb.run.name = utils.set_wandb_runname(args)
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
                                                        model_name=args.model
                                                        )
elif args.dataset in ["eurosat", "dtd", "oxford_flowers", "oxford_pets", "food101", "ucf101", 'fgvc_aircraft']:
    tr_loader, val_loader, te_loader, num_classes = utils.get_dataset_dassl(args)
    


if args.dat_per_cls >= 0:
    print(f"Load Data : {args.dataset}-{args.dat_per_cls}shot")
else:
    print(f"Load Data : {args.dataset}")
print("-"*30)
#------------------------------------------------------------------

# Define Model-----------------------------------------------------
if args.method == 'mcmc':
    model = utils.get_backbone(args.model, num_classes, args.device, args.pre_trained)
    model.to(args.device)
elif args.method == 'emcmc':
    model_s = utils.get_backbone(args.model, num_classes, args.device, args.pre_trained)
    
    model_a = utils.get_backbone(args.model, num_classes, args.device, args.pre_trained)
    model_a = mcmc_utils.resample(model_a, model_s)
    
    model_s, model_a = model_s.to(args.device), model_a.to(args.device)


tab_name = ["Model", "# of Tr Params"]
num_params = 0
if args.method == 'mcmc':
    for param in model.parameters():
        num_params += param.numel()
elif args.method == 'emcmc':
    for param in model_s.parameters():
        num_params += param.numel()
    for param in model_a.parameters():
        num_params += param.numel()
tab_contents= [args.model, num_params]
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
if args.method == 'mcmc':
    optim_param = model.parameters()
elif args.method == 'emcmc':
    optim_param = list(model_s.parameters()) + list(model_a.parameters())

optimizer = torch.optim.SGD(optim_param,
                        lr=args.lr_init, weight_decay=args.wd,
                        momentum=args.momentum)
print(f"Set {args.optim} optimizer with lr_init {args.lr_init} / wd {args.wd}") # / momentum {args.momentum}")
print("-"*30)
#----------------------------------------------------------------

start_epoch = 1

## Set AMP --------------------------------------------------------------------------
# scaler, first_step_scaler, second_step_scaler = utils.get_scaler(args)
# print("-"*30)
#------------------------------------------------------------------------------------


## Training -------------------------------------------------------------------------
print(f"Start training {args.method} with {args.optim} optimizer from {start_epoch} epoch!")

## print setting
columns = ["epoch", "method", "lr", "tr_loss", "tr_acc",
        "val_loss", "val_acc", "val_nll", "val_ece",
        "time"]

w_list = []
for epoch in range(start_epoch, int(args.epochs)+1):
    time_ep = time.time()

    ## train
    if args.method == 'mcmc':
        tr_res = mcmc_utils.train_sgld(args, epoch, tr_loader, model, criterion, optimizer)
    elif args.method == 'emcmc':
        tr_res = mcmc_utils.train_emcmc(args, epoch, tr_loader, model_s, model_a, criterion, optimizer)
        if args.model == 'resnet18':
            mcmc_utils.additional_forward(args, tr_loader, model_a)
    
    time_ep = time.time() - time_ep
    
    ## valid
    if args.method == 'mcmc':
        val_res = utils.eval(val_loader, model, criterion, args.device, args.num_bins, args.eps)    
    elif args.method == 'emcmc':
        val_res = utils.eval(val_loader, model_s, criterion, args.device, args.num_bins, args.eps)    
    
    ## print result
    values = [epoch, f"{args.method}-{args.optim}", tr_res["lr"], tr_res["loss"], tr_res["accuracy"],
            val_res["loss"], val_res["accuracy"], val_res["nll"], val_res["ece"],
            time_ep]
    
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % args.print_epoch == 1:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)
    

    ## test and save sample
    if ((epoch - 1) % int(args.epochs//args.n_cycle)) >= (int(args.epochs//args.n_cycle) - (args.bma_num_models//args.n_cycle)):
        # test
        if args.method == 'mcmc':
            te_res = utils.eval(te_loader, model, criterion, args.device, args.num_bins, args.eps)
        elif args.method == 'emcmc':
            te_res = utils.eval(te_loader, model_s, criterion, args.device, args.num_bins, args.eps)
        print(f"Epoch: {epoch} / Acc: {te_res['accuracy']:.2f} / ECE: {te_res['ece']:.3f} / NLL: {te_res['nll']:.4f}")
        
        if not args.ignore_wandb:
            wandb.run.summary[f'test accuracy_{epoch}'] = te_res['accuracy']
            wandb.run.summary[f'test ece_{epoch}'] = te_res['ece']
            wandb.run.summary[f'test nll_{epoch}'] = te_res['nll']
            
        # save sample
        if args.method == 'mcmc':
            w_list.append(mcmc_utils.save_sample(args, model, f'{args.save_path}/{epoch}.pt'))
        elif args.method == 'emcmc':
            w_list.append(mcmc_utils.save_sample(args, model_s, f'{args.save_path}/{epoch}.pt'))
    
    ## wandb
    if not args.ignore_wandb:
        wandb.log({"Train Loss ": tr_res["loss"], "Train Accuracy" : tr_res["accuracy"],
                "Validation loss" : val_res["loss"], "Validation Accuracy" : val_res["accuracy"],
                "Validation nll" : val_res["nll"], "Validation ece" : val_res["ece"],
                "lr" : tr_res["lr"],})
    
#------------------------------------------------------------------------------------------------------------



## Test ------------------------------------------------------------------------------------------------------
##### Get test nll, Entropy, ece, Reliability Diagram on best model
utils.set_seed(args.seed)
print(f"Start Bayesian Model Averaging with {len(w_list)} samples")
if args.method == 'mcmc':
    bma_res = mcmc_utils.bma_mcmc(args, te_loader, num_classes, w_list, model, criterion)
elif args.method == 'emcmc':
    bma_res = mcmc_utils.bma_mcmc(args, te_loader, num_classes, w_list, model_s, criterion)

table = [["Num BMA models", "Test Accuracy", "Test NLL", "Test Ece"],
        [len(w_list), format(bma_res['accuracy'], '.4f'), format(bma_res['nll'], '.4f'), format(bma_res['ece'], '.4f')]]
print(tabulate.tabulate(table, tablefmt="simple"))

if not args.ignore_wandb:
    wandb.run.summary['bma accuracy'] = bma_res['accuracy']
    wandb.run.summary['bma nll'] = bma_res['nll']
    wandb.run.summary['bma ece'] = bma_res['ece']