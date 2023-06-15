import argparse
import torch
import pickle
import os, copy
import utils.data.data as data
import utils.utils as utils
import numpy as np

from hessian_eigenthings import compute_hessian_eigenthings

from utils.swag import swag, swag_utils
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Get Hessian of saved model")

parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")

## Data ---------------------------------------------------------
parser.add_argument(
    "--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"],
                    help="dataset name")

parser.add_argument(
    "--data_path",
    type=str,
    default='/data1/lsj9862/data/cifar10',
    help="path to datasets location (default: None)",)

parser.add_argument("--batch_size", type=int, default = 256,
            help="batch size (default : 256)")

parser.add_argument("--num_workers", type=int, default=4,
            help="number of workers (default : 4)")

parser.add_argument("--use_validation", action='store_true', default=True,
            help ="Use validation for hyperparameter search (Default : True)")

parser.add_argument("--dat_per_cls", type=int, default=-1,
            help="Number of data points per class in few-shot setting. -1 denotes deactivate few-shot setting (Default : -1)")
#----------------------------------------------------------------


## Model ---------------------------------------------------------
parser.add_argument(
    "--model",
    type=str, default='resnet18', required=True,
    choices=['resnet18', 'resnet50', 'resnet101', 'wideresnet28x10', 'wideresnet40x10',
            'resnet18-noBN', 'resnet50-noBN', 'resnet101-noBN', 'wideresnet28x10-noBN', 'wideresnet40x10-noBN',
            "vitb16-i21k"],
    help="model name (default : resnet18)")

parser.add_argument("--swag_load_path", type=str, default=None,
    help="path to load saved swag model (default: None)",)

parser.add_argument("--load_path", type=str, default=None,
    help="path to load saved model (default: None)")

parser.add_argument(
    "--swag",
    action='store_true',
    help ="When model trained with swag (Default : False)"
)

parser.add_argument(
    "--pre_trained", action='store_true', default=True,
    help="Using pre-trained model from zoo"
    )

parser.add_argument(
    "--last_layer",
    action='store_true',
    help ="Calculate the hessian of last layer only"
)
#-------------------------------------------------------------------------

## Arguments for hessian approximate --------------------------------------
parser.add_argument("--num_eigen", type=int, default=5,
    help="number of eigenvalues to get (default : 5)")

parser.add_argument("--max_possible_gpu_samples", type=int,
    default=2048, help="number of max possible samples on gpu (default : 2048")
#-------------------------------------------------------------------------

args = parser.parse_args()
#-------------------------------------------------------------------------

# Set Device and Seed------------------------------------------------------
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {args.device}")

if args.model.split("-")[-1] == "noBN":
    args.batch_norm = False
else:
    args.batch_norm = True
    
args.aug = False

utils.set_seed(args.seed)
#---------------------------------------------------------------------------

# Load Data -----------------------------------------------------------------
if not args.last_layer:
    tr_loader, val_loader, te_loader, num_classes = utils.get_dataset(args.dataset,
                                                                args.data_path,
                                                                args.batch_size,
                                                                args.num_workers,
                                                                use_validation = args.use_validation,
                                                                aug = args.aug,
                                                                dat_per_cls = args.dat_per_cls,
                                                                seed = args.seed)
    print(f"Load Data : {args.dataset}")
        
else:
    if args.dataset == 'cifar10':
        tr_loader, _, te_loader, num_classes = data.get_cifar10_fe(fe_dat=args.model,
                                                    batch_size=args.batch_size,
                                                    num_workers=0,
                                                    use_validation=args.use_validation,
                                                    dat_per_cls=args.dat_per_cls)
    elif args.dataset == 'cifar100':
        tr_loader, _, te_loader, num_classes = data.get_cifar100_fe(fe_dat=args.model,
                                                    batch_size=args.batch_size,
                                                    num_workers=0,
                                                    use_validation=args.use_validation,
                                                    dat_per_cls=args.dat_per_cls)
    print(f"Load Data : Feature Extracted {args.dataset}")
#---------------------------------------------------------------------------

## Define Model--------------------------------------------------------------
model = utils.get_backbone(args.model, num_classes, args.device, args.pre_trained)
if args.last_layer:
    # Get last layer name
    for name, mod in model.named_modules():
        model = mod
#----------------------------------------------------------------------------

## Load Model ---------------------------------------------------------------
if args.swag:
    # Load SWAG weight 
    if args.load_path is not None:
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint)
    # Get bma weights list
    bma_load_paths = sorted(os.listdir(args.swag_load_path))
else:
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
# ------------------------------------------------------------------------------

## Set save_path ---------------------------------------------------------------
if args.swag:
    save_path = f"{args.swag_load_path}/performance"
    os.makedirs(save_path, exist_ok=True)
else:
    save_path = f"{args.load_path}/performance"
    os.makedirs(save_path, exist_ok=True)
# ------------------------------------------------------------------------------

model.eval()

criterion = torch.nn.CrossEntropyLoss()
## Calculate Hessian ------------------------------------------------------------
if args.swag:
    model_num_list = list(); acc_list = list(); ece_list = list(); nll_list = list()
    tr_cum_eigenval_list = list() ; tr_max_eigenval_list = list()
    for path in bma_load_paths:
        model_num = path.split(".")[0]
        model_num = model_num.split("-")[-1]
        
        # get sampled model
        bma_sample = torch.load(f"{args.swag_load_path}/{path}")
        bma_state_dict = utils.list_to_state_dict(model, bma_sample, last=True)     ## If use full stochastic SWAG, you need to change argument last to False
        model.load_state_dict(bma_state_dict, strict=False)
        
        if args.batch_norm:
          swag_utils.bn_update(tr_loader, model)
        
        res = utils.eval(te_loader, model, criterion, args.device)
        print(f"Test Accuracy : {res['accuracy']:8.4f}% / ECE : {res['ece']} / NLL : {res['nll']}")
        acc_list.append(res['accuracy']); ece_list.append(res['ece']); nll_list.append(res['nll'])
        
        # get eigenvalue for train set
        try:
            tr_eigenvals, _ = compute_hessian_eigenthings(
                    model,
                    tr_loader,
                    criterion,
                    num_eigenthings=args.num_eigen,
                    mode="lanczos", #"power_iter"
                    # power_iter_steps=50,
                    max_possible_gpu_samples=args.max_possible_gpu_samples,
                    # momentum=args.momentum,
                    use_gpu=True,
                )

            tr_cum_eigenval_list.append(tr_eigenvals)
            tr_max_eigenval_list.append(max(tr_eigenvals))
            print(f"Successfully get {model_num}-th swag bma model eigenvalues for train set")
            print(f"Train Eigenvalues for {model_num}-th bma model : {tr_eigenvals}")
        except:
           print(f"Numerical Issue on {model_num}-th model with train data")
           tr_cum_eigenval_list.append(99999)
           tr_max_eigenval_list.append(99999)
        
        # save results
        performance = dict({"model_num" : model_num_list,
                            "accuracy": acc_list,
                            "ece" : ece_list,
                            "nll" : nll_list,
                            "tr_cum_eign" : tr_cum_eigenval_list,
                            "tr_max_eign" : tr_max_eigenval_list})
        
        torch.save(performance, f"{save_path}/performance.pt")        
        print("-"*15)
        
else:
    # Check performance
    res = utils.eval(te_loader, model, criterion, args.device)
    print(f"Test Accuracy : {res['accuracy']:8.4f}% / ECE : {res['ece']} / NLL : {res['nll']}")    
    
    # get eigenvalue for train set
    tr_eigenvals, _ = compute_hessian_eigenthings(
                model,
                tr_loader,
                criterion,
                num_eigenthings=args.num_eigen,
                mode="lanczos",
                # power_iter_steps=args.num_steps,
                max_possible_gpu_samples=args.max_possible_gpu_samples,
                # momentum=args.momentum,
                use_gpu=True,
            )
    print(f"Train Eigenvalues : {tr_eigenvals}")
    print(f"Max Train Eigenvalue : {max(tr_eigenvals)}")
    
    # save results
    performance = dict({"model_num" : 0,
                        "accuracy": res['accuracy'],
                        "ece" : res['ece'],
                        "nll" : res['nll'],
                        "tr_cum_eign" : tr_eigenvals,
                        "tr_max_eign" : max(tr_eigenvals)})
        
    torch.save(performance, f"{save_path}/performance.pt")        
    print("-"*15)
# -----------------------------------------------------------------------------