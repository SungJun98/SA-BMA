import argparse
import pickle, os, copy

import utils.data.data as data
import utils.utils as utils
from utils.swag import swag, swag_utils

import torch
import numpy as np

from hessian_eigenthings import compute_hessian_eigenthings
import torch.nn.functional as F

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
    choices=['resnet18', 'resnet18-noBN', "vitb16-i21k"],
    help="model name (default : resnet18)")

parser.add_argument("--swag_load_path", type=str, default=None,
    help="path to load saved swag model (default: None)",)

parser.add_argument("--vi_load_path", type=str, default=None,
    help="path to load saved vi model (default: None)",)

parser.add_argument("--sabma_load_path", type=str, default=None,
    help="path to load saved sabma model (default: None)",)

parser.add_argument("--emcmc_load_path", type=str, default=None,
    help="path to load save emcmc model (default : None)")

parser.add_argument("--ptl_load_path", type=str, default=None,
    help="path to load save pre-train your loss model (default : None)")

parser.add_argument("--load_path", type=str, default=None,
    help="path to load saved model (default: None)")

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


def load_swag_model_to_base_model(model, bma_sample):
    """
    Load SWAG model in form of base model to calculate hessian
    """
    import collections
    state_dict = collections.OrderedDict()        
    for key in bma_sample.base.state_dict().keys():
        if ("sq_mean" in key) or ("cov_mat_sqrt" in key):
            pass
        else:
            if "-" in key:
                key_ = key.split(".")[-1]
            else:
                key_ = key
            new_key = key_.replace("-", ".")
            
            if (not "running" in new_key) and ("mean" in new_key):
                new_key = new_key.split("_")[0]
            
            state_dict[new_key] = bma_sample.base.state_dict()[key]
    model.load_state_dict(state_dict)      
    return model




# Set Device and Seed------------------------------------------------------
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {args.device}")

if args.model.split("-")[-1] == "noBN":
    args.batch_norm = False
else:
    args.batch_norm = True
    
# args.aug = False

utils.set_seed(args.seed)
#---------------------------------------------------------------------------

# Load Data -----------------------------------------------------------------
tr_loader, _, te_loader, num_classes = utils.get_dataset(dataset=args.dataset,
                                            data_path=args.data_path,
                                            dat_per_cls=args.dat_per_cls,
                                            use_validation=args.use_validation, 
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            seed=args.seed,
                                            aug=args.batch_norm,
                                            )
if args.dat_per_cls >= 0:
    print(f"Load Data : {args.dataset}-{args.dat_per_cls}shot")
else:
    print(f"Load Data : {args.dataset}")
    
#---------------------------------------------------------------------------

## Define Model--------------------------------------------------------------
model = utils.get_backbone(args.model, num_classes, args.device, args.pre_trained)
last_layer_name = None
if args.last_layer:
    # Get last layer name
    for name, mod in model.named_modules():
        model = mod
        last_layer_name = name
#----------------------------------------------------------------------------

## Load Model ---------------------------------------------------------------
if args.swag_load_path is not None:
    """
    # Load SWAG weight 
    if args.load_path is not None:
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint)
    """
    # Get bma weights list
    bma_load_paths = sorted(os.listdir(args.swag_load_path))
    
elif args.sabma_load_path is not None:
    bma_load_paths = sorted(os.listdir(args.sabma_load_path))
    model_path = args.sabma_load_path.split("/")[:-2]
    model_path = '/'+os.path.join(*model_path)
    model_path = os.path.join(model_path, "sabma-sabma_best_val_model.pt" )
    model = torch.load(model_path)
    model = model.backbone

elif args.vi_load_path is not None:
    bma_load_paths = sorted(os.listdir(args.vi_load_path))

elif args.emcmc_load_path is not None:
    bma_load_paths = sorted(os.listdir(args.emcmc_load_path))

elif args.ptl_load_path is not None:
    checkpoint = torch.load(args.ptl_load_path)
    for key in list(checkpoint.keys()):
        if 'backbone.' in key:
            new_key = key.replace('backbone.', '')
            checkpoint[new_key] = checkpoint.pop(key)
        elif 'classifier.' in key:
            new_key = key.replace('classifier', 'fc')
            checkpoint[new_key] = checkpoint.pop(key)
    model.load_state_dict(checkpoint)

else:
    checkpoint = torch.load(args.load_path)
    if hasattr(checkpoint, "temperature"):
        # Load Temperature Scaled Model
        model = checkpoint        
    else:
        model.load_state_dict(checkpoint["state_dict"], strict=args.batch_norm)
        
model.to(args.device)
# ------------------------------------------------------------------------------

## Set save path ---------------------------------------------------------------
if args.swag_load_path is not None:
    save_path = f"{args.swag_load_path}/performance"
    
elif args.sabma_load_path is not None:
    save_path = f"{args.sabma_load_path}/performance"

elif args.vi_load_path is not None:
    save_path = f"{args.vi_load_path}/performance"

elif args.emcmc_load_path is not None:
    save_path = f"{args.emcmc_load_path}/performance"

elif args.ptl_load_path is not None:
    save_path = args.ptl_load_path.split("/")[:-1]
    save_path = os.path.join(*save_path)
    save_path = f"/{save_path}/performance"
else:
    save_path = args.load_path.split("/")[:-1]
    save_path = os.path.join(*save_path)
    save_path = f"/{save_path}/performance"
os.makedirs(save_path, exist_ok=True)
print(f"Save path : {save_path}")
# ------------------------------------------------------------------------------

# model.eval()

criterion = torch.nn.CrossEntropyLoss()

## Calculate Hessian ------------------------------------------------------------
if (args.swag_load_path is not None) or (args.vi_load_path is not None) or (args.sabma_load_path is not None) or (args.emcmc_load_path is not None):
    model_num_list = list(); acc_list = list(); ece_list = list(); nll_list = list()
    tr_cum_eigenval_list = list() ; tr_max_eigenval_list = list()
    for path in bma_load_paths:
        if args.emcmc_load_path is None:
            model_num = path.split(".")[0]
            model_num = model_num.split("-")[-1]
        else:
            model_num = path.split(".")[0]
            model_num = model_num.split("_")[-1]
        
        # get sampled model
        if args.swag_load_path is not None:
            bma_sample = torch.load(f"{args.swag_load_path}/{path}")
            model = load_swag_model_to_base_model(model, bma_sample)
            # model = torch.load(f"{args.swag_load_path}/{path}")
        elif args.vi_load_path is not None:
            bma_sample = torch.load(f"{args.vi_load_path}/{path}")
            model = bma_sample
            # model = torch.load(f"{args.vi_load_path}/{path}")
        elif args.sabma_load_path is not None:
            bma_sample = torch.load(f"{args.sabma_load_path}/{path}")
            model.load_state_dict(bma_sample, strict=False)
        elif args.emcmc_load_path is not None:
            try:
                bma_sample = torch.load(f"{args.emcmc_load_path}/{path}", map_location=args.device)
            except:
                pass
            model.load_state_dict(bma_sample)
            
            
        if (args.swag_load_path) is not None or (args.vi_load_path is not None):
            res = utils.eval(te_loader, bma_sample, criterion, args.device)
        elif (args.sabma_load_path is not None) or (args.emcmc_load_path is not None):
            res = utils.eval(te_loader, model, criterion, args.device)
        for p in model.parameters():
            p.requires_grad_(True)
        
        print(f"# {model_num} / Test Accuracy : {res['accuracy']:8.4f}% / ECE : {res['ece']} / NLL : {res['nll']}")
        
        model_num_list.append(model_num); acc_list.append(res['accuracy']); ece_list.append(res['ece']); nll_list.append(res['nll'])
        # get eigenvalue for train set
        try:
            tr_eigenvals, _ = compute_hessian_eigenthings(
                    model,
                    tr_loader,
                    criterion,
                    num_eigenthings=args.num_eigen,
                    mode="lanczos",
                    # mode="power_iter",
                    # power_iter_steps=50,
                    max_possible_gpu_samples=args.max_possible_gpu_samples,
                    # momentum=args.momentum,
                    use_gpu=True,
                )

            tr_cum_eigenval_list.append(tr_eigenvals)
            tr_max_eigenval_list.append(max(tr_eigenvals))
            if args.swag_load_path is not None:
                print(f"Successfully get {model_num}-th swag bma model eigenvalues for train set")
            elif args.vi_load_path is not None:
                print(f"Successfully get {model_num}-th vi bma model eigenvalues for train set")
            elif args.sabma_load_path is not None:
                print(f"Successfully get {model_num}-th sabma bma model eigenvalues for train set")
            print(f"Train Eigenvalues for {model_num}-th bma model : {tr_eigenvals}")
        except:
           print(f"Numerical Issue on {model_num}-th model with train data")
           tr_cum_eigenval_list.append(999999999)
           tr_max_eigenval_list.append(999999999)
        
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
