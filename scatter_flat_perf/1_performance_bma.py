# %%
import argparse
import torch
import pickle
import os, copy
import data, utils
import numpy as np

from baselines.swag import swag, swag_utils
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Get Hessian of saved model")

parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")

## Data ---------------------------------------------------------
parser.add_argument(
    "--dataset", type=str, default="cifar10", choices=["mnist-source", "mnist-down", "cifar10", "cifar100"],
                    help="dataset name")

parser.add_argument(
    "--data_path",
    type=str,
    default=None,
    help="path to datasets location (default: None)",)

parser.add_argument("--batch_size", type=int, default = 164,
            help="batch size (default : 64)")

parser.add_argument("--num_workers", type=int, default=4,
            help="number of workers (default : 4)")

parser.add_argument("--use_validation", action='store_true', default=True,
            help ="Use validation for hyperparameter search (Default : True)")
#----------------------------------------------------------------


## Model ---------------------------------------------------------
parser.add_argument(
    "--model",
    type=str, default='mlp', required=True,
    choices=['mlp', 'resnet18', 'resnet50', 'wideresnet28x10', 'wideresnet40x10',
            'resnet18-noBN', 'resnet50-noBN', 'wideresnet28x10-noBN', 'wideresnet40x10-noBN'],
    help="model name (default : mlp)")

parser.add_argument("--swag_load_path", type=str, default=None,
    help="path to load saved swag model (default: None)",)
#----------------------------------------------------------------

parser.add_argument("--eps", type=float, default=1e-8, help="small float to calculate nll")
parser.add_argument("--num_bins", type=int, default=15, help="bin number for ece")

args = parser.parse_args()



if args.model.split("-")[-1] == "noBN":
    args.batch_norm = False
else:
    args.batch_norm = True


# Set Device and Seed--------------------------------------------
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {args.device}")

utils.set_seed(args.seed)
#----------------------------------------------------------------



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


model = utils.get_backbone(args.model, num_classes, args.device)

model.eval()

criterion = torch.nn.CrossEntropyLoss()


bma_load_paths = sorted(os.listdir(args.swag_load_path))


acc_list = []; ece_list = []; nll_list = []
for cnt, path in enumerate(bma_load_paths):

    # get sampled model
    bma_sample = torch.load(f"{args.swag_load_path}/{path}")
    bma_state_dict = utils.list_to_state_dict(model, bma_sample)
    model.load_state_dict(bma_state_dict)
    
    if args.batch_norm:
        swag_utils.bn_update(tr_loader, model)
    
    res = utils.eval(te_loader, model, criterion, args.device, args.num_bins, args.eps)
    
    print(f"Accuracy : {res['accuracy']:8.4f}% / ECE :{res['ece']} / NLL : {res['nll']}")
    acc_list.append(res['accuracy']); ece_list.append(res['ece']); nll_list.append(res['nll'])
    performance = dict({"accuracy": acc_list, "ece" : ece_list, "nll" : nll_list})
    with open(f'{args.swag_load_path}/performance.pickle', 'wb') as f:
        pickle.dump(performance, f)

