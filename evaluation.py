import argparse
import os, sys
import time, copy
import tabulate

import torch
import torch.nn.functional as F

import utils.utils as utils
from utils.swag import swag, swag_utils
from utils.vi import vi_utils
from utils.la import la_utils
from utils.sabma import sabma_utils
from utils import temperature_scaling as ts
import utils.data.data as data

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

def save_to_csv_accumulated(df, filename):
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, index=False)




parser = argparse.ArgumentParser(description="training baselines")

parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")

parser.add_argument("--method", type=str, default="dnn",
                    choices=["dnn", "swag", "ll_swag", "vi", "ll_vi", "la", "ll_la", "sabma", "ptl"],
                    help="Learning Method")

parser.add_argument("--optim", type=str, default="sgd",
                    choices=["sgd", "sgld", "sam", "bsam", "sabma"],
                    help="Learning Method")

parser.add_argument("--load_path", type=str, default=None,
                    help="Path to test")

parser.add_argument("--save_path", type=str, default=None,
                    help="Path to save result")

## Data ---------------------------------------------------------
parser.add_argument(
    "--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"],
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
    type=str, default='resnet18', required=True,
    choices=['resnet18', 'resnet50', 'resnet101',
            'resnet18-noBN', "vitb16-i21k"],
    help="model name (default : resnet18)")

parser.add_argument(
    "--pre_trained", action='store_true', default=True,
    help="Using pre-trained model from zoo"
    )
#----------------------------------------------------------------

## SABMA---------------------------------------------------------
parser.add_argument("--tr_layer", type=str, default="nl_ll",
        help="Traning layer of SABMA")
#----------------------------------------------------------------

## bma or metrics -----------------------------------------------
parser.add_argument("--eps", type=float, default=1e-8, help="small float to calculate nll")
parser.add_argument("--bma_num_models", type=int, default=30, help="Number of models for bma")
parser.add_argument("--num_bins", type=int, default=15, help="bin number for ece")
parser.add_argument("--no_save_bma", action='store_true', default=False,
            help="Deactivate saving model samples in BMA")
#----------------------------------------------------------------

## OOD test -----------------------------------------------------
parser.add_argument("--corrupt_option",
    default=['brightness.npy','contrast.npy','defocus_blur.npy','elastic_transform.npy','fog.npy',
    'frost.npy','gaussian_blur.npy','gaussian_noise.npy','glass_blur.npy','impulse_noise.npy','jpeg_compression.npy',
    'motion_blur.npy','pixelate.npy','saturate.npy','shot_noise.npy','snow.npy','spatter.npy','speckle_noise.npy','zoom_blur.npy'],
    help='corruption option of CIFAR10/100-C'
        )
parser.add_argument("--severity",
    default=1,
    type=int,
    help='Severity of corruptness in CIFAR10/100-C (1 to 5)')
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

if args.no_aug:
    args.aug = False

utils.set_seed(args.seed)

print(f"Device : {args.device} / Seed : {args.seed}")
print("-"*30)
#------------------------------------------------------------------

# Set BMA and Save Setting-----------------------------------------
if args.method == 'dnn':
    args.bma_num_models = 1
args.ignore_wandb = True
#------------------------------------------------------------------


# Load Data --------------------------------------------------------
data_path_ood = args.data_path
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
model = utils.get_backbone(args.model, num_classes, args.device, True)

swag_model=None
if args.method == "swag":
    swag_model = swag.SWAG(copy.deepcopy(model),
                        no_cov_mat=False,
                        max_num_models=5,
                        last_layer=False).to(args.device)
    print("Preparing SWAG model")

print("-"*30)
#-------------------------------------------------------------------

# Set Criterion-----------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()
print("Set Criterion as Cross Entropy")
print("-"*30)
#-------------------------------------------------------------------


method = args.method
if args.method == 'ptl':
    args.method = 'dnn'

## Test ------------------------------------------------------------------------------------------------------
##### Get test nll, Entropy, ece, Reliability Diagram on best model
## Load Distributional shifted data
if args.dataset == 'cifar10':
    ood_loader = data.corrupted_cifar10(data_path=data_path_ood,
                            corrupt_option=args.corrupt_option,
                            severity=args.severity,
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers)
elif args.dataset == 'cifar100':
        ood_loader = data.corrupted_cifar100(data_path=data_path_ood,
                            corrupt_option=args.corrupt_option,
                            severity=args.severity,
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers)

### Load Best Model
print("Load Best Validation Model (Lowest Loss)")
if args.method != 'sabma':
    state_dict_path = f'{args.load_path}/{method}-{args.optim}_best_val.pt'
    checkpoint = torch.load(state_dict_path)
else:
    model = torch.load(f'{args.load_path}/{args.method}-{args.optim}_best_val_model.pt')
    

    
mean = None; variance = None
if args.method in ["swag", "ll_swag"]:
    swag_model.load_state_dict(checkpoint["state_dict"])
    model = swag_model
    
elif args.method in ["vi", "ll_vi"]:
    model = utils.get_backbone(args.model, num_classes, args.device, True)
    if args.method == "ll_vi":
        vi_utils.make_ll_vi(args, model)
    vi_utils.load_vi(model, checkpoint)
    mean = torch.load(f'{args.load_path}/{args.method}-{args.optim}_best_val_mean.pt')
    variance = torch.load(f'{args.load_path}/{args.method}-{args.optim}_best_val_variance.pt')
    
elif args.method == 'dnn':
    if method == 'dnn':
        model.load_state_dict(checkpoint["state_dict"])
    elif method == 'ptl':
        for key in list(checkpoint.keys()):
            if 'backbone.' in key:
                new_key = key.replace('backbone.', '')
                checkpoint[new_key] = checkpoint.pop(key)
            elif 'classifier.' in key:
                new_key = key.replace('classifier', 'fc')
                checkpoint[new_key] = checkpoint.pop(key)
        model.load_state_dict(checkpoint)
    
else:
    pass
model.to(args.device)        



if args.method != 'sabma':
    #### MAP
    ## Unscaled Results
    res = utils.no_ts_map_estimation(args, te_loader, num_classes, model, mean, variance, criterion)
    ood_res = utils.no_ts_map_estimation(args, ood_loader, num_classes, model, mean, variance, criterion)

    print(f"1) Unscaled Results:")
    table = [["Test Accuracy", "Test NLL", "Test Ece"],
            [format(res['accuracy'], '.2f'), format(res['nll'], '.4f'), format(res['ece'], '.4f')]]
    print(tabulate.tabulate(table, tablefmt="simple", floatfmt="8.4f"))
    table = [["OOD Accuracy", "OOD NLL", "OOD ECE"],
            [format(ood_res['accuracy'], '.2f'), format(ood_res['nll'], '.4f'), format(ood_res['ece'], '.4f')]]
    print(tabulate.tabulate(table, tablefmt="simple", floatfmt="8.4f"))

    ## Temperature Scaled Results
    res_ts, temperature = utils.ts_map_estimation(args, val_loader, te_loader, num_classes, model, mean, variance, criterion, save=False)
    print(f"2) Scaled Results:")
    table = [["Test Accuracy", "Test NLL", "Test Ece", "Temperature"],
            [format(res_ts['accuracy'], '.2f'), format(res_ts['nll'],'.4f'), format(res_ts['ece'], '.4f'), format(temperature.item(), '.4f')]]
    print(tabulate.tabulate(table, tablefmt="simple", floatfmt="8.4f"))



    #### Bayesian Model Averaging
    if args.no_save_bma:
        bma_save_path  = None
    else:
        bma_save_path = f"{args.save_path}/bma_models"
        os.makedirs(bma_save_path, exist_ok=True)
    
    if args.method in ["swag", "ll_swag", "vi", "ll_vi"]:
        utils.set_seed(args.seed)
        print(f"Start Bayesian Model Averaging with {args.bma_num_models} samples")
        bma_res, bma_accuracy, bma_nll, bma_ece, bma_accuracy_ts, bma_nll_ts, bma_ece_ts, temperature, bma_ood_accuracy, bma_ood_nll, bma_ood_ece = utils.bma(args, tr_loader, val_loader, te_loader, ood_loader, num_classes, model, mean, variance, criterion, None, temperature)
    else:
        pass



else:
    ### Get temperature
    val_res = sabma_utils.bma_sabma(val_loader, model, 1,
                        num_classes, criterion, args.device,
                        bma_save_path=None, eps=args.eps, num_bins=args.num_bins,
                        validation=True, tr_layer=args.tr_layer)
    scaled_model = ts.ModelWithTemperature(model, ens=True)
    scaled_model.set_temperature(val_loader, ens_logits=torch.tensor(val_res['logits']), ens_pred=torch.tensor(val_res['targets']))
    temperature = scaled_model.temperature


    ### BMA prediction
    if args.no_save_bma:
        bma_save_path  = None
    else:
        bma_save_path = f"{args.save_path}/bma_models"
        os.makedirs(bma_save_path, exist_ok=True)

    ## BMA result w/o Ts
    bma_res = sabma_utils.bma_sabma(te_loader, model, args.bma_num_models,
                        num_classes, criterion, args.device,
                        bma_save_path=bma_save_path, eps=args.eps, num_bins=args.num_bins,
                        validation=False, tr_layer=args.tr_layer, ood_loader=ood_loader)
    bma_logits = bma_res["logits"]
    bma_predictions = bma_res["predictions"]
    bma_targets = bma_res["targets"]

    bma_accuracy = bma_res["accuracy"]
    bma_nll = bma_res["nll"]
    unc = utils.calibration_curve(bma_predictions, bma_targets, args.num_bins)
    bma_ece = bma_res['ece']

    print("[BMA w/o TS Results]\n")
    tab_name = ["# of Models", "BMA Accuracy", "BMA NLL", "BMA ECE"]
    tab_contents = [args.bma_num_models, format(bma_accuracy, '.2f'), format(bma_nll, '.4f'), format(bma_ece, '.4f')]
    table = [tab_name, tab_contents]
    print(tabulate.tabulate(table, tablefmt="simple"))
    print("-"*30)

    ## OOD result
    bma_ood_accuracy = bma_res["ood_accuracy"]
    bma_ood_nll = bma_res["ood_nll"]
    bma_ood_ece = bma_res["ood_ece"]
    
    tab_name = ["# of Models", "OOD BMA Accuracy", "OOD BMA NLL", "OOD BMA ECE"]
    tab_contents = [args.bma_num_models, format(bma_ood_accuracy, '.2f'), format(bma_ood_nll, '.4f'), format(bma_ood_ece, '.4f')]
    table = [tab_name, tab_contents]
    print(tabulate.tabulate(table, tablefmt="simple"))
    print("-"*30)


    ## BMA w/ TS
    bma_logits = torch.tensor(bma_logits) / temperature.cpu()
    bma_predictions_ts = F.softmax(bma_logits, dim=1).detach().numpy()
    bma_accuracy_ts = np.mean(np.argmax(bma_predictions_ts, axis=1) == bma_targets) * 100
    bma_nll_ts = -np.mean(np.log(bma_predictions_ts[np.arange(bma_predictions_ts.shape[0]), bma_targets] + args.eps))
    bma_unc_ts = utils.calibration_curve(bma_predictions_ts, bma_targets, args.num_bins)
    bma_ece_ts = bma_unc_ts['ece']
    temperature = temperature.cpu().item()

    print("[BMA w/ TS Results]\n")
    tab_name = ["# of Models", "BMA Accuracy", "BMA NLL", "BMA ECE", "BMA Temperature"]
    tab_contents = [args.bma_num_models, format(bma_accuracy_ts, '.2f'), format(bma_nll_ts, '.4f'), format(bma_ece_ts, '.4f'), format(temperature, '.4f')]
    table = [tab_name, tab_contents]
    print(tabulate.tabulate(table, tablefmt="simple"))
    print("-"*30)


    ## MAP Prediction
    tr_params, _, _ = model.sample(0, sample_param='tr')
    frz_params, _, _ = model.sample(0, sample_param='frz')
    params = sabma_utils.format_weights(tr_params, frz_params, model)

    res = sabma_utils.eval_sabma(te_loader, model, params, criterion, args.device, args.num_bins, args.eps)
    ood_res = sabma_utils.eval_sabma(ood_loader, model, params, criterion, args.device, args.num_bins, args.eps)
    unc = utils.calibration_curve(res['predictions'], res['targets'], args.num_bins)
    te_ece = unc["ece"]

    print("[Best Test Results]\n")
    tab_name = ["MAP Accuracy", "MAP NLL", "MAP ECE"]
    tab_contents= [format(res['accuracy'], '.2f'), format(res['nll'], '.4f'), format(te_ece, '.4f')]
    table = [tab_name, tab_contents]
    print(tabulate.tabulate(table, tablefmt="simple"))
    
    res_ts = dict()
    res_ts['accuracy'] = None
    res_ts['nll'] = None
    res_ts['ece'] = None

 
if args.corrupt_option == ['brightness.npy','contrast.npy','defocus_blur.npy','elastic_transform.npy','fog.npy',
    'frost.npy','gaussian_blur.npy','gaussian_noise.npy','glass_blur.npy','impulse_noise.npy','jpeg_compression.npy',
    'motion_blur.npy','pixelate.npy','saturate.npy','shot_noise.npy','snow.npy','spatter.npy','speckle_noise.npy','zoom_blur.npy']:
    corr = 'all'
else:
    corr = args.corrupt_option
        

result_df = pd.DataFrame({"method" : [args.method],
                "optim" : [args.optim],
                "seed" : [args.seed],
                "dataset" : [args.dataset],
                "dat_per_cls" : [args.dat_per_cls],
                "corrupt_option" : [corr],
                "severity" : [args.severity],
                "Test Accuracy" : [res['accuracy']],
                "Test NLL" : [res['nll']],
                "Test Ece" : [res['ece']],
                "OOD Accuracy" : [ood_res['accuracy']],
                "OOD NLL" : [ood_res['nll']],
                "OOD ECE" : [ood_res['ece']],
                "Test Accuracy ts" : [res_ts['accuracy']],
                "Test NLL ts" : [res_ts['nll']],
                "Test Ece ts" : [res_ts['ece']],
                })


if method == 'ptl':
    args.method = 'ptl'

if args.method in ["dnn", "ptl"]:
    bma_accuracy = None
    bma_nll = None
    bma_ece = None
    bma_accuracy_ts = None
    bma_nll_ts = None
    bma_ece_ts = None
    bma_ood_accuracy = None
    bma_ood_nll = None
    bma_ood_ece = None

try:
    temperature = temperature.cpu().detach().numpy()
except:
    pass
    
result_df["BMA Accuracy"] = bma_accuracy
result_df["BMA NLL"] = bma_nll
result_df["BMA ECE"] = bma_ece
result_df["BMA Accuracy ts"] = bma_accuracy_ts
result_df["BMA NLL ts"] = bma_nll_ts
result_df["BMA ECE ts"] = bma_ece_ts
result_df["temperature"] = temperature
result_df["BMA OOD Accuracy"] = bma_ood_accuracy
result_df["BMA OOD NLL"] = bma_ood_nll
result_df["BMA OOD ECE"] = bma_ood_ece

save_to_csv_accumulated(result_df, args.save_path)