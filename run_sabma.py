import argparse, os, sys, time, copy, tabulate

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import pickle, wandb

import utils.utils as utils
from utils import temperature_scaling as ts
from utils.swag import swag, swag_utils
from utils.sabma import sabma, sabma_utils
import utils.data.data as data

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Run sabma")

parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")

parser.add_argument("--method", type=str, default="sabma",
                    choices=["sabma"],
                    help="Learning Method")

parser.add_argument("--no_amp", action="store_true", default=False, help="Deactivate AMP")

parser.add_argument("--print_epoch", type=int, default=10, help="Printing epoch")

parser.add_argument("--resume", type=str, default=None,
    help="path to load saved model to resume training (default: None)",)


parser.add_argument("--tr_layer", type=str, default="nl_ll", choices=["last_layer", "full_layer", "last_block", "nl_ll"],
            help="Choose layer which would be trained with our method (Default : nl_ll)")

parser.add_argument("--tol", type=int, default=50,
        help="tolerance for early stopping (Default : 50)")

parser.add_argument("--ignore_wandb", action="store_true", default=False, help="Deactivate wandb")

## Data ---------------------------------------------------------
parser.add_argument(
    "--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", 'imagenet'],
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

parser.add_argument("--use_validation", action='store_true',
            help ="Use validation for hyperparameter search (Default : False)")

parser.add_argument("--dat_per_cls", type=int, default=-1,
            help="Number of data points per class in few-shot setting. -1 denotes deactivate few-shot setting (Default : -1)")
#----------------------------------------------------------------

## Model ---------------------------------------------------------
parser.add_argument(
    "--model",
    type=str, default='resnet18', required=True,
    choices=['resnet18', 'resnet50', 'resnet101',
            'resnet50-clip', 'resnet101-clip', 'vitb16-clip',
            'resnet18-noBN', "vitb16-i21k"],
    help="model name (default : resnet18)")

parser.add_argument(
    "--pre_trained", action='store_true', default=True,
    help="Using pre-trained model from zoo"
    )

parser.add_argument("--model_path",
            type=str,
            help="Path to load state dict of backbone (to get bn statistics)")

parser.add_argument("--save_path",
            type=str, default=None,
            help="Path to save best model dict")
#----------------------------------------------------------------

## Optimizer Hyperparameter --------------------------------------
parser.add_argument("--optim", type=str, default="sabma",
                    choices=["sgd", "sam", "sabma"],
                    help="Optimization options")

parser.add_argument("--lr_init", type=float, default=0.01,
                help="learning rate (Default : 0.01)")

parser.add_argument("--momentum", type=float, default=0.9,
                help="momentum (Default : 0.9)")

parser.add_argument("--epochs", type=int, default=100, metavar="N",
    help="number epochs to train (default : 100)")

parser.add_argument("--wd", type=float, default=5e-4, help="weight decay (default: 5e-4)")

parser.add_argument("--rho", type=float, default=0.05, help="size of pertubation ball for SAM / SABMA")

parser.add_argument("--kl_eta", type=float, default=0.0,
                help="Hyperparameter for KLD loss")

parser.add_argument("--scheduler", type=str, default='cos_decay', choices=['constant', "step_lr",  "swag_lr", "cos_decay"])

parser.add_argument("--lr_min", type=float, default=1e-8,
                help="Min learning rate. (Cosine Annealing Warmup Restarts)")

parser.add_argument("--warmup_t", type=int, default=10,
                help="Linear warmup step size. (Cosine Annealing Warmup Restarts)")

parser.add_argument("--warmup_lr_init", type=float, default=1e-7,
                help="Linear warmup initial learning rate (Cosine Annealing Warmup Restarts)")
#----------------------------------------------------------------


## sabma ---------------------------------------------------------
parser.add_argument("--src_bnn", type=str, default="swag", choices=["swag", "la", "vi"],
        help="Type of pre-trained BNN model")

parser.add_argument("--pretrained_set", type=str, default='source', choices=['source', 'down'],
        help="Trained set to make prior (Default: source)")

parser.add_argument("--diag_only", action="store_true", default=False, help="Consider only diagonal variance")

parser.add_argument("--low_rank", type=int, default=-1, help="Low-rank component")

parser.add_argument("--var_scale", type=float, default=1, help="Scaling prior variance")

parser.add_argument("--cov_scale", type=float, default=1, help="Scaling prior covariance")

parser.add_argument("--prior_path", type=str, required=True, default=None,
    help="path to load saved swag model for transfer learning (default: None)")

parser.add_argument("--alpha", type=float, default=1e-2, help="Scale of variance initialized with classifier")
#----------------------------------------------------------------

## bma or metrics -----------------------------------------------
parser.add_argument("--val_mc_num", type=int, default=3, help="Number of models for Mc integration in validation phase")
parser.add_argument("--eps", type=float, default=1e-8, help="small float to calculate nll")
parser.add_argument("--bma_num_models", type=int, default=30, help="Number of models for bma in test phase")
parser.add_argument("--num_bins", type=int, default=15, help="bin number for ece")
parser.add_argument("--no_save_bma", action='store_true', default=False,
            help="Deactivate saving model samples in BMA")
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
args.save_path = utils.set_save_path(args)
print(f"Save Results on {args.save_path}")
print("-"*30)
#------------------------------------------------------------------

# wandb config---------------------------------------------------
if not args.ignore_wandb:
    wandb.init(project="SA-BTL", entity=None)
    wandb.config.update(args)
    wandb.run.name = utils.set_wandb_runname(args)
#----------------------------------------------------------------

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


# Define Model------------------------------------------------------
model = utils.get_backbone(args.model, num_classes, args.device, args.pre_trained)

w_mean = torch.load(f"{args.prior_path}/{args.model}_mean.pt")
w_var = torch.load(f"{args.prior_path}/{args.model}_variance.pt")
if not args.diag_only:
    w_covmat = torch.load(f"{args.prior_path}/{args.model}_covmat.pt")
else:
    w_covmat=None
sabma_model = sabma.SABMA(copy.deepcopy(model),
                        src_bnn=args.src_bnn,
                        w_mean = w_mean,
                        diag_only=args.diag_only,
                        w_var=w_var,
                        var_scale=args.var_scale,
                        low_rank=args.low_rank,
                        w_cov_sqrt=w_covmat,
                        cov_scale=args.cov_scale,
                        tr_layer=args.tr_layer,
                        pretrained_set = 'source',
                        alpha=args.alpha
                        ).to(args.device)
if not args.ignore_wandb:
    wandb.config.update({"low_rank_true" : sabma_model.low_rank})

print(f"Load sabma Model with prior made of {args.src_bnn} with rank {sabma_model.low_rank}")
if not args.diag_only:
    tab_name = ["# of Mean Trainable Params", "# of Var Trainable Params", "# of Cov Trainable Params"]
    tab_contents= [sabma_model.bnn_param.mean.numel(), sabma_model.bnn_param.log_std.numel(), sabma_model.bnn_param.cov_sqrt.numel()]
else:
    tab_name = ["# of Mean Trainable Params", "# of Var Trainable Params"]
    tab_contents= [sabma_model.bnn_param.mean.numel(), sabma_model.bnn_param.log_std.numel()]
table = [tab_name, tab_contents]
print(tabulate.tabulate(table, tablefmt="simple"))
print("-"*30)
#----------------------------------------------------------------

# Set Criterion------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()
print("Set Criterion as Cross Entropy")
print("-"*30)
#-------------------------------------------------------------------



# Set Optimizer--------------------------------------
## Optimizer
optimizer = sabma_utils.get_optimizer(args, sabma_model)
print(f"Set {args.optim} optimizer with lr_init {args.lr_init} / wd {args.wd} / momentum {args.momentum}")
print("-"*30)
#----------------------------------------------------------------

## Set Scheduler-------------------------------------------------------
if args.scheduler not in ["constant", "swag_lr"]:
    scheduler = utils.get_scheduler(args, optimizer)
print(f"Set {args.scheduler}")
print("-"*30)
#-------------------------------------------------------------------

## Resume ---------------------------------------------------------------------------
start_epoch = 1
#------------------------------------------------------------------------------------

## Set AMP --------------------------------------------------------------------------
scaler, first_step_scaler, second_step_scaler = utils.get_scaler(args)
print("-"*30)
#------------------------------------------------------------------------------------

## Training -------------------------------------------------------------------------
print(f"Start training sabma with {args.optim} optimizer from {start_epoch} epoch!")


## print setting
columns = ["epoch", "method", "lr",
        "tr_loss", "tr_acc",
        f"val_loss(MC{args.val_mc_num})", f"val_acc(MC{args.val_mc_num})", f"val_nll(MC{args.val_mc_num})", f"val_ece(MC{args.val_mc_num})",
        "time"]

best_val_loss=9999 ; best_val_acc=0 ; best_epoch=0 ; duration=0; cnt=0
print("Start Training!!")
for epoch in range(start_epoch, int(args.epochs)+1):
    time_ep = time.time()

    ## lr scheduling
    if args.scheduler == "swag_lr":
        lr = swag_utils.schedule(epoch, args.lr_init, args.epochs, False, None, None)
        swag_utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = optimizer.param_groups[0]['lr']

    ## train
    if args.optim == "sgd":
        tr_res = sabma_utils.train_sabma_sgd(tr_loader, sabma_model, criterion, optimizer, args.device, scaler)
    elif args.optim == "sam":
        tr_res = sabma_utils.train_sabma_sam(tr_loader, sabma_model, criterion, optimizer, args.device, first_step_scaler, second_step_scaler)
    elif args.optim == "sabma":
        tr_res = sabma_utils.train_sabma_sabma(tr_loader, sabma_model, criterion, optimizer, args.device, first_step_scaler, second_step_scaler, args.kl_eta)

    # validation / test
    if args.val_mc_num ==1:
        tr_params, _, _ = sabma_model.sample(z_scale=1.0, sample_param='tr')
        frz_params, _, _ = sabma_model.sample(z_scale=1.0, sample_param='frz')
        params = sabma_utils.format_weights(tr_params, frz_params, sabma_model)
        val_res = sabma_utils.eval_sabma(val_loader, sabma_model, params, criterion, args.device, args.num_bins, args.eps)
    else:
        val_res = sabma_utils.bma_sabma(val_loader, sabma_model, args.val_mc_num,
                            num_classes, criterion, args.device,
                            bma_save_path=None, eps=1e-8, num_bins=50,
                            validation=True, tr_layer=args.tr_layer, ood_loader=None
                            )

    time_ep = time.time() - time_ep
    values = [epoch, f"sabma-{args.optim}", lr, tr_res["loss"], tr_res["accuracy"],
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
        wandb.log({
            "Train Loss ": tr_res["loss"], "Train Accuracy" : tr_res["accuracy"],
            f"Validation loss (MC{args.val_mc_num})" : val_res["loss"], f"Validation Accuracy (MC{args.val_mc_num})" : val_res["accuracy"],
            f"Validation nll (MC{args.val_mc_num})" : val_res["nll"], f"Validation ece (MC{args.val_mc_num})" : val_res["ece"],
            "lr" : lr,
            "max(mean)" : torch.max(sabma_model.bnn_param['mean']),
            "mean(mean)" : torch.mean(sabma_model.bnn_param['mean']),
            "std(mean)" : torch.std(sabma_model.bnn_param['mean']),
            "min(mean)" : torch.min(sabma_model.bnn_param['mean']),
            "max(std)" : torch.max(torch.exp(sabma_model.bnn_param['log_std'])),
            "mean(std)" : torch.mean(torch.exp(sabma_model.bnn_param['log_std'])),
            "std(std)" : torch.std(torch.exp(sabma_model.bnn_param['log_std'])),
            "min(std)" : torch.min(torch.exp(sabma_model.bnn_param['log_std'])),
            },
            step=epoch)
        if not args.diag_only:
            wandb.log({
                "max(cov_sqrt)" : torch.max(sabma_model.bnn_param['cov_sqrt']),
                "mean(cov_sqrt)" : torch.mean(sabma_model.bnn_param['cov_sqrt']),
                "std(cov_sqrt)" : torch.std(sabma_model.bnn_param['cov_sqrt']),
                "min(cov_sqrt)" : torch.min(sabma_model.bnn_param['cov_sqrt']),},
                step=epoch)

    # Save best model (Early Stopping)
    if (val_res['loss'] < best_val_loss):
        cnt = 0
        best_val_loss = val_res['loss']
        best_val_acc = val_res['accuracy']
        best_epoch = epoch
        
        # save state_dict
        os.makedirs(args.save_path, exist_ok=True)
        sabma_utils.save_best_sabma_model(args, best_epoch, sabma_model, optimizer, scaler, first_step_scaler, second_step_scaler)
    else:
        cnt += 1

    ## Scheduler step
    if args.scheduler in ["step_lr", "cos_decay"]:
        scheduler.step(epoch)

    ## Early Stopping
    if cnt == args.tol and args.method:
        break
#------------------------------------------------------------------------------------------------------------


## Test ------------------------------------------------------------------------------------------------------
##### Get test nll, Entropy, ece, Reliability Diagram on best model
## Load Distributional shifted data
# Load Best Model
print("Load Best Validation Model (Lowest Loss)")
state_dict_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt"
checkpoint = torch.load(state_dict_path)   
sabma_model.load_state_dict(checkpoint["state_dict"])
sabma_model.to(args.device)


### Get temperature
val_res = sabma_utils.bma_sabma(val_loader, sabma_model, 1,
                    num_classes, criterion, args.device,
                    bma_save_path=None, eps=args.eps, num_bins=args.num_bins,
                    validation=False, tr_layer=args.tr_layer, ood_loader=None)
scaled_model = ts.ModelWithTemperature(sabma_model, ens=True)
scaled_model.set_temperature(val_loader, ens_logits=torch.tensor(val_res['logits']), ens_pred=torch.tensor(val_res['targets']))
bma_temperature = scaled_model.temperature



### BMA prediction
if args.no_save_bma:
    bma_save_path  = None
else:
    bma_save_path = f"{args.save_path}/bma_models"
    os.makedirs(bma_save_path, exist_ok=True)

bma_res = sabma_utils.bma_sabma(te_loader, sabma_model, args.bma_num_models,
                    num_classes, criterion, args.device,
                    bma_save_path=bma_save_path, eps=args.eps, num_bins=args.num_bins,
                    validation=False, tr_layer=args.tr_layer, ood_loader=None)
bma_logits = bma_res["logits"]
bma_predictions = bma_res["predictions"]
bma_targets = bma_res["targets"]


bma_accuracy = bma_res["accuracy"]
bma_nll = bma_res["nll"]
unc = utils.calibration_curve(bma_predictions, bma_targets, args.num_bins)
bma_ece = bma_res['ece']

# Temperature Scaling
bma_logits = torch.tensor(bma_logits) / bma_temperature.cpu()
bma_predictions_ts = F.softmax(bma_logits, dim=1).detach().numpy()
bma_accuracy_ts = np.mean(np.argmax(bma_predictions_ts, axis=1) == bma_targets) * 100
bma_nll_ts = -np.mean(np.log(bma_predictions_ts[np.arange(bma_predictions_ts.shape[0]), bma_targets] + args.eps))
bma_unc_ts = utils.calibration_curve(bma_predictions_ts, bma_targets, args.num_bins)
bma_ece_ts = bma_unc_ts['ece']

    
if not args.ignore_wandb:
    wandb.run.summary['bma accuracy'] = bma_accuracy
    wandb.run.summary['bma nll'] = bma_nll
    wandb.run.summary['bma ece'] = bma_ece
    
    wandb.run.summary['bma accuracy w/ ts'] = bma_accuracy_ts
    wandb.run.summary['bma nll w/ ts'] = bma_nll_ts
    wandb.run.summary['bma ece w/ ts'] = bma_ece_ts
    wandb.run.summary['bma temperature'] = bma_temperature.item()

print("[BMA w/o TS Results]\n")
tab_name = ["# of Models", "BMA Accuracy", "BMA NLL", "BMA ECE"]
tab_contents = [args.bma_num_models, format(bma_accuracy, '.2f'), format(bma_nll, '.4f'), format(bma_ece, '.4f')]
table = [tab_name, tab_contents]
print(tabulate.tabulate(table, tablefmt="simple"))
print("-"*30)

print("[BMA w/ TS Results]\n")
tab_name = ["# of Models", "BMA Accuracy", "BMA NLL", "BMA ECE", "BMA Temperature"]
tab_contents = [args.bma_num_models, format(bma_accuracy_ts, '.2f'), format(bma_nll_ts, '.4f'), format(bma_ece_ts, '.4f'), format(bma_temperature.item(), '.4f')]
table = [tab_name, tab_contents]
print(tabulate.tabulate(table, tablefmt="simple"))
print("-"*30)

# Save ece for reliability diagram
os.makedirs(f'{args.save_path}/unc_result', exist_ok=True)
with open(f"{args.save_path}/unc_result/{args.method}-{args.optim}_bma_wo_ts_uncertainty.pkl", 'wb') as f:
    pickle.dump(unc, f)
    
with open(f"{args.save_path}/unc_result/{args.method}-{args.optim}_bma_w_ts_uncertainty.pkl", 'wb') as f:
    pickle.dump(bma_unc_ts, f)

# Save Reliability Diagram 
utils.save_reliability_diagram(args.method, args.optim, args.save_path, unc, True)




### MAP Prediction
tr_params, _, _ = sabma_model.sample(0, sample_param='tr')
frz_params, _, _ = sabma_model.sample(0, sample_param='frz')
params = sabma_utils.format_weights(tr_params, frz_params, sabma_model)

res = sabma_utils.eval_sabma(te_loader, sabma_model, params, criterion, args.device, args.num_bins, args.eps)
unc = utils.calibration_curve(res['predictions'], res['targets'], args.num_bins)
te_ece = unc["ece"]

if not args.ignore_wandb:
    wandb.run.summary['Best epoch'] = checkpoint["epoch"]
    # Acc
    wandb.run.summary['test accuracy'] = res['accuracy']
    # nll
    wandb.run.summary['test nll'] = res['nll']
    wandb.run.summary["test ece"]  = te_ece

print("[Best MAP Results]\n")
tab_name = ["Best Epoch", "MAP Accuracy", "MAP NLL", "MAP ECE"]
tab_contents= [checkpoint['epoch'], format(res['accuracy'], '.2f'), format(res['nll'], '.4f'), format(te_ece, '.4f')]
table = [tab_name, tab_contents]
print(tabulate.tabulate(table, tablefmt="simple"))
print("-"*30)


# Save ece for reliability diagram
os.makedirs(f'{args.save_path}/unc_result', exist_ok=True)
with open(f"{args.save_path}/unc_result/{args.method}-{args.optim}_uncertainty.pkl", 'wb') as f:
    pickle.dump(unc, f)

# Save Reliability Diagram 
utils.save_reliability_diagram(args.method, args.optim, args.save_path, unc, False)