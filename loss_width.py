# %%
import torch
import torch.nn as nn
import numpy as np
import sklearn.decomposition
import tabulate
import time
import argparse

# %%
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Width along PCA directions")

parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")

parser.add_argument("--method", type=str, default="dnn",
                    choices=["dnn", "swag", "vi", "la", "ptl"],
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
#----------------------------------------------------------------

## Model ---------------------------------------------------------
parser.add_argument(
    "--model",
    type=str, default='resnet18', required=True,
    choices=['resnet18', 'vitb16-i1k', "vitb16-i21k"],
    help="model name (default : resnet18)")

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
                    choices=["sgd", "sam", "fsam", "adam"],
                    help="Optimization options")

parser.add_argument("--lr_init", type=float, default=0.01,
                help="learning rate (Default : 0.01)")

parser.add_argument("--momentum", type=float, default=0.9,
                help="momentum (Default : 0.9)")

parser.add_argument("--epochs", type=int, default=100, metavar="N",
    help="number epochs to train (default : 100)")

parser.add_argument("--wd", type=float, default=5e-4, help="weight decay (default: 5e-4)")

parser.add_argument("--rho", type=float, default=0.05, help="size of pertubation ball for SAM / FSAM")

parser.add_argument("--eta", type=float, default=1.0, help="diagonal fisher inverse weighting term in FSAM")

# Scheduler
parser.add_argument("--scheduler", type=str, default='cos_decay', choices=['constant', "step_lr", "cos_anneal", "swag_lr", "cos_decay"])

parser.add_argument("--lr_min", type=float, default=1e-8,
                help="Min learning rate. (Cosine Annealing Warmup Restarts)")

parser.add_argument("--warmup_t", type=int, default=10,
                help="Linear warmup step size. (Cosine Annealing Warmup Restarts)")

parser.add_argument("--warmup_lr_init", type=float, default=1e-7,
                help="Linear warmup initial learning rate. (Cosine Annealing Warmup Restarts)")
#----------------------------------------------------------------

## SWAG ---------------------------------------------------------
parser.add_argument("--swa_start", type=int, default=161, help="Start epoch of SWAG")
parser.add_argument("--swa_lr", type=float, default=0.05, help="Learning rate for SWAG")
parser.add_argument("--diag_only", action="store_true", default=False, help="Calculate only diagonal covariance")
parser.add_argument("--swa_c_epochs", type=int, default=1, help="Cycle to calculate SWAG statistics")
parser.add_argument("--max_num_models", type=int, default=5, help="Number of models to get SWAG statistics")

parser.add_argument("--swag_resume", type=str, default=None,
    help="path to load saved swag model to resume training (default: None)",)
#----------------------------------------------------------------

## bma or metrics -----------------------------------------------
parser.add_argument("--val_mc_num", type=int, default=1, help="number of MC sample in validation phase")
parser.add_argument("--eps", type=float, default=1e-8, help="small float to calculate nll")
parser.add_argument("--bma_num_models", type=int, default=30, help="Number of models for bma")
parser.add_argument("--num_bins", type=int, default=15, help="bin number for ece")
parser.add_argument("--no_save_bma", action='store_true', default=False,
            help="Deactivate saving model samples in BMA")
#----------------------------------------------------------------

parser.add_argument(
    "--low_rank", type=int, default=5, help="SWAG rank (default: 20)"
)

parser.add_argument("--checkpoint", action="append")
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    required=True,
    help="path to npz results file",
)

parser.add_argument(
    "--dist",
    type=float,
    default=30.0,
    metavar="D",
    help="dist to travel along a direction (default: 30.0)",
)
parser.add_argument(
    "--N",
    type=int,
    default=21,
    metavar="N",
    help="number of points on a grid (default: 21)",
)


args = parser.parse_args()
#----------------------------------------------------------------


args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

utils.set_seed(args.seed)

print(f"Device : {args.device} / Seed : {args.seed}")
print("-"*30)
#------------------------------------------------------------------

# Load Data --------------------------------------------------------
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
                        max_num_models=args.low_rank,
                        last_layer=False).to(args.device)
    print("Preparing SWAG model")

print("-"*30)
#-------------------------------------------------------------------

# Set Criterion-----------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()
print("Set Criterion as Cross Entropy")
print("-"*30)
#-------------------------------------------------------------------


############################
if args.method != 'sabma':
    state_dict_path = f'{args.load_path}/{method}-{args.optim}_best_val.pt'
    checkpoint = torch.load(state_dict_path)
else:
    model = torch.load(f'{args.load_path}/{args.method}-{args.optim}_best_val_model.pt')
checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint["state_dict"])

mean = f"{args.load_path}/~~.pt"
var = f"{args.load_path}/~~.pt"
cov_mat = f"{args.load_path}/~~.pt"
# cov_mat to matrix

tsvd = sklearn.decomposition.TruncatedSVD(n_components=args.swag_rank, n_iter=7)
tsvd.fit(cov_mat)

component_variances = np.dot(
    np.dot(tsvd.components_, cov_mat.T), np.dot(cov_mat, tsvd.components_.T)
) / (cov_mat.shape[0] - 1)

pc_idx = [
    0,
    1,
    2,
    3,
    4,
    args.swag_rank // 2 - 1,
    args.swag_rank // 2,
    args.swag_rank // 2 + 1,
    args.swag_rank - 2,
    args.swag_rank - 1,
]
pc_idx = np.sort(np.unique(np.minimum(pc_idx, args.swag_rank - 1)))
K = len(pc_idx)

ts = np.linspace(-args.dist, args.dist, args.N)

train_acc = np.zeros((K, args.N))
train_loss = np.zeros((K, args.N))
test_acc = np.zeros((K, args.N))
test_loss = np.zeros((K, args.N))

columns = ["PC", "t", "tr_loss", "tr_acc", "te_loss", "te_acc", "time"]

for i, id in enumerate(pc_idx):
    v = tsvd.components_[id, :].copy()
    v /= np.linalg.norm(v)
    for j, t in enumerate(ts):
        start_time = time.time()
        w = mean + t * v

        offset = 0
        for param in model.parameters():
            size = np.prod(param.size())
            param.data.copy_(
                param.new_tensor(w[offset : offset + size].reshape(param.size()))
            )
            offset += size

        utils.bn_update(loaders["train"], model)
        train_res = utils.eval(loaders["train"], model, criterion)
        test_res = utils.eval(loaders["test"], model, criterion)

        train_acc[i, j] = train_res["accuracy"]
        train_loss[i, j] = train_res["loss"]
        test_acc[i, j] = test_res["accuracy"]
        test_loss[i, j] = test_res["loss"]

        run_time = time.time() - start_time
        values = [
            id,
            t,
            train_loss[i, j],
            train_acc[i, j],
            test_loss[i, j],
            test_acc[i, j],
            run_time,
        ]
        table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
        if j == 0:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        print(table)

np.savez(
    args.save_path,
    N=num_checkpoints,
    dim=W.shape[1],
    ts=ts,
    explained_variance=tsvd.explained_variance_,
    explained_variance_ratio=tsvd.explained_variance_ratio_,
    pc_idx=pc_idx,
    train_acc=train_acc,
    train_err=100.0 - train_acc,
    train_loss=train_loss,
    test_acc=test_acc,
    test_err=100.0 - test_acc,
    test_loss=test_loss,
    component_variances=component_variances,
)