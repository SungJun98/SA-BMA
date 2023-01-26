# %%
import argparse
import torch
import pickle
import os, copy
import data, utils
import numpy as np

from hessian_eigenthings import compute_hessian_eigenthings
from pyhessian import hessian

from baselines.swag import swag, swag_utils
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Get Hessian of saved model")

parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
# %%
## Data ---------------------------------------------------------
parser.add_argument(
    "--dataset", type=str, default="cifar10", choices=["mnist-source", "mnist-down", "cifar10", "cifar100"],
                    help="dataset name")

parser.add_argument(
    "--data_path",
    type=str,
    default=None,
    help="path to datasets location (default: None)",)

parser.add_argument("--batch_size", type=int, default = 128,
            help="batch size (default : 128)")

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

parser.add_argument("--load_path", type=str, default=None, required=True,
    help="path to load saved model (default: None)")

parser.add_argument(
    "--swag",
    action='store_true',
    help ="When model trained with swag (Default : False)"
)
#----------------------------------------------------------------


## SWAG ---------------------------------------------------------
parser.add_argument("--diag_only", action="store_true", default=False, help="Calculate only diagonal covariance")
parser.add_argument("--max_num_models", type=int, default=20, help="Number of models to get SWAG statistics")
#----------------------------------------------------------------


## Arguments for hessian approximate ------------------------------
parser.add_argument("--num_eigen", type=int, default=5,
    help="number of eigenvalues to get (default : 5)")

parser.add_argument("--max_possible_gpu_samples", type=int,
    default=2048, help="number of max possible samples on gpu (default : 2048")
#----------------------------------------------------------------

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
    if args.data_path is not None: 
        tr_loader, _, te_loader, num_classes = data.get_cifar10(args.data_path, args.batch_size,
                                                                    args.num_workers,
                                                                    use_validation = args.use_validation)
    else:
        tr_loader, _, te_loader, num_classes = data.get_cifar10(batch_size = args.batch_size,
                                                                    num_workers = args.num_workers,
                                                                    use_validation = args.use_validation)

elif args.dataset == 'mnist-source':
    if args.data_path is not None:
        tr_loader, _, te_loader, num_classes = data.get_mnist_source(args.data_path, args.batch_size,
                                                                        args.num_workers,
                                                                        use_validation = args.use_validation,
                                                                        )
    else:
        tr_loader, _, te_loader, num_classes = data.get_mnist_source(batch_size = args.batch_size,
                                                                        num_workers = args.num_workers,
                                                                        use_validation = args.use_validation,
                                                                        )

elif args.dataset == 'mnist-down':
    if args.data_path is not None:
        _, _, source_te_loader, _ = data.get_mnist_source(args.data_path,
                                                        args.batch_size,
                                                        args.num_workers,
                                                        use_validation = args.use_validation,
                                                        )
        tr_loader, _, te_loader, num_classes = data.get_mnist_down(args.data_path,
                                                                        args.batch_size,
                                                                        args.num_workers,
                                                                        use_validation = args.use_validation,
                                                                        )
    else:
        _, _, source_te_loader, _ = data.get_mnist_source(batch_size = args.batch_size,
                                                        num_workers = args.num_workers,
                                                        use_validation = args.use_validation,
                                                        )
        tr_loader, _, te_loader, num_classes = data.get_mnist_down(
                                                                    batch_size = args.batch_size,
                                                                    num_workers = args.num_workers,
                                                                    use_validation = args.use_validation,
                                                                    )
if not args.use_validation:
    val_loader = te_loader

print(f"Load Data : {args.dataset}")
#----------------------------------------------------------------




## Define Model------------------------------------------------------
if args.model == "mlp":
    from models import mlp
    model = mlp.MLP(output_size=num_classes).to(args.device)
elif args.model in ["resnet18", "resnet18-noBN"]:
    from torchvision.models import resnet18
    model = resnet18(pretrained=False, num_classes=num_classes).to(args.device)
elif args.model in ["resnet50", "resnet50-noBN"]:
    from torchvision.models import resnet50
    model = resnet50(pretrained=False, num_classes=num_classes).to(args.device)
elif args.model in ["wideresnet40x10", "wideresnet40x10-noBN"]:
    from models import wide_resnet
    model_cfg = getattr(wide_resnet, "WideResNet40x10")
    model = model_cfg.base(num_classes=num_classes).to(args.device)
elif args.model in ["wideresnet28x10", "wideresnet28x10-noBN"]:
    from models import wide_resnet
    model_cfg = getattr(wide_resnet, "WideResNet28x10")
    model = model_cfg.base(num_classes=num_classes).to(args.device)

print(f"Preparing model {args.model}")


if args.swag:
    # Load SWAG weight 
    swag_model = swag.SWAG(copy.deepcopy(model), no_cov_mat=args.diag_only, max_num_models=args.max_num_models).to(args.device)

    # Get bma weights list
    bma_load_paths = os.listdir(args.swag_load_path)
else:
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint["state_dict"])


model.eval()


criterion = torch.nn.CrossEntropyLoss()


if args.swag:
    tr_cum_eigenval_list = list() ; tr_max_eigenval_list = list()
    te_cum_eigenval_list = list() ; te_max_eigenval_list = list()
    for cnt, path in enumerate(bma_load_paths):
        model.load_state_dict(torch.load(args.load_path))
        model_state_dict = model.state_dict()

        # get sampled model
        bma_sample = torch.load(f"{args.swag_load_path}/{path}")
        bma_state_dict = utils.list_to_state_dict(model, bma_sample)

        model_state_dict.update(bma_state_dict)
        model.load_state_dict(model_state_dict)
        if args.batch_norm:
            swag_utils.bn_update(tr_loader, model)

        res = utils.eval(te_loader, model, criterion, args.device)
        print(f"Test Accuracy : {res['accuracy']:8.4f}%")
        
        
        # get eigenvalue for train set
        try:
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

            # tr_cum_eigenval = tr_cum_eigenval + tr_eigenvals
            # tr_max_eigenval = tr_max_eigenval + max(tr_eigenvals)
            tr_cum_eigenval_list.append(tr_eigenvals)
            tr_max_eigenval_list.append(max(tr_eigenvals))
            print(f"Successfully get {cnt}-th swag bma model eigenvalues for train set")
            print(f"Train Eigenvalues for {cnt}-th bma model : {tr_eigenvals}")

        except:
            print(f"Numerical Issue on {cnt}-th model with train data")

        print("-"*15)

        # get eigenvalue for test set
        try: 
            te_eigenvals, _ = compute_hessian_eigenthings(
                    model,
                    te_loader,
                    criterion,
                    num_eigenthings=args.num_eigen,
                    mode="lanczos",
                    # power_iter_steps=args.num_steps,
                    max_possible_gpu_samples=args.max_possible_gpu_samples,
                    # momentum=args.momentum,
                    use_gpu=True,
                )
            
            # te_cum_eigenval = te_cum_eigenval + te_eigenvals
            # te_max_eigenval = te_max_eigenval + max(te_eigenvals)
            te_cum_eigenval_list.append(te_eigenvals)
            te_max_eigenval_list.append(max(te_eigenvals))
            print(f"Successfully get {cnt}-th swag bma model eigenvalues for test set")
            print(f"Test Eigenvalues for {cnt}-th bma model : {te_eigenvals}")
        except:
            print(f"Numerical Issue on {cnt}-th model with test data")
            
        print("-"*15)
        print("-"*15)
    

    ## Save pickle file
    with open(f'{args.swag_load_path}/tr_eigenval_list.pickle', 'wb') as f:
        pickle.dump(tr_cum_eigenval_list, f)

    with open(f'{args.swag_load_path}/te_eigenval_list.pickle', 'wb') as f:
        pickle.dump(te_cum_eigenval_list, f)


else:
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
    print("-"*15)

    # get eigenvalue for test set
    te_eigenvals, _ = compute_hessian_eigenthings(
                model,
                te_loader,
                criterion,
                num_eigenthings=args.num_eigen,
                mode="lanczos",
                # power_iter_steps=args.num_steps,
                max_possible_gpu_samples=args.max_possible_gpu_samples,
                # momentum=args.momentum,
                use_gpu=True,
            )
    
    print(f"Test Eigenvalues : {te_eigenvals}")
    print(f"Max Test Eigenvalue : {max(te_eigenvals)}")
    print("-"*15)
    print("-"*15)

    ## Save pickle file
    with open(f'{args.load_path}/tr_eigenval_list.pickle', 'wb') as f:
        pickle.dump(tr_eigenvals, f)

    with open(f'{args.load_path}/te_eigenval_list.pickle', 'wb') as f:
        pickle.dump(te_eigenvals, f)
# -----------------------------------------------------------------------------

'''
### pyhessian
# train set
for inputs, targets in tr_loader:
    break
inputs, targets = inputs.to(args.device), targets.to(args.device)

# create the hessian computation module
hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=True)

tr_top_eigenvalues, tr_top_eigenvector = hessian_comp.eigenvalues()
print("The top Hessian eigenvalue of this model on train set is %.4f" %tr_top_eigenvalues[-1])


# test set
for inputs, targets in te_loader:
    break
inputs, targets = inputs.to(args.device), targets.to(args.device)

# create the hessian computation module
hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=True)

te_top_eigenvalues, te_top_eigenvector = hessian_comp.eigenvalues()
print("The top Hessian eigenvalue of this model on test set is %.4f" %te_top_eigenvalues[-1])
'''
