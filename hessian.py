import argparse
import torch
import pickle
import os, copy
import data, utils
import numpy as np

from hessian_eigenthings import compute_hessian_eigenthings

from baselines.swag import swag, swag_utils
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

parser.add_argument("--batch_size", type=int, default = 64,
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

parser.add_argument("--load_path", type=str, default=None,
    help="path to load saved model (default: None)")

parser.add_argument(
    "--swag",
    action='store_true',
    help ="When model trained with swag (Default : False)"
)

parser.add_argument(
    "--last_layer",
    action='store_true',
    help ="Calculate the hessian of last layer only"
)
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
tr_loader, val_loader, te_loader, num_classes = utils.get_dataset(args.dataset,
                                                            args.data_path,
                                                            args.batch_size,
                                                            args.num_workers,
                                                            args.use_validation)

print(f"Load Data : {args.dataset}")
#----------------------------------------------------------------

## Define Model------------------------------------------------------
model = utils.get_backbone(args.model, num_classes, args.device)

if args.swag:
    # Load SWAG weight 
    # swag_model = swag.SWAG(copy.deepcopy(model), no_cov_mat=args.diag_only, max_num_models=args.max_num_models).to(args.device)
    if args.load_path is not None:
        checkpoint = torch.load(args.load_path)    
        model.load_state_dict(checkpoint)
        
    # Get bma weights list
    bma_load_paths = sorted(os.listdir(args.swag_load_path))
else:
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint["state_dict"])


model.eval()

criterion = torch.nn.CrossEntropyLoss()

if args.swag:
    acc_list = list(); ece_list = list(); nll_list = list()
    tr_cum_eigenval_list = list() ; tr_max_eigenval_list = list()
    # te_cum_eigenval_list = list() ; te_max_eigenval_list = list()
    for cnt, path in enumerate(bma_load_paths):

        # get sampled model
        bma_sample = torch.load(f"{args.swag_load_path}/{path}")
        bma_state_dict = utils.list_to_state_dict(model, bma_sample, last=args.last_layer)
        model.load_state_dict(bma_state_dict, strict=False)
        
        if args.batch_norm:
          swag_utils.bn_update(tr_loader, model)
        
        res = utils.eval(te_loader, model, criterion, args.device)
        print(f"Test Accuracy : {res['accuracy']:8.4f}% / ECE : {res['ece']} / NLL : {res['nll']}")
        acc_list.append(res['accuracy']); ece_list.append(res['ece']); nll_list.append(res['nll'])
        performance = dict({"accuracy": acc_list, "ece" : ece_list, "nll" : nll_list})
        with open(f'{args.swag_load_path}/performance.pickle', 'wb') as f:
            pickle.dump(performance, f)
        
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
            print(f"Successfully get {cnt}-th swag bma model eigenvalues for train set")
            print(f"Train Eigenvalues for {cnt}-th bma model : {tr_eigenvals}")

        except:
           print(f"Numerical Issue on {cnt}-th model with train data")
        
        ## save tr_eign as pickle
        with open(f'{args.swag_load_path}/tr_eigenval_list.pickle', 'wb') as f:
            pickle.dump(tr_cum_eigenval_list, f)

        print("-"*15)

        # # get eigenvalue for test set
        # try: 
        #     te_eigenvals, _ = compute_hessian_eigenthings(
        #             model,
        #             te_loader,
        #             criterion,
        #             num_eigenthings=args.num_eigen,
        #             mode="power_iter", #"lanczos",
        #             power_iter_steps=50,
        #             max_possible_gpu_samples=args.max_possible_gpu_samples,
        #             # momentum=args.momentum,
        #             use_gpu=True,
        #         )
            
        #     te_cum_eigenval_list.append(te_eigenvals)
        #     te_max_eigenval_list.append(max(te_eigenvals))
        #     print(f"Successfully get {cnt}-th swag bma model eigenvalues for test set")
        #     print(f"Test Eigenvalues for {cnt}-th bma model : {te_eigenvals}")
        # except:
        #     print(f"Numerical Issue on {cnt}-th model with test data")
            
        # print("-"*15)
        # print("-"*15)
    
        # ## save te_eign as pickle
        # with open(f'{args.swag_load_path}/te_eigenval_list.pickle', 'wb') as f:
        #     pickle.dump(te_cum_eigenval_list, f)


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

    # # get eigenvalue for test set
    # te_eigenvals, _ = compute_hessian_eigenthings(
    #             model,
    #             te_loader,
    #             criterion,
    #             num_eigenthings=args.num_eigen,
    #             mode="lanczos",
    #             # power_iter_steps=args.num_steps,
    #             max_possible_gpu_samples=args.max_possible_gpu_samples,
    #             # momentum=args.momentum,
    #             use_gpu=True,
    #         )
    
    # print(f"Test Eigenvalues : {te_eigenvals}")
    # print(f"Max Test Eigenvalue : {max(te_eigenvals)}")
    # print("-"*15)
    # print("-"*15)

    if not os.path.isdir(args.load_path):
        save_path = args.load_path.split('/')[:-1]
        save_path = '/'.join(save_path)
    
    ## Save pickle file
    with open(f'{save_path}/tr_eigenval_list.pickle', 'wb') as f:
        pickle.dump(tr_eigenvals, f)

    # with open(f'{save_path}/te_eigenval_list.pickle', 'wb') as f:
    #     pickle.dump(te_eigenvals, f)
# -----------------------------------------------------------------------------