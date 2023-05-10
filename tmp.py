# %%
import os, sys
import time, copy
import tabulate
import pickle

import torch
import torch.nn.functional as F

import utils, data

from baselines.sam.sam import SAM, FSAM
from baselines.swag import swag, swag_utils

import sabtl

import warnings
warnings.filterwarnings('ignore')

# %%
seed = 0
utils.set_seed(seed)

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

tr_loader, val_loader, te_loader, num_classes = data.get_cifar10(data_path = "/data1/lsj9862/data/cifar10/",
                                                            batch_size = 64,
                                                            num_workers = 4,
                                                            use_validation = True)

# from models import resnet_noBN
# model = resnet_noBN.resnet18(num_classes=num_classes).to(device)

import torch.nn as nn
from torchvision.models import resnet18
model = resnet18(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

utils.freeze_fe(model)

# %%
w_mean = torch.load("/data2/lsj9862/exp_result/cifar10/resnet18/last_swag-sgd_swag_lr_300/20_51_1_0.0005/last_swag-sgd_best_val_mean.pt")
w_var = torch.load("/data2/lsj9862/exp_result/cifar10/resnet18/last_swag-sgd_swag_lr_300/20_51_1_0.0005/last_swag-sgd_best_val_variance.pt") 
w_covmat = torch.load("/data2/lsj9862/exp_result/cifar10/resnet18/last_swag-sgd_swag_lr_300/20_51_1_0.0005/last_swag-sgd_best_val_covmat.pt")

# %%
sabtl_model = sabtl.SABTL(copy.deepcopy(model), src_bnn='swag', w_mean = w_mean, w_var=w_var, w_cov_sqrt=w_covmat).to(device)


# %%
criterion = torch.nn.CrossEntropyLoss()

# %%
## BSAM
lr_init = 0.01 ; rho = 0.1

base_optimizer = torch.optim.SGD
optimizer = sabtl.BSAM(sabtl_model.bnn_param.values(), base_optimizer, rho=rho, lr=lr_init, momentum=0.9,
        weight_decay=5e-4)

first_step_scaler = torch.cuda.amp.GradScaler(2 ** 8)
second_step_scaler = torch.cuda.amp.GradScaler(2 ** 8)

# %%
### Full Training
epochs = 100

columns = ["epoch", "method", "lr", "tr_loss", "tr_acc", "val_loss (MAP)", "val_acc (MAP)", "time"]

best_val_loss=9999 ; best_val_acc=0 ; best_epoch=0 ; cnt=0; best_val_loss=9999
print("Start Training!!")
for epoch in range(int(epochs)):
    ## Check parameter scales (Debugging) -------------------------------------------
    print(f"max(mean) : {torch.max(sabtl_model.bnn_param['mean'])}   \
            / min(mean) :{torch.min(sabtl_model.bnn_param['mean'])} \
            / nan(mean) {torch.sum(torch.isnan(sabtl_model.bnn_param['mean']))}") 
    print(f"max(std) : {torch.max(torch.exp(sabtl_model.bnn_param['log_std']))}   \
            / min(std) :{torch.min(torch.exp(sabtl_model.bnn_param['log_std']))} \
            / nan(std) {torch.sum(torch.isnan(torch.exp((utils.softclip(sabtl_model.bnn_param['log_std'])))))}") 
    print(f"max(cov_sqrt) : {torch.max(sabtl_model.bnn_param['cov_sqrt'])}   \
            / min(cov_sqrt) :{torch.min(sabtl_model.bnn_param['cov_sqrt'])} \
            / nan(cov_sqrt) {torch.sum(torch.isnan(sabtl_model.bnn_param['cov_sqrt']))}") 
    # --------------------------------------------------------------------


    time_ep = time.time()
    ### train --------------------------------------------------
    # (나중에 train_sabtl 함수화 예정)
    loss_sum = 0.0
    correct = 0.0
    num_objects_current = 0
    for batch, (X, y) in enumerate(tr_loader):
        X, y = X.to(device), y.to(device)
           
        # Sample weight
        params, z_ = sabtl_model.sample(1.0)
        # compute Fisher inverse
        fish_inv = sabtl_model.fish_inv(params)
        # Change weight sample shape to input model
        params = utils.format_weights(params, sabtl_model)

        # first forward & backward -------------------------------------------
        with torch.cuda.amp.autocast():
            pred = sabtl_model(params, X)
            loss = criterion(pred, y)
        first_step_scaler.scale(loss).backward()
        first_step_scaler.unscale_(optimizer)
        
        optimizer_state = first_step_scaler._per_optimizer_states[id(optimizer)]
        
        inf_grad_cnt = sum(v.item() for v in optimizer_state["found_inf_per_device"].values())      # Check if any gradients are inf/nan
        if inf_grad_cnt == 0:
            # if valid graident, apply sam_first_step
            optimizer.first_step(fish_inv, zero_grad=True)
            sam_first_step_applied = True
        else:
            # if invalid graident, skip sam and revert to single optimization step
            optimizer.zero_grad()
            sam_first_step_applied = False  
        first_step_scaler.update()
        # --------------------------------------------------------------

        ### second forward-backward pass
        params = optimizer.second_sample(z_, sabtl_model, scale=1.0)
        
        with torch.cuda.amp.autocast():
            pred = sabtl_model(params, X)
            loss = criterion(pred, y)
        second_step_scaler.scale(loss).backward()
        
        if sam_first_step_applied:
            optimizer.second_step()  
        second_step_scaler.step(optimizer)
        second_step_scaler.update()

        # ### Checking accuracy with MAP (Mean) solution
        params, _ = sabtl_model.sample(0.0)
        params = utils.format_weights(params, sabtl_model)
        pred = sabtl_model(params, X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss_sum += loss.data.item() * X.size(0)
        num_objects_current += X.size(0)
        
        
    tr_loss = loss_sum / num_objects_current
    tr_acc = correct / num_objects_current * 100
    # -------------------------------------------------------------------


    ## eval --------------------------------------------------------------
    # (eval_sabtl로 옮겨질 함수화 예정)
    params, _ = sabtl_model.sample(0.0)
    params = utils.format_weights(params, sabtl_model)
    
    loss_sum = 0.0
    correct = 0.0
    num_objects_current = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(te_loader):
            X, y = X.to(device), y.to(device)

            ### Checking accuracy with MAP (Mean) solution

            pred = sabtl_model(params, X)
            loss = criterion(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            loss_sum += loss.data.item() * X.size(0)
            num_objects_current += X.size(0)
        
        val_loss = loss_sum / num_objects_current
        val_acc = correct / num_objects_current * 100

        time_ep = time.time() - time_ep
        values = [epoch + 1, f"sabtl-bsam", lr_init,
                tr_loss, tr_acc,
                val_loss, val_acc,
                time_ep]
    
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 10 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)
    # --------------------------------------------------------------------------
    
    
    # if val_loss < best_val_loss: #### 지금 loss scale이...
    #     best_val_loss = val_loss
    #     best_val_acc = val_acc
    #     best_epoch = epoch + 1

    #     # save state_dict
    #     utils.save_checkpoint(file_path = f"./exp_result/bsam_best_val.pt",
    #                     epoch = epoch,
    #                     state_dict = model.state_dict(),
    #                     optimizer = optimizer.state_dict(),
    #                     # scheduler = scheduler.state_dict(),
    #                     first_step_scaler = first_step_scaler.state_dict(),
    #                     second_step_scaler = second_step_scaler.state_dict()
    #                     )


# %%        
'''
## Test ------------------------------------------------------------------------------------------------------
##### Get test nll, Entropy, ece, Reliability Diagram on best model
# Load Best Model
print("Load Best Validation Model (Lowest Loss)")
state_dict_path = "./exp_result/bsam_best_val.pt"
checkpoint = torch.load(state_dict_path)

sabtl_model.load_state_dict(checkpoint["state_dict"])
model.to(args.device)


### BMA prediction
bma_save_path = f"./exp_result/bma_models"
os.makedirs(bma_save_path, exist_ok=True)

bma_res = utils.bma(tr_loader, te_loader, swag_model, args.bma_num_models, num_classes, bma_save_path=bma_save_path, eps=args.eps, batch_norm=args.batch_norm)
bma_predictions = bma_res["predictions"]
bma_targets = bma_res["targets"]

# Acc
bma_accuracy = bma_res["bma_accuracy"] * 100
print(f"bma accuracy : {bma_accuracy:8.4f}")

# nll
bma_nll = bma_res["nll"]
print(f"bma nll : {bma_nll:8.4f}")       

# ece
unc = utils.calibration_curve(bma_predictions, bma_targets, args.num_bins)
bma_ece = unc["ece"]
print(f"bma ece : {bma_ece:8.4f}")
'''

# %%