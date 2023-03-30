# %%
import os, sys
import time, copy
import tabulate
import pickle

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

import utils, data

from baselines.sam.sam import SAM, FSAM
from baselines.swag import swag, swag_utils

import sabtl

import warnings
warnings.filterwarnings('ignore')

# %%
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

tr_loader, val_loader, te_loader, num_classes = data.get_cifar10(data_path = "/DATA1/lsj9862/cifar10/",
                                                            batch_size = 64,
                                                            num_workers = 4,
                                                            use_validation = True)

from models import resnet_noBN
model = resnet_noBN.resnet18(num_classes=num_classes).to(device)

# %%
w_mean = torch.load("/home/lsj9862/BayesianSAM/exp_result/swag-sgd_best_val_mean.pt")

w_var_sqrt = torch.load("/home/lsj9862/BayesianSAM/exp_result/swag-sgd_best_val_variance.pt") 

w_covmat = torch.load("/home/lsj9862/BayesianSAM/exp_result/swag-sgd_best_val_covmat.pt")

# %%
sabtl_model = sabtl.SABTL(copy.deepcopy(model),  w_mean = w_mean,
                        w_var = w_var_sqrt, diag_only=False, w_cov=w_covmat).to(device)


# %%
criterion = torch.nn.CrossEntropyLoss()

# %%
## BSAM
lr_init = 5e-4 ; rho = 0.01

base_optimizer = torch.optim.SGD
optimizer = sabtl.BSAM([sabtl_model.mean_param, sabtl_model.var_param, sabtl_model.cov_param], base_optimizer, sabtl_model, rho=rho, lr=lr_init, momentum=0.9,
        weight_decay=5e-4, nesterov=False)


# %%
'''
### Check Model Load
loss_sum = 0.0
correct = 0.0

num_objects_current = 0

sabtl_model.backbone.train() 
# sabtl_model.apply(utils.deactivate_batchnorm)
# print("Deactivate Batch Norm")
for batch, (X, y) in enumerate(te_loader):
    
    X, y = X.to(device), y.to(device)
    
    # Set weight sample
    params, z_1, z_2 = sabtl_model.sample(scale=0.0)

    ### first forward-backward pass
    pred = torch.nn.utils.stateless.functional_call(sabtl_model.backbone, params, X)
    loss = criterion(pred, y)
    
    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss_sum += loss.data.item() * X.size(0)
    num_objects_current += X.size(0)

print(f"loss : {loss_sum / num_objects_current}")
print(f"accuracy :{correct / num_objects_current * 100}%")
'''

# %%
'''
### 1 epoch test
loss_sum = 0.0
correct = 0.0

num_objects_current = 0

# sabtl_model.apply(utils.deactivate_batchnorm)
# sabtl_model.apply(utils.deactivate_batchnorm_v2)
# print("Deactivate Batch Norm")
for batch, (X, y) in enumerate(tr_loader):
    X, y = X.to(device), y.to(device)
    
    ### first forward-backward pass
    # sample weight from bnn params
    params, z_1, z_2 = sabtl_model.sample(1.0)

    # forward & backward
    pred = torch.nn.utils.stateless.functional_call(sabtl_model.backbone, params, X)
    loss = criterion(pred, y)
    loss.backward()

    bnn_params = optimizer.first_step(zero_grad=True)
    params = sabtl.second_sample(bnn_params, z_1, z_2, sabtl_model, scale=1.0)

    ### second forward-backward pass
    pred = torch.nn.utils.stateless.functional_call(sabtl_model.backbone, params, X)

    criterion(pred, y).backward()
    optimizer.second_step(zero_grad=True)  

    ### Checking accuracy with MAP (Mean) solution
    params, _, _ = sabtl_model.sample(scale=0.0)
    pred = torch.nn.utils.stateless.functional_call(sabtl_model.backbone, params, X)
    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss_sum += loss.data.item() * X.size(0)
    num_objects_current += X.size(0)

print(f"loss : {loss_sum / num_objects_current}")
print(f"accuracy : {correct / num_objects_current * 100}%")
# ------''-------------------------------------------------------------------
'''

# %%
### Full Training
epochs = 50


columns = ["epoch", "method", "lr", "tr_loss", "tr_acc", "val_loss", "val_acc", "val_nll", "val_ece", "time"]
if True == True:
    columns = columns[:-1] + ["mc_val_loss", "mc_val_acc", "mc_val_nll", "mc_val_ece"] + columns[-1:]
    mc_res = {"loss": None, "accuracy": None, "nll" : None, "ece" : None}


best_val_loss=9999 ; best_val_acc=0 ; best_epoch=0 ; cnt=0; best_val_loss=9999
print("Start Training!!")
for epoch in range(int(epochs)):
    time_ep = time.time()

    '''
    ## lr scheduling
    if args.scheduler == "swag_lr":
        if args.method == "swag":
            lr = swag_utils.schedule(epoch, args.lr_init, args.epochs, True, args.swa_start, args.swa_lr)
        else:
            lr = swag_utils.schedule(epoch, args.lr_init, args.epochs, False, None, None)
        swag_utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = optimizer.param_groups[0]['lr']
    '''

    # train
    loss_sum = 0.0
    correct = 0.0

    num_objects_current = 0

    for batch, (X, y) in enumerate(tr_loader):
        X, y = X.to(device), y.to(device)
        
        ### first forward-backward pass
        # sample weight from bnn params
        params, z_1, z_2 = sabtl_model.sample(1.0)

        # forward & backward
        pred = torch.nn.utils.stateless.functional_call(sabtl_model.backbone, params, X)
        loss = criterion(pred, y)
        loss.backward()

        bnn_params = optimizer.first_step(zero_grad=True)
        params = sabtl.second_sample(bnn_params, z_1, z_2, sabtl_model, scale=1.0)

        ### second forward-backward pass
        pred = torch.nn.utils.stateless.functional_call(sabtl_model.backbone, params, X)

        criterion(pred, y).backward()
        optimizer.second_step(zero_grad=True)  

        ### Checking accuracy with MAP (Mean) solution
        params, _, _ = sabtl_model.sample(scale=0.0)
        pred = torch.nn.utils.stateless.functional_call(sabtl_model.backbone, params, X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss_sum += loss.data.item() * X.size(0)
        num_objects_current += X.size(0)
    print(f"Epoch {epoch} / Tr loss : {loss_sum / num_objects_current} / Tr accuracy : {correct / num_objects_current * 100}%")
    
    
    # eval
    loss_sum = 0.0
    correct = 0.0

    num_objects_current = 0

    for batch, (X, y) in enumerate(te_loader):
        X, y = X.to(device), y.to(device)
        
        ### Checking accuracy with MAP (Mean) solution
        pred = torch.nn.utils.stateless.functional_call(sabtl_model.backbone, params, X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss_sum += loss.data.item() * X.size(0)
        num_objects_current += X.size(0)
    print(f"Epoch {epoch} / Te loss : {loss_sum / num_objects_current} / Te accuracy : {correct / num_objects_current * 100}%") 
    
    # if True == True:
    #     values = [epoch + 1, f"sabtl-bsam", lr_init, tr_res["loss"], tr_res["accuracy"],
    #         val_res["loss"], val_res["accuracy"], val_res["nll"], val_res["ece"],
    #         mc_res["loss"], mc_res["accuracy"], mc_res["nll"], mc_res["ece"],   # 코딩 필요
    #             time_ep]

    # table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    # if epoch % 10 == 0:
    #     table = table.split("\n")
    #     table = "\n".join([table[1]] + table)
    # else:
    #     table = table.split("\n")[2]
    # print(table)

# %%