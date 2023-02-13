# %%
import os, sys
import time, copy
import tabulate

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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tr_loader, val_loader, te_loader, num_classes = data.get_cifar10(data_path = "/DATA1/lsj9862/cifar10/",
                                                            batch_size = 128,
                                                            num_workers = 4,
                                                            use_validation = True)

from torchvision.models import resnet18
model = resnet18(pretrained=False, num_classes=num_classes).to(device)

# %%
w_mean = torch.load("/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/swag-sgd_constant/10_101_1_0.01/swag-sgd_best_val_mean.pt")

w_var_sqrt = torch.load("/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/swag-sgd_constant/10_101_1_0.01/swag-sgd_best_val_variance.pt") 

w_covmat = torch.load("/mlainas/lsj9862/exp_result/cifar10/resnet18-noBN/swag-sgd_constant/10_101_1_0.01/swag-sgd_best_val_covmat.pt")

# %%
sabtl_model = sabtl.SABTL(copy.deepcopy(model),  w_mean = w_mean,
                        w_var = w_var_sqrt, diag_only=False, w_cov=w_covmat).to(device)


# swag_model = swag.SWAG(copy.deepcopy(model), no_cov_mat=True, max_num_models=20).to(device)


# %%
criterion = torch.nn.CrossEntropyLoss()


# %%
## SGD
# optimizer = torch.optim.SGD(sabtl_model.parameters(), momentum=0.9,
#                     lr=0.01, weight_decay=5e-4, nesterov=False)


## SAM
# base_optimizer = torch.optim.SGD
# optimizer = SAM(sabtl_model.parameters(), base_optimizer, rho=0.05, lr=0.01, momentum=0.9,
#                 weight_decay=5e-4, nesterov=False)

## BSAM
base_optimizer = torch.optim.SGD
optimizer = sabtl.BSAM(sabtl_model.parameters(), base_optimizer, sabtl_model, rho=0.05, lr=0.01, momentum=0.9,
        weight_decay=5e-4, nesterov=False)


# %%
for batch, (X, y) in enumerate(tr_loader):
    X, y = X.to(device), y.to(device)

    # sample_w, z_1, z_2 = sabtl_model.sample()
    # sabtl_model.set_sampled_parameters(sample_w)
    
    # first forward-backward pass
    pred, z_1, z_2 = sabtl_model(X)
    loss = criterion(pred, y)
    loss.backward()

    '''
    gradient가 bnn param까지 안 흐른다..
    '''
    optimizer.first_step(z_1=z_1, z_2=z_2, zero_grad=True)
    
    # second forward-backward pass
    
    criterion(sabtl_model.backbone(X), y).backward()
    optimizer.second_step(zero_grad=True)
    
    

# %%
'''
columns = ["epoch", "method", "lr", "tr_loss", "tr_acc", "val_loss", "val_acc", "val_nll", "val_ece", "time"]
print("Start Training!")
for epoch in range(300):
    time_ep = time.time()

    ## train
    sample_param = sabtl_model.sample(1.0)
    sabtl.set_sampled_parameters(model, sample_param)
    tr_res = utils.train_sgd(tr_loader, model, criterion, optimizer, device, True)
    # tr_res = utils.train_sam(tr_loader, model, criterion, optimizer, device, True)

    ## eval
    map_param = sabtl_model.sample(0.0)
    sabtl.set_sampled_parameters(model, sample_param)
    val_res = utils.eval_metrics(val_loader, model, criterion, device, 50, 1e-8)

    time_ep = time.time() - time_ep


    values = [epoch + 1, "sabtl", 0.01, tr_res["loss"], tr_res["accuracy"],
        val_res["loss"], val_res["accuracy"], val_res["nll"], val_res["ece"],
        time_ep]
    
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 5 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)
'''
# %%
