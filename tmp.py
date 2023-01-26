# %%
import os, sys
import time, copy
import tabulate

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

import utils, data #, losses

from baselines.sam.sam import SAM, FSAM, BSAM
from baselines.swag import swag, swag_utils

import sabtl

import warnings
warnings.filterwarnings('ignore')

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tr_loader, val_loader, te_loader, num_classes = data.get_cifar10(data_path = "/data1/lsj9862/cifar10/",
                                                            batch_size = 128,
                                                            num_workers = 4,
                                                            use_validation = True)

from torchvision.models import resnet18
model = resnet18(pretrained=False, num_classes=num_classes).to(device)

# %%
w_mean = torch.load("/home/lsj9862/BayesianSAM/exp_result/cifar10/resnet18/swag-sgd_swag_lr/20_161_1_0.01/swag-sgd_best_val_mean.pt")

w_var_sqrt = torch.load("/home/lsj9862/BayesianSAM/exp_result/cifar10/resnet18/swag-sgd_swag_lr/20_161_1_0.01/swag-sgd_best_val_variance.pt") 

# %%
sabtl_model = sabtl.SABTL(copy.deepcopy(model),  w_mean = w_mean,
                        w_var = w_var_sqrt, no_cov_mat=True).to(device)


# %%
criterion = torch.nn.CrossEntropyLoss()


# %%
## SGD
optimizer = torch.optim.SGD(sabtl_model.parameters(), momentum=0.9,
                    lr=0.01, weight_decay=5e-4, nesterov=False)


## SAM
# base_optimizer = torch.optim.SGD
# optimizer = SAM(sabtl_model.parameters(), base_optimizer, rho=0.05, lr=0.01, momentum=0.9,
#                 weight_decay=5e-4, nesterov=False)

## BSAM
# base_optimizer = torch.optim.SGD
# optimizer = BSAM(sabtl_model.parameters(), base_optimizer, rho=0.05, lr=0.01, momentum=0.9,
#         weight_decay=5e-4, nesterov=False)



# %%
columns = ["epoch", "method", "lr", "tr_loss", "tr_acc", "val_loss", "val_acc", "val_nll", "val_ece", "time"]
print("Start Training!")
for epoch in range(10):
    time_ep = time.time()

    ## train
    sabtl_model.sample(0.5)
    tr_res = utils.train_sgd(tr_loader, sabtl_model, criterion, optimizer, device, True)
    # tr_res = utils.train_sam(tr_loader, model, criterion, optimizer, device, True)

    ## eval
    sabtl_model.sample(0.0)
    val_res = utils.eval_metrics(val_loader, sabtl_model, criterion, device, 50, 1e-8)

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

# %%
const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
}

from torchvision.models import resnet18
model = resnet18()

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
dnn_to_bnn(model, const_bnn_prior_parameters)

# %%
