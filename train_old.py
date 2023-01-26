# %%
import os, sys
import time, copy
# import tabulate
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision

import numpy as np
import pickle

import utils
from models import mlp

from baselines.sam.sam import SAM
from baselines.swag import swag, swag_utils
from sghmc.losses import GaussianPriorCELossShifted
from sghmc.sghmc_model import configure_optimizers
from sghmc.sgld import SGLD
from sghmc.lr_scheduler import CosineLR



## Optimizer & Scheduler
# https://github.com/hsouri/BayesianTransferLearning/blob/main/priorBox/sghmc/sghmc_model.py#L116
def configure_optimizers(is_sgld=False, lr_init=0.05, weight_decay=0, temperature=2e-8, N=20,
                    cyclic_lr=False, n_cycles=4, n_samples=12, num_of_batches=2, epochs=200,
):
    """
    Input:
    - is_sgld : run SGHMC (default : False)
    - temperature : temperature of the posterior (defalt : 2e-8)
    - N : len(loader.dataset) 
    - cyclic_lr : run cyclic lr schedule (default : False)
    - n_cycles : number of lr annealing cycles (default : 4)
    - n_samples : number of total samples (default : 12)
    - num_of_batches : len(loader)
    """
    if is_sgld:
        optimizer = SGLD(model.parameters(), lr=lr_init, weight_decay=weight_decay,
                        temperature=temperature / N,
        )
        if cyclic_lr:
            scheduler = CosineLR(optimizer, n_cycles=n_cycles, n_samples=n_samples,
                                    T_max=num_of_batches * epochs)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_of_batches * epochs)
    else:
        optimizer = torch.optim.SGD
        optimizer = optimizer(model.parameters(), lr=lr_init, nesterov=True, momentum=momentum,
            weight_decay=weight_decay,)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_of_batches * epochs)
    return optimizer, scheduler


is_sgld=True; lr_init = 0.1; momentum=0.9; nesterov=True; weight_decay=0
cyclic_lr=True; n_cycles=4; n_samples=12; max_epochs=100; temperature=2e-8
N=len(down_loader.dataset); num_of_batches=len(down_loader)

optimizer, scheduler = configure_optimizers(is_sgld, lr_init, weight_decay, temperature, N,
                    cyclic_lr, n_cycles, n_samples, num_of_batches, max_epochs)

# %%
## Running PTL
raw_params=True; clip_val=2
samples_dir='/home/lsj9862/BayesianSAM/samples/'

print("Running PTL...")
utils.set_seed(0)
best_val_loss=9999 ; best_val_acc=0 ; best_epoch=0
# Training full epochs
for epoch in tqdm(range(int(max_epochs))):
    print('-'*30)
    print(f"[Epoch {epoch+1}]")

    # train
    model.train()
    for batch, (X, y) in enumerate(tr_loader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)

        if raw_params:
            params = torch.flatten(torch.cat([torch.flatten(p) for p in model.parameters()])).to(device)
        else:
            params = torch.flatten(torch.cat([torch.flatten(model.state_dict()[p]) for p in model.state_dict()])).to(device)
        
        metrices = criterion(pred, y, N=N, params=params)
        

        # Backprop
        optimizer.zero_grad()
        metrices["loss"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        if is_sgld:
            if cyclic_lr:
                if scheduler.get_last_beta() < scheduler.beta:
                    optimizer.step(noise=False)
                    # optimizer.step()
                else:
                    optimizer.step()
            else:
                optimizer.step()
            if cyclic_lr and scheduler.should_sample():
                torch.save(model.state_dict(), samples_dir + f'/s_e{epoch+1}_m{batch+1}.pt')
        else:
            optimizer.step()
    scheduler.step()
    torch.cuda.empty_cache()

    print(f"train loss : {metrices['loss']}")


    # Validation (BMA를 적용 전)
    size = len(te_loader.dataset)
    num_batches = len(te_loader)
    model.eval()
    loss, correct = 0, 0

    with torch.no_grad():
        for X, y in te_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            metrices = criterion(pred, y, N, params)
            loss += criterion(pred, y ,N, params)["loss"].item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    accuracy = correct * 100
    
    # if accuracy > best_val_acc:
    if loss < best_val_loss:
        cnt=1
        best_val_acc = accuracy
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "")
    else:
        cnt = cnt + 1

    if cnt==10:
        break

    print(f"Test Loss : {loss} / Test Accuracy : {accuracy} ")
    print(f"nll: {metrices['nll']} / prior: {metrices['prior']}")
print(f"Best test accuracy {best_val_acc}% on epoch {best_epoch}")


'''
# %%
### Bayesian SAM

utils.set_seed(0) # args.seed

## Setting
device = "cuda" if torch.cuda.is_available() else "cpu"
# 추후에 multiple GPU 쓸 경우 코드 수정

with open('/home/lsj9862/BayesianSAM/data/toy_7/toy_7.pkl','rb') as f: # open 안에 args.data_path로 교체
    _, down_loader, val_loader, te_loader  = pickle.load(f)

## Model
model = mlp.MLP().to(device)
model.load_state_dict(torch.load("/home/lsj9862/BayesianSAM/source_pt/sgd_source.pt"))


# hyperparameters
no_schedule=False; swa_start=1 ; swa=True ; swa_lr=0.01 ; lr_init=0.1 ; momentum=0 ; nesterov=False
diag_only=False; swa_c_epochs=1 ; max_num_models=20; mask=None
swa_c_batches=None; parallel=False ; n_epochs=500

# criterion
criterion = torch.nn.CrossEntropyLoss()

## Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr_init, momentum=momentum, nesterov=nesterov)

# # optimizer
# base_optimizer = torch.optim.SGD
# optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, lr=0.1, momentum=0.9, nesterov=True)


## Rescaling
mean = torch.load('/home/lsj9862/BayesianSAM/source_pt/swag_source_mean.pt')
variance = torch.load('/home/lsj9862/BayesianSAM/source_pt/swag_source_variance.pt')
cov_mat = torch.load('/home/lsj9862/BayesianSAM/source_pt/swag_source_covmat.pt')

prior_scale = 1 # variance for the prior
prior_eps = 1e1 # adding to the prior variance
scale_low_rank = True # if we scale also the low rank cov matrix
number_of_samples_prior = 5 # the number of samples for the covariance of the prior

if number_of_samples_prior > 0:
    if scale_low_rank:
        cov_mat_sqrt = prior_scale * (cov_mat[:number_of_samples_prior])
    else:
        cov_mat_sqrt = cov_mat[:number_of_samples_prior]

mean = mean
variance = prior_scale * variance + prior_eps


# %%
# from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
# mvn = LowRankMultivariateNormal(mean, cov_mat_sqrt.t(), variance)

print("Running SWAG...")
swag_model = swag.SWAG(copy.deepcopy(model), no_cov_mat=diag_only, max_num_models=max_num_models).to(device)
swag_model.load_state_dict(torch.load("/home/lsj9862/BayesianSAM/source_pt/swag_source.pt"))
# Rescaling하기 전에 불러와야되는거 아닌가???
# 그렇게 하면 어떻게 rescaling을 해줄 것인가??
# 이게 더 현실적이다
# ----
# 반대로 rescaling한 mean, variance, cov_mat을 저장해놓고 swag_model 불러서 앞의 통계량을 적용하는 방식은 어떨까?



## SAM class 좀 더 뜯어서 공부(즉, optimizer에 대한 이해)하고 Bayesian SAM을 위한 optimizer 정의하자
# 

best_val_loss=9999 ; best_val_acc=0 ; best_epoch=0 ; cnt=0
# Training full epochs
for epoch in tqdm(range(int(n_epochs))):
    print('-'*30)
    print(f"[Epoch {epoch+1}]")
    
    # lr scheduling
    if not no_schedule:
        lr = swag_utils.schedule(epoch, lr_init, n_epochs, swa, swa_start, swa_lr)
        swag_utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = lr_init
    
    swag_model.sample(1.0)

    size = len(down_loader.dataset)
    model.train()

    for batch, (X, y) in enumerate(down_loader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss= criterion(pred, y)

        # Backprop
        optimizer.zero_grad()
        loss.backward()

        #################################
        ## 3) \nabla_w l(w) 구하기
        # w까지만 gradient 흐르게....
        # w.grad ??
        #################################

        #################################
        ## 4) Approximate Fisher 계산
        #################################

        #################################
        ## 5) \Delta \theta^*(w) 계산
        #################################

        #################################
        ## 6) BSAM loss 계산
        # \theta까지 gradient 흐르게 계산
        # mean.grad, variance.grad, cov_mat_sqrt.grad
        #################################
'''



'''
    tr_loss = utils.train(tr_loader, model, criterion, optimizer, device)
    print(f"(1) Training loss : {tr_loss}")
    if (swa and (epoch + 1) > swa_start and (epoch + 1 - swa_start) % swa_c_epochs == 0):
        swag_model.collect_model(model)
        swag_model.sample(0.0)
        swag_utils.bn_update(tr_loader, swag_model)
        swag_res = swag_utils.eval(te_loader, swag_model, criterion, device)
        print(f"(3) Test Loss : {swag_res['loss']} / (4) Test Accuracy : {swag_res['accuracy']} ")
        # if swag_res['accuracy'] > best_val_acc:
        if swag_res['loss'] < best_val_loss:
            cnt=1
            best_val_loss = swag_res["loss"]
            best_val_acc = swag_res["accuracy"]
            best_epoch = epoch
        else:
            cnt = cnt + 1

            # # save state_dict
            # torch.save(swag_model.state_dict(), '/home/lsj9862/BayesianSAM/source_pt/swag_source.pt')
            # torch.save(model.state_dict(), '/home/lsj9862/BayesianSAM/source_pt/swag_source_model.pt')

            # # Save Mean, variance, Covariance matrix
            # mean, variance, cov_mat_sqrt = swag_model.generate_mean_var_covar()
            
            # mean = swag_utils.flatten(mean)             # flatten
            # variance = swag_utils.flatten(variance)     # flatten
            # cov_mat = torch.cat([layer for layer in cov_mat_sqrt], dim=1)   # [max_num_model, num_of_params]
                
            # torch.save(mean,'/home/lsj9862/BayesianSAM/source_pt/swag_source_mean.pt')
            # torch.save(variance,'/home/lsj9862/BayesianSAM/source_pt/swag_source_variance.pt')
            # # torch.save(cov_mat_sqrt, '/home/lsj9862/BayesianSAM/source_pt/swag_source_cov_mat_sqrt.pt')
            # torch.save(cov_mat, '/home/lsj9862/BayesianSAM/source_pt/swag_source_covmat.pt')

    if cnt==10:
        break

print(f"Best Test Accuracy : {best_val_acc}% on epoch {best_epoch+1}")
print("Finish Training SWAG")
'''