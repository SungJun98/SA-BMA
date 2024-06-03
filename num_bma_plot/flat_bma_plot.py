## flatness-base bma plot
# %%
import seaborn as sns
import numpy as nps
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
plt.rc('font', size=18)        # 기본 폰트 크기
plt.rc('axes', labelsize=16)   # x,y축 label 폰트 크기
plt.rc('xtick', labelsize=15)  # x축 눈금 폰트 크기 
plt.rc('ytick', labelsize=15)  # y축 눈금 폰트 크기
plt.rc('legend', fontsize=16)  # 범례 폰트 크기
plt.rc('figure', titlesize=16) # figure title 폰트 크기

# %%
## Load
SEED = 0                    # option [0]
DATASET = 'cifar100'        # option ['cifar10', 'cifar100']
OPTIM = 'sgd'               # option ['sgd', 'sam']
SCHEDULER = 'swag_lr'       # option ['constant', 'cos_decay', 'swag_lr']
PATH = f"/data2/lsj9862/bma_num_plot/seed_{SEED}/{DATASET}/swag-{OPTIM}/{SCHEDULER}/performance"

SAVE_PATH = f"./max/flat_bma_plot"  # set save path

### Load Data
# Flat
flat_dat = torch.load(f"{PATH}/flat_bma_num.pt")
flat_df = pd.DataFrame(flat_dat)
flat_df = flat_df.iloc[:-1, :]
flat_df['error'] = 1 - flat_df['acc']

# Random
rand_dat = torch.load(f"{PATH}/rand_bma_num.pt")   
rand_df = pd.DataFrame(rand_dat)
rand_df = rand_df.iloc[:-1, :]
rand_df['error'] = 1 - rand_df['acc']

# Sharp
sharp_dat = torch.load(f"{PATH}/sharp_bma_num.pt")   
sharp_df = pd.DataFrame(sharp_dat)
sharp_df = sharp_df.iloc[:-1, :]
sharp_df['error'] = 1 - sharp_df['acc']
# ----------------------------------------------   


# %%
### Plot Error
# acc = pd.concat([flat_df.iloc[:,:2], rand_df.iloc[:,1], sharp_df.iloc[:,1]], axis=1)
# acc.columns = ['num_models', 'Sharp', 'Rand', 'Flat']
err = pd.concat([flat_df['num_models'], 
                flat_df['error'],
                rand_df['error'],
                sharp_df['error']], axis=1)
err.columns = ['num_models', 'Sharp', 'Rand', 'Flat']

labels = ['Sharp', 'Rand', 'Flat']

fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
for _, label in enumerate(labels):
    sns.lineplot(x=err['num_models'], y=err[label], label=label) # label 범례 라벨
    plt.legend()

plt.xlabel('# of BMA Models')
plt.ylabel('BMA Error')

plt.savefig(f'{SAVE_PATH}/{DATASET}_swag-{OPTIM}-{SCHEDULER}_acc.png', transparent=True, dpi=500)




# %%
### Plot nll
nll = pd.concat([flat_df['num_models'],
            flat_df['nll'],
            rand_df['nll'],
            sharp_df['nll']], axis=1)
nll.columns = ['num_models', 'Sharp', 'Rand', 'Flat']

labels = ['Sharp', 'Rand', 'Flat']

fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
for _, label in enumerate(labels):
    sns.lineplot(x=nll['num_models'], y=nll[label], label=label)
    plt.legend()

plt.xlabel('# of BMA Models')
plt.ylabel('BMA NLL')

plt.savefig(f'{SAVE_PATH}/{DATASET}_swag-{OPTIM}-{SCHEDULER}_nll.png', transparent=True, dpi=500)



# %%
### Plot ECE
ece = pd.concat([flat_df['num_models'],
            flat_df['ece'],
            rand_df['ece'],
            sharp_df['ece']], axis=1)
ece.columns = ['num_models', 'Sharp', 'Rand', 'Flat']

labels = ['Sharp', 'Rand', 'Flat']

fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
for _, label in enumerate(labels):
    sns.lineplot(x=ece['num_models'], y=ece[label], label=label)
    plt.legend()

plt.xlabel('# of BMA Models')
plt.ylabel('BMA ECE')

plt.savefig(f'{SAVE_PATH}/{DATASET}_swag-{OPTIM}-{SCHEDULER}_ece.png', transparent=True, dpi=500)
# %%
