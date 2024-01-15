# %%
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rc('font', size=20)        # 기본 폰트 크기
plt.rc('axes', labelsize=20)   # x,y축 label 폰트 크기
plt.rc('xtick', labelsize=20)  # x축 눈금 폰트 크기 
plt.rc('ytick', labelsize=20)  # y축 눈금 폰트 크기
plt.rc('legend', fontsize=20)  # 범례 폰트 크기
plt.rc('figure', titlesize=20) # figure title 폰트 크기

# %%
### Load Data
path = r"C:\Users\LimSungJun\Desktop\exp_result\cifar10\resnet18-noBN\swag-sgd\swag_lr\bma_num_plot"

# Flat
with open(f"{path}/flat_bma_num.pickle", 'rb') as f:
    flat_dat = pickle.load(f)

flat_df = pd.DataFrame(flat_dat)
flat_df = flat_df.iloc[:-1, :]
    
# Random
with open(f"{path}/rand_bma_num.pickle", 'rb') as f:
    rand_dat = pickle.load(f)    

rand_df = pd.DataFrame(rand_dat)
rand_df = rand_df.iloc[:-1, :]

# Sharp
with open(f"{path}/sharp_bma_num.pickle", 'rb') as f:
    sharp_dat = pickle.load(f)    
    
sharp_df = pd.DataFrame(sharp_dat)
sharp_df = sharp_df.iloc[:-1, :]
# ----------------------------------------------   


# %%
### Plot Acc
acc = pd.concat([flat_df.iloc[:,:2], rand_df.iloc[:,1], sharp_df.iloc[:,1]], axis=1)
acc.columns = ['num_models', 'Sharp', 'Rand', 'Flat']

labels = ['Sharp', 'Rand', 'Flat']

fig = plt.figure(figsize=(10, 6))
fig.set_facecolor('white')
for _, label in enumerate(labels):
    sns.lineplot(x=acc['num_models'], y=acc[label], label=label) # label 범례 라벨
    plt.legend()

plt.xlabel('# of BMA Models')
plt.ylabel('BMA Accuracy')

plt.savefig(f'{path}/acc.png', transparent=True, dpi=500)




# %%
### Plot nll
nll = pd.concat([flat_df.iloc[:,0], flat_df.iloc[:,2], rand_df.iloc[:,2], sharp_df.iloc[:,2]], axis=1)
nll.columns = ['num_models', 'Sharp', 'Rand', 'Flat']

labels = ['Sharp', 'Rand', 'Flat']

fig = plt.figure(figsize=(10, 6))
fig.set_facecolor('white')
for _, label in enumerate(labels):
    sns.lineplot(x=nll['num_models'], y=nll[label], label=label)
    plt.legend()

plt.xlabel('# of BMA Models')
plt.ylabel('BMA nll')

plt.savefig(f'{path}/nll.png', transparent=True, dpi=500)



# %%
### Plot ECE
ece = pd.concat([flat_df.iloc[:,0], flat_df.iloc[:,3], rand_df.iloc[:,3], sharp_df.iloc[:,3]], axis=1)
ece.columns = ['num_models', 'Sharp', 'Rand', 'Flat']

labels = ['Sharp', 'Rand', 'Flat']

fig = plt.figure(figsize=(10, 6))
fig.set_facecolor('white')
for _, label in enumerate(labels):
    sns.lineplot(x=ece['num_models'], y=ece[label], label=label)
    plt.legend()

plt.xlabel('# of BMA Models')
plt.ylabel('BMA ECE')

plt.savefig(f'{path}/ece.png', transparent=True, dpi=500)
# %%
