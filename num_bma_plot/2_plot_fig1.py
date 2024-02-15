# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

plt.rc('font', size=20)        # 기본 폰트 크기
plt.rc('axes', labelsize=20)   # x,y축 label 폰트 크기
plt.rc('xtick', labelsize=20)  # x축 눈금 폰트 크기 
plt.rc('ytick', labelsize=20)  # y축 눈금 폰트 크기
plt.rc('legend', fontsize=20)  # 범례 폰트 크기
plt.rc('figure', titlesize=20) # figure title 폰트 크기

# %%
### Load Data
path = "/home/lsj9862/SA-BTL"

# Flat
flat_dat = torch.load(f"{path}/flat_bma_num.pt")
flat_df = pd.DataFrame(flat_dat)
flat_df = flat_df.iloc[:-1, :]
flat_df['error'] = 1 - flat_df['acc']

# Random
rand_dat = torch.load(f"{path}/rand_bma_num.pt")   
rand_df = pd.DataFrame(rand_dat)
rand_df = rand_df.iloc[:-1, :]
rand_df['error'] = 1 - rand_df['acc']

# Sharp
sharp_dat = torch.load(f"{path}/sharp_bma_num.pt")   
sharp_df = pd.DataFrame(sharp_dat)
sharp_df = sharp_df.iloc[:-1, :]
sharp_df['error'] = 1 - sharp_df['acc']
# ----------------------------------------------   

# %%
## normalization
flat_df['error_norm'] = (flat_df['error'] - flat_df['error'].min())/(flat_df['error'].max() - flat_df['error'].min())
flat_df['ece_norm'] = (flat_df['ece'] - flat_df['ece'].min())/(flat_df['ece'].max() - flat_df['ece'].min())
flat_df['nll_norm'] = (flat_df['nll'] - flat_df['nll'].min())/(flat_df['nll'].max() - flat_df['nll'].min())

rand_df['error_norm'] = (rand_df['error'] - rand_df['error'].min())/(rand_df['error'].max() - rand_df['error'].min())
rand_df['ece_norm'] = (rand_df['ece'] - rand_df['ece'].min())/(rand_df['ece'].max() - rand_df['ece'].min())
rand_df['nll_norm'] = (rand_df['nll'] - rand_df['nll'].min())/(rand_df['nll'].max() - rand_df['nll'].min())

sharp_df['error_norm'] = (sharp_df['error'] - sharp_df['error'].min())/(sharp_df['error'].max() - sharp_df['error'].min())
sharp_df['ece_norm'] = (sharp_df['ece'] - sharp_df['ece'].min())/(sharp_df['ece'].max() - sharp_df['ece'].min())
sharp_df['nll_norm'] = (sharp_df['nll'] - sharp_df['nll'].min())/(sharp_df['nll'].max() - sharp_df['nll'].min())




# %%
### Plot Acc (error)
acc = pd.concat([flat_df['num_models'], flat_df['error'], rand_df['error'], sharp_df['error']], axis=1)
acc.columns = ['num_models', 'Flat', 'Rand', 'Sharp']

fig = plt.figure()

sns.lineplot(x=acc['num_models'], y=acc['Sharp'], label='Sharp') # label 범례 라벨
sns.lineplot(x=acc['num_models'], y=acc['Rand'], label='Rand') # label 범례 라벨
sns.lineplot(x=acc['num_models'], y=acc['Flat'], label='Flat') # label 범례 라벨
plt.legend()

plt.xlabel('# of BMA Models')
plt.ylabel('BMA Error')

# plt.savefig(f'{path}/acc.png', transparent=True, dpi=500)




# %%
### Plot nll
nll = pd.concat([flat_df['num_models'], flat_df['nll'], rand_df['nll'], sharp_df['nll']], axis=1)
nll.columns = ['num_models', 'Sharp', 'Rand', 'Flat']

labels = ['Sharp', 'Rand', 'Flat']

fig = plt.figure(figsize=(10, 6))
fig.set_facecolor('white')
for _, label in enumerate(labels):
    sns.lineplot(x=nll['num_models'], y=nll[label], label=label)
    plt.legend()

# plt.xlabel('# of BMA Models')
# plt.ylabel('BMA NLL')
plt.xlabel('')
plt.ylabel('')

plt.savefig(f'{path}/nll.png', transparent=True, dpi=500)



# %%
### Plot ECE
ece = pd.concat([flat_df['num_models'], flat_df['ece'], rand_df['ece'], sharp_df['ece']], axis=1)
ece.columns = ['num_models', 'Sharp', 'Rand', 'Flat']

labels = ['Sharp', 'Rand', 'Flat']

fig = plt.figure(figsize=(10, 6))
fig.set_facecolor('white')
for _, label in enumerate(labels):
    sns.lineplot(x=ece['num_models'], y=ece[label], label=label)
    plt.legend()

plt.xlabel('# of BMA Models')
plt.ylabel('BMA ECE')

# plt.savefig(f'{path}/ece.png', transparent=True, dpi=500)
# %%
