## scatter plot
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
SEED = 0
DATASET = 'cifar100'     # option :['cifar10', 'cifar100']
OPTIM = 'sgd'            # optoin : ['sgd', 'sam']
SCHEDULER = 'swag_lr'    # option : ['constant', 'cos_decay', 'swag_lr']

PERFORMANCE_PATH = f"/data2/lsj9862/bma_num_plot/seed_{SEED}/{DATASET}/swag-{OPTIM}/{SCHEDULER}/performance/performance_final.pt"
dict = torch.load(PERFORMANCE_PATH)
del dict['tr_cum_eign']
df = pd.DataFrame(dict)
df["error"] = 100 - df['accuracy']

# %%
## Normalizing Error, ECE, NLL to plot in one figure
df['error_norm'] = (df['error'] - df['error'].min()) / (df['error'].max() - df['error'].min())
df['ece_norm'] = (df['ece'] - df['ece'].min()) / (df['ece'].max() - df['ece'].min())
df['nll_norm'] = (df['nll'] - df['nll'].min()) / (df['nll'].max() - df['nll'].min())


# %%
## Plot Scatterplot and Calculate Correlation 
plt.figure(figsize=(8,6))

# Acc v.s. Eigen
label_error = rf"Error ($\rho$={df['error_norm'].corr(df['tr_max_eign']):.2f})"
sns.scatterplot(x=df['tr_max_eign'], y=df['error_norm'], label=label_error, alpha=0.6)
# sns.scatterplot(x=df['tr_eign_ratio_5'], y=df['error_norm'], label='Error', alpha=0.6)

# ECE v.s. Eign
label_ece = rf"ECE ($\rho$={df['ece_norm'].corr(df['tr_max_eign']):.2f})"
sns.scatterplot(x=df['tr_max_eign'], y=df['ece_norm'], label=label_ece, alpha=0.6)
# sns.scatterplot(x=df['tr_eign_ratio_5'], y=df['ece_norm'], label='ECE', alpha=0.6)

# NLL v.s. Eign
label_nll = rf"NLL ($\rho$={df['nll_norm'].corr(df['tr_max_eign']):.2f})"
sns.scatterplot(x=df['tr_max_eign'], y=df['nll_norm'], label=label_nll, alpha=0.6)
# sns.scatterplot(x=df['tr_eign_ratio_5'], y=df['nll_norm'], label='NLL', alpha=0.6)

plt.xlabel('Max Eign')  ##### 여기에 lambda_1 추가 좀 부탁해요!! 즉 "lambda_1 (Max Eigen)" 이런식으로
plt.ylabel('Performance')
plt.legend()


## save figure
plt.savefig(f'./max/corr_plot/{DATASET}_swag-{OPTIM}-{SCHEDULER}_corr.png', transparent=True, dpi=500)
# print(f"Corr w Err : {df['error_norm'].corr(df['tr_eign_ratio_5'])}")
# print(f"Corr w ECE : {df['ece_norm'].corr(df['tr_eign_ratio_5'])}")
# print(f"Corr w NLL : {df['nll_norm'].corr(df['tr_eign_ratio_5'])}")
print(f"Corr w Err : {df['error_norm'].corr(df['tr_max_eign'])}")
print(f"Corr w ECE : {df['ece_norm'].corr(df['tr_max_eign'])}")
print(f"Corr w NLL : {df['nll_norm'].corr(df['tr_max_eign'])}")
# %%
