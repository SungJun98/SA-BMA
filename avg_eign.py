# %%
import torch
import os.path as osp
import numpy as np


# %%
PATH = f"/data2/lsj9862/exp_result/seed_2/cifar100/scratch_resnet18-noBN/dnn-sgd/swag_lr_0.005/0.05_0.005_0.9/performance/performance.pt"
BNN = False
performance = torch.load(PATH)

# %%
acc = performance['accuracy']
ece = performance['ece']
nll = performance['nll']

max_eign = np.mean(performance['tr_max_eign'])

if BNN:
    tr_eign_matrix = np.array(performance['tr_cum_eign'])
    eign_ratio =np.mean(tr_eign_matrix[:,-1] / tr_eign_matrix[:,0])
else:
    eign_ratio = performance['tr_cum_eign'][-1] / performance['tr_cum_eign'][0]
print(f"acc : {acc}")
print(f"ece : {ece}")
print(f"nll : {nll}")
print(f"max eign : {max_eign}")
print(f"eign ratio : {eign_ratio}")
# %%