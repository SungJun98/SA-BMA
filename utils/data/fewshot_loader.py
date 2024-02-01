# %%
import sys, os
import torch
import numpy as np
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from timm.data.transforms_factory import create_transform

# %%
DATASET='imagenet'
DATA_PATH=f'/data1/lsj9862/data/{DATASET}'
BATCH_SIZE=256
USE_VALIDATION=True
AUG=False
VAL_RATIO=0.1
DAT_PER_CLS=16
SEED=1


# %%
## Set Seed
def set_seed(RANDOM_SEED=0):
    '''
    Set seed for reproduction
    '''
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random
    random.seed(RANDOM_SEED)

set_seed(SEED)

# %%
## Get transform
from data import create_transform_v2
transform_train, transform_test = create_transform_v2(data_name=DATASET, aug=AUG)   

# %%
## Load Data
if DATASET == 'cifar10':
    tr_data = CIFAR10(DATA_PATH, train=True, transform = transform_train,
                                        download=True)
    te_data = CIFAR10(DATA_PATH, train=False, transform = transform_test,
                                        download=True)
    num_classes = max(te_data.targets) + 1

elif DATASET == 'cifar100':
    tr_data = CIFAR100(DATA_PATH, train=True, transform = transform_train,
                                        download=True)
    te_data = CIFAR100(DATA_PATH, train=False, transform = transform_test,
                                            download=True)
    num_classes = max(te_data.targets) + 1

elif DATASET == 'imagenet':
    tr_data = ImageNet(DATA_PATH, split='train')
    te_data = ImageNet(DATA_PATH, split='val')
    num_classes = max(te_data.targets) + 1

print(f"Number of classes : {num_classes} :: Data point per class : {DAT_PER_CLS}")




# %%
## Split Data
val_len = int(len(tr_data) * VAL_RATIO)
tr_len = len(tr_data) - val_len

tr_data, val_data = random_split(tr_data, [tr_len, val_len])

# %%
## Pre-Setting for Few-shot Setting
class_indices = [[] for _ in range(num_classes)]
for idx, (_, target) in enumerate(tr_data):
    class_indices[target].append(idx)

few_shot_indices = []
for indices in class_indices:
    few_shot_indices.extend(indices[:DAT_PER_CLS])

sampler = SubsetRandomSampler(few_shot_indices)

# %%
## Make Loader
tr_loader = DataLoader(tr_data,
                    batch_size=BATCH_SIZE,
                    num_workers=4,
                    pin_memory=True,
                    sampler=sampler)

if DATASET in ['imagenet']:
    val_loader = DataLoader(val_data,
                        batch_size=BATCH_SIZE,
                        num_workers=4,
                        pin_memory=True,)

    te_loader = DataLoader(te_data,
                        batch_size=BATCH_SIZE,
                        num_workers=4,
                        pin_memory=True,)


# %%
os.makedirs(f'/mlainas/lsj9862/data/{DATASET}/{DAT_PER_CLS}shot', exist_ok=True)
torch.save(tr_loader, f'/mlainas/lsj9862/data/{DATASET}/{DAT_PER_CLS}shot/tr_loader_seed{SEED}.pth')
if DATASET in ['imagenet']:
    torch.save(val_loader, f'/mlainas/lsj9862/data/{DATASET}/{DAT_PER_CLS}shot/val_loader_seed{SEED}.pth')
    torch.save(te_loader, f'/mlainas/lsj9862/data/{DATASET}/{DAT_PER_CLS}shot/te_loader_seed{SEED}.pth')
# %%
