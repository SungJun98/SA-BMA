# %%
import sys, os
import torch
import numpy as np
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from timm.data.transforms_factory import create_transform

# %%
dataset='cifar100'
data_path=f'/data1/lsj9862/data/{dataset}'
batch_size=256
use_validation=True
aug=True
val_ratio=0.1
dat_per_cls=16
seed=2


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

set_seed(seed)

# %%
## Get transform
if aug:
    transform_train = create_transform(224, is_training=True)
    transform_test = create_transform(224)
else:
    transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

    transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])        

# %%
## Load Data
if dataset=='cifar10':
    tr_data = CIFAR10(data_path, train=True, transform=transform_train,
                                            download=True)
    te_data = CIFAR10(data_path, train=False, transform = transform_test,
                                            download=True)
elif dataset=='cifar100':
    tr_data = CIFAR100(data_path, train=True, transform=transform_train,
                                            download=True)
    te_data = CIFAR100(data_path, train=False, transform = transform_test,
                                            download=True)
    
num_classes = max(te_data.targets) + 1
print(f"Number of classes : {num_classes} :: Data point per class : {dat_per_cls}")


# %%
## Split Data
val_len = int(len(tr_data) * val_ratio)
tr_len = len(tr_data) - val_len

tr_data, val_data = random_split(tr_data, [tr_len, val_len])

# %%
## Pre-Setting for Few-shot Setting
class_indices = [[] for _ in range(num_classes)]
for idx, (_, target) in enumerate(tr_data):
    class_indices[target].append(idx)

few_shot_indices = []
for indices in class_indices:
    few_shot_indices.extend(indices[:dat_per_cls])

sampler = SubsetRandomSampler(few_shot_indices)

# %%
## Make Loader
tr_loader = DataLoader(tr_data,
                    batch_size=batch_size,
                    num_workers=0,
                    pin_memory=True,
                    sampler=sampler)
    
# %%
os.makedirs(f'/mlainas/lsj9862/data/{dataset}/{dat_per_cls}shot', exist_ok=True)
torch.save(tr_loader, f'/mlainas/lsj9862/data/{dataset}/{dat_per_cls}shot/tr_loader_seed{seed}.pth')
# %%
