import random

import torch
import torchvision
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
import torchvision.transforms as transforms

from timm.data.transforms_factory import create_transform

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

"""
class ExtractedDataSet(Dataset): 
    '''
    Class for Feature Extracted data
    '''
    def __init__(self, feature, target):
        self.x = feature
        self.y = target
            
    def __len__(self): 
        return len(self.x)

    def __getitem__(self, idx): 
        x = self.x[idx]
        y = self.y[idx]
        return x, y
"""

def create_transform_v2(data_name='cifar10', aug=True, scale=None, ratio=None,
                    hflip=0.5, vflip=0, color_jitter=0.4, aa=None,
                    re_prob=0., re_mode='const', re_count=1):
    # Create transform for dataset
    if data_name in ['cifar10', 'cifar100']:
        if aug:
            transform_train = create_transform(224, is_training=True,
                                        scale=scale, ratio=ratio, hflip=hflip, vflip=vflip,
                                        color_jitter=color_jitter, auto_augment=aa, 
                                        re_prob=re_prob, re_mode=re_mode, re_count=re_count)
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
    elif data_name in ['imagenet']:
        if aug:
            raise NotImplementedError("No code for augmentation on ImageNet")
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

            transform_train = transforms.Compose(
                [transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
                ])
            
            transform_test = transforms.Compose(
                [transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
                ])
        
    return transform_train, transform_test



def create_dataset(data_name='cifar10', data_path='/data1/lsj9862/data/cifar10',
            use_validation=True, val_ratio=0.1,
            dat_per_cls=-1, seed=0, 
            transform_train=None, transform_test=None):
    # Load Data
    if data_name in ['cifar10', 'cifar100']:
        if data_name == 'cifar10':
            tr_data = CIFAR10(data_path, train=True, transform = transform_train,
                                                download=True)
            te_data = CIFAR10(data_path, train=False, transform = transform_test,
                                                download=True)
            num_classes = max(te_data.targets) + 1
        
        elif data_name == 'cifar100':
            tr_data = CIFAR100(data_path, train=True, transform = transform_train,
                                                download=True)
            te_data = CIFAR100(data_path, train=False, transform = transform_test,
                                                    download=True)
            num_classes = max(te_data.targets) + 1
        
        # Split Validation Data
        val_data = None
        if use_validation:
            val_len = int(len(tr_data) * val_ratio)
            tr_len = len(tr_data) - val_len
            
            tr_data, val_data = random_split(tr_data, [tr_len, val_len])

        if dat_per_cls > 0:
            tr_data = torch.load(f'{data_path}/{dat_per_cls}shot/tr_loader_seed{seed}.pth')
        
        
    elif data_name == 'imagenet':
        import pickle
        with open(f'{data_path}/{dat_per_cls}shot/tr_loader_seed{seed}.pth', 'rb') as f:
            tr_data = pickle.load(f)
        with open(f'{data_path}/{dat_per_cls}shot/val_loader_seed{seed}.pth', 'rb') as f:
            val_data = pickle.load(f)
        with open(f'{data_path}/{dat_per_cls}shot/te_loader_seed{seed}.pth', 'rb') as f:
            te_data = pickle.load(f)
        num_classes = max(te_data.targets) + 1

    return tr_data, val_data, te_data, num_classes



def create_loader(data_name='cifar10',
                tr_data=None, val_data=None, te_data=None,
                use_validation=True,
                batch_size=256,
                num_workers=4,
                dat_per_cls=-1,
                ):
    
    if data_name in ['cifar10', 'cifar100']:
        if dat_per_cls < 0:
            tr_loader = DataLoader(tr_data,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    )
        else:
            tr_loader = tr_data

        te_loader = DataLoader(te_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            )

        if use_validation:
            val_loader = DataLoader(val_data,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    )
        else:
            val_loader = te_loader
            print("[Warning] You are going to run models on the test set.")
    
    elif data_name in ['imagenet']:
        if dat_per_cls < 0:
            raise NotImplementedError("No data for full-shot ImageNet")
        else:
            tr_loader = tr_data
            val_loader = val_data
            te_loader = te_data
        
        
    return tr_loader, val_loader, te_loader

