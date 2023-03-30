import random

import torch
from torch.utils.data import random_split

import torchvision.transforms as transforms
import torchvision

from torchvision.datasets import CIFAR10, CIFAR100

import utils


####################################################################################################################################
### CIFAR-10 ----------------------------------------------------------------------------------------------------------------------|
def get_cifar10(data_path='/mlainas/lsj9862/cifar10', batch_size=256, num_workers=4, use_validation=False, val_ratio=0.1):
    transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    tr_data = CIFAR10(data_path, train=True, transform=transform_train,
                                            download=True)
    te_data = CIFAR10(data_path, train=False, transform = transform_test,
                                            download=True)
    
    
    if use_validation:
        val_len = int(len(tr_data) * val_ratio)
        tr_len = len(tr_data) - val_len
        
        tr_data, val_data = random_split(tr_data, [tr_len, val_len])


        tr_loader = torch.utils.data.DataLoader(tr_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(val_data,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers,
                                        pin_memory=True)
    else:
        print("[Warning] You are going to run models on the test set.")

        tr_loader = torch.utils.data.DataLoader(tr_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=True)
        val_loader= None

    te_loader = torch.utils.data.DataLoader(te_data,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        pin_memory=True)

    num_classes = max(te_data.targets) + 1

    return tr_loader, val_loader, te_loader, num_classes

####################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------|


####################################################################################################################################
### CIFAR-100 ---------------------------------------------------------------------------------------------------------------------|
def get_cifar100(data_path='/mlainas/lsj9862/cifar100', batch_size=256, num_workers=4, use_validation=False, val_ratio=0.1):
    transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    tr_data = CIFAR100(data_path, train=True, transform=transform_train,
                                            download=True)
    te_data = CIFAR100(data_path, train=False, transform = transform_test,
                                            download=True)
    
    
    if use_validation:
        val_len = int(len(tr_data) * val_ratio)
        tr_len = len(tr_data) - val_len
        
        tr_data, val_data = random_split(tr_data, [tr_len, val_len])


        tr_loader = torch.utils.data.DataLoader(tr_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(val_data,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers,
                                        pin_memory=True)
    else:
        print("[Warning] You are going to run models on the test set.")

        tr_loader = torch.utils.data.DataLoader(tr_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=True)
        val_loader= None

    te_loader = torch.utils.data.DataLoader(te_data,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        pin_memory=True)

    num_classes = max(te_data.targets) + 1

    return tr_loader, val_loader, te_loader, num_classes

####################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------|
