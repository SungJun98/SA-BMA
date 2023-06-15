import random

import torch
import torchvision
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms

from timm.data.transforms_factory import create_transform

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


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


####################################################################################################################################
### CIFAR-10 ----------------------------------------------------------------------------------------------------------------------|
def get_cifar10(data_path='/data1/lsj9862/data/cifar10',
            batch_size=256, num_workers=4, use_validation=True,
            aug=True, val_ratio=0.1, dat_per_cls=-1,
            seed=0
            ):
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
        

    ## Load Data
    tr_data = CIFAR10(data_path, train=True, transform=transform_train,
                                            download=True)
    te_data = CIFAR10(data_path, train=False, transform = transform_test,
                                            download=True)
    num_classes = max(te_data.targets) + 1
    
    if use_validation:
        val_len = int(len(tr_data) * val_ratio)
        tr_len = len(tr_data) - val_len
        
        tr_data, val_data = random_split(tr_data, [tr_len, val_len])
        
        
        ## Pre-Setting for Few-shot Setting
        if dat_per_cls >= 0:
            tr_loader = torch.load(f'/data1/lsj9862/data/cifar10/{dat_per_cls}shot/tr_loader_seed{seed}.pth')
        else:
            tr_loader = DataLoader(tr_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)
        
        val_loader = DataLoader(val_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)
    else:
        print("[Warning] You are going to run models on the test set.")

        tr_loader = DataLoader(tr_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)
        val_loader= None

    te_loader = DataLoader(te_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)

    return tr_loader, val_loader, te_loader, num_classes



def get_cifar10_fe(fe_dat="vitb16-i21k", batch_size=256, num_workers=0, use_validation=True, dat_per_cls=-1):
    if dat_per_cls >= 0:
        data_path= f"/data1/lsj9862/data/cifar10/{fe_dat}/{dat_per_cls}_shot/"
    else:
        data_path = f"/data1/lsj9862/data/cifar10/{fe_dat}/full_shot"
        
    tr_x = torch.load(f"{data_path}/tr_x.pt", map_location='cuda'); tr_y = torch.load(f"{data_path}/tr_y.pt", map_location='cuda')
    val_x = torch.load(f"{data_path}/val_x.pt", map_location='cuda'); val_y = torch.load(f"{data_path}/val_y.pt", map_location='cuda')
    te_x = torch.load(f"{data_path}/te_x.pt", map_location='cuda'); te_y = torch.load(f"{data_path}/te_y.pt", map_location='cuda')
    
    tr_data = ExtractedDataSet(tr_x, tr_y)
    val_data = ExtractedDataSet(val_x, val_y)
    te_data = ExtractedDataSet(te_x, te_y)
    
    tr_loader = torch.utils.data.DataLoader(tr_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=False)
    
    te_loader = torch.utils.data.DataLoader(te_data,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        pin_memory=False)
    
    if use_validation:
        val_loader = torch.utils.data.DataLoader(val_data,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers,
                                        pin_memory=False)
    else:
        print("[Warning] You are going to run models on the test set.")
        val_loader = te_loader
    
    num_classes = max(te_data.y) + 1

    return tr_loader, val_loader, te_loader, num_classes

####################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------|


####################################################################################################################################
### CIFAR-100 ---------------------------------------------------------------------------------------------------------------------|
def get_cifar100(data_path='/data1/lsj9862/data/cifar100',
            batch_size=256, num_workers=4, use_validation=True,
            aug=True, val_ratio=0.1, dat_per_cls=-1,
            seed=0):
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
        
    ## Load Data
    tr_data = CIFAR100(data_path, train=True, transform=transform_train,
                                            download=True)
    te_data = CIFAR100(data_path, train=False, transform = transform_test,
                                            download=True)
    num_classes = max(te_data.targets) + 1
    
    if use_validation:
        val_len = int(len(tr_data) * val_ratio)
        tr_len = len(tr_data) - val_len
        
        tr_data, val_data = random_split(tr_data, [tr_len, val_len])
        
        
        ## Pre-Setting for Few-shot Setting
        if dat_per_cls >= 0:
            tr_loader = torch.load(f'/data1/lsj9862/data/cifar100/{dat_per_cls}shot/tr_loader_seed{seed}.pth')
        else:
            tr_loader = DataLoader(tr_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)
        
        val_loader = DataLoader(val_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)
    else:
        print("[Warning] You are going to run models on the test set.")

        tr_loader = DataLoader(tr_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)
        val_loader= None

    te_loader = DataLoader(te_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)

    return tr_loader, val_loader, te_loader, num_classes



def get_cifar100_fe(fe_dat="vitb16-i21k", batch_size=256, num_workers=0, use_validation=True, dat_per_cls=-1):
    if dat_per_cls >= 0:
        data_path= f"/data1/lsj9862/data/cifar100/{fe_dat}/{dat_per_cls}_shot/"
    else:
        data_path = f"/data1/lsj9862/data/cifar100/{fe_dat}/full_shot"
    
    
    data_path = f"/mlainas/lsj9862/data/cifar100_{fe_dat}_fe"
    tr_x = torch.load(f"{data_path}/tr_x.pt", map_location='cuda'); tr_y = torch.load(f"{data_path}/tr_y.pt", map_location='cuda')
    val_x = torch.load(f"{data_path}/val_x.pt", map_location='cuda'); val_y = torch.load(f"{data_path}/val_y.pt", map_location='cuda')
    te_x = torch.load(f"{data_path}/te_x.pt", map_location='cuda'); te_y = torch.load(f"{data_path}/te_y.pt", map_location='cuda')
    tr_data = ExtractedDataSet(tr_x, tr_y)
    val_data = ExtractedDataSet(val_x, val_y)
    te_data = ExtractedDataSet(te_x, te_y)
    
    tr_loader = torch.utils.data.DataLoader(tr_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=False)
    
    te_loader = torch.utils.data.DataLoader(te_data,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        pin_memory=False)
    
    if use_validation:
        val_loader = torch.utils.data.DataLoader(val_data,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers,
                                        pin_memory=False)
    else:
        print("[Warning] You are going to run models on the test set.")
        val_loader = te_loader
    
    num_classes = max(te_data.y) + 1

    return tr_loader, val_loader, te_loader, num_classes

####################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------|
