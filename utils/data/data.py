import random

import torch
import torchvision
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100, FGVCAircraft, StanfordCars
import torchvision.transforms as transforms

from timm.data.transforms_factory import create_transform

from torch.utils.data import Dataset


class ExtractedDataSet(Dataset): 
    """
    Class for Feature Extracted data
    """
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
def get_cifar10(data_path='/data2/lsj9862/data/cifar10', batch_size=256, num_workers=4, use_validation=False, val_ratio=0.1, aug=True):
    if aug:
        transform_train = create_transform(224, is_training=True)
        transform_test = create_transform(224)
    else:
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



def get_cifar10_vitb16_fe(data_path="/mlainas/lsj9862/data/cifar10_vitb16-i21k_fe", batch_size=256, num_workers=0, use_validation=True):
    tr_x = torch.load(f"{data_path}/tr_x.pt"); tr_y = torch.load(f"{data_path}/tr_y.pt")
    val_x = torch.load(f"{data_path}/val_x.pt"); val_y = torch.load(f"{data_path}/val_y.pt")
    te_x = torch.load(f"{data_path}/te_x.pt"); te_y = torch.load(f"{data_path}/te_y.pt")
    
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
def get_cifar100(data_path='/data2/lsj9862/data/cifar100', batch_size=256, num_workers=4, use_validation=False, val_ratio=0.1, aug=True):
    if aug:
        transform_train = create_transform(224, is_training=True)
        transform_test = create_transform(224)
    else:
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



def get_cifar100_vitb16_fe(data_path="/mlainas/lsj9862/data/cifar100_vitb16-i21k_fe", batch_size=256, num_workers=0, use_validation=False):
    tr_x = torch.load(f"{data_path}/tr_x.pt"); tr_y = torch.load(f"{data_path}/tr_y.pt")
    val_x = torch.load(f"{data_path}/val_x.pt"); val_y = torch.load(f"{data_path}/val_y.pt")
    te_x = torch.load(f"{data_path}/te_x.pt"); te_y = torch.load(f"{data_path}/te_y.pt")
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
### AirCraft ---------------------------------------------------------------------------------------------------------------------|
def get_aircraft(data_path='/data2/lsj9862/data/aircraft', batch_size=256, num_workers=4, use_validation=False, val_ratio=0.1, aug=True):
    if aug:
        transform_train = create_transform(224, is_training=True)
        transform_test = create_transform(224)
    else:
        transform_train = transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
    tr_data = FGVCAircraft(data_path, split='train', transform=transform_train,
                                            download=True)
    te_data = FGVCAircraft(data_path, split='test', transform = transform_test,
                                            download=True)
    
    tr_loader = torch.utils.data.DataLoader(tr_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=True)
    if use_validation:
        val_data = FGVCAircraft(data_path, split='val', transform=transform_test,
                                            download=True)
        
        val_loader = torch.utils.data.DataLoader(val_data,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers,
                                        pin_memory=True)
    else:
        print("[Warning] You are going to run models on the test set.")

        val_loader= None

    te_loader = torch.utils.data.DataLoader(te_data,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        pin_memory=True)

    num_classes = max(te_data._labels) + 1

    return tr_loader, val_loader, te_loader, num_classes



def get_aircraft_vitb16_fe(data_path="/mlainas/lsj9862/data/aircraft_vitb16-i21k_fe", batch_size=256, num_workers=0, use_validation=False):
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
### NABirds -----------------------------------------------------------------------------------------------------------------------|
def get_nabirds(data_path='/data2/lsj9862/data/nabirds', batch_size=256, num_workers=4, use_validation=False, val_ratio=0.1, aug=True):
    from utils.data.nabirds import NABirds
    if aug:
        transform_train = create_transform(224, is_training=True)
        transform_test = create_transform(224)
    else:
        transform_train = transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
    tr_data = NABirds(data_path, train=True, transform=transform_train,
                                            download=False)
    te_data = NABirds(data_path, train=False, transform = transform_test,
                                            download=False)
    
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

    num_classes = len(te_data.class_names)

    return tr_loader, val_loader, te_loader, num_classes



def get_nabirds_vitb16_fe(data_path="/mlainas/lsj9862/data/nabirds_vitb16-i21k_fe", batch_size=256, num_workers=0, use_validation=False):
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
### Stanford Cars -----------------------------------------------------------------------------------------------------------------------|
def get_cars(data_path='/data1/lsj9862/data/stanfordcars', batch_size=256, num_workers=4, use_validation=False, val_ratio=0.1, aug=True):
    if aug:
        transform_train = create_transform(224, is_training=True)
        transform_test = create_transform(224)
    else:
        transform_train = transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
    tr_data = StanfordCars(data_path, split="train", transform=transform_train,
                                            download=True)
    te_data = StanfordCars(data_path, split="test", transform = transform_test,
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

    import pdb; pdb.set_trace()
    num_classes = len(te_data.loaded_mat)

    return tr_loader, val_loader, te_loader, num_classes



def get_cars_vitb16_fe(data_path="/mlainas/lsj9862/data/nabirds_vitb16-i21k_fe", batch_size=256, num_workers=0, use_validation=False):
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