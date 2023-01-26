import random

import torch
from torch.utils.data import random_split

import torchvision.transforms as transforms
import torchvision

import utils

def get_mnist_source(data_path='/DATA1/lsj9862/mnist', batch_size=64, num_workers=4, use_validation=False, val_size=5000):
    
    # Use 0~4 class for pre-training
    transform = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))])
    tr_data = torchvision.datasets.MNIST(data_path, train=True, transform=transform,
                                            download=True)
    target_idx = torch.arange(5).unsqueeze(1)
    target_map = (target_idx == tr_data.targets).float()
    idx_0 = (target_map[0].nonzero(as_tuple=True)[0])
    idx_1 = (target_map[1].nonzero(as_tuple=True)[0])
    idx_2 = (target_map[2].nonzero(as_tuple=True)[0])
    idx_3 = (target_map[3].nonzero(as_tuple=True)[0])
    idx_4 = (target_map[4].nonzero(as_tuple=True)[0])
    idx = torch.cat([idx_0, idx_1, idx_2, idx_3, idx_4])
    tr_data_source = [tr_data[i] for i in idx]

    if use_validation:
        random.shuffle(tr_data_source)
        val_data_source = tr_data_source[-val_size:]
        tr_data_source = tr_data_source[:-val_size]

        val_loader = torch.utils.data.DataLoader(val_data_source,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers,
                                        pin_memory=True)
    else:
        print("[Warning] You are going to run models on the test set.")
        val_loader= None

    tr_loader = torch.utils.data.DataLoader(tr_data_source,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers,
                                        pin_memory=True)


    te_data = torchvision.datasets.MNIST(data_path, train=False, transform = transform,
                                            download=True)
    target_idx = torch.arange(5).unsqueeze(1)
    target_map = (target_idx == te_data.targets).float()
    idx_0 = (target_map[0].nonzero(as_tuple=True)[0])
    idx_1 = (target_map[1].nonzero(as_tuple=True)[0])
    idx_2 = (target_map[2].nonzero(as_tuple=True)[0])
    idx_3 = (target_map[3].nonzero(as_tuple=True)[0])
    idx_4 = (target_map[4].nonzero(as_tuple=True)[0])
    idx = torch.cat([idx_0, idx_1, idx_2, idx_3, idx_4])
    te_data_source = [te_data[i] for i in idx]
    
    te_loader = torch.utils.data.DataLoader(te_data_source,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        pin_memory=True)

    num_classes = 5

    return tr_loader, val_loader, te_loader, num_classes


def get_mnist_down(data_path='/DATA1/lsj9862/mnist', batch_size=64, num_workers=4, dat_per_cls=16, use_validation=False, val_size=5000):

    # Use 5~9 class for fine-tuning stage
    transform = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))])
    tr_data = torchvision.datasets.MNIST(data_path, train=True, transform=transform,
                                            download=True)
    target_idx = torch.arange(10).unsqueeze(1)
    target_map = (target_idx == tr_data.targets).float()
    idx_5 = (target_map[5].nonzero(as_tuple=True)[0])
    idx_6 = (target_map[6].nonzero(as_tuple=True)[0])
    idx_7 = (target_map[7].nonzero(as_tuple=True)[0])
    idx_8 = (target_map[8].nonzero(as_tuple=True)[0])
    idx_9 = (target_map[9].nonzero(as_tuple=True)[0])

    ## Train
    tr_idx = torch.cat([idx_5[:dat_per_cls],
                    idx_6[:dat_per_cls], 
                    idx_7[:dat_per_cls],
                    idx_8[:dat_per_cls],
                    idx_9[:dat_per_cls]])
    tr_data_down = [(tr_data[i][0], tr_data[i][1]-5) for i in tr_idx] # change 5~9 labels to 0~4
    
    tr_loader = torch.utils.data.DataLoader(tr_data_down,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers,
                                        pin_memory=True)

    ## Validation
    if use_validation:
        # random.shuffle(tr_data_down)
        # val_data_down = tr_data_down[-val_size:]
        # tr_data_down = tr_data_down[:-val_size]
        val_idx = torch.cat([idx_5[dat_per_cls:dat_per_cls+val_size],
                    idx_6[dat_per_cls:dat_per_cls+val_size], 
                    idx_7[dat_per_cls:dat_per_cls+val_size],
                    idx_8[dat_per_cls:dat_per_cls+val_size],
                    idx_9[dat_per_cls:dat_per_cls+val_size]])
        val_data_down = [(tr_data[i][0], tr_data[i][1]-5) for i in val_idx] # change 5~9 labels to 0~4

        val_loader = torch.utils.data.DataLoader(val_data_down,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers,
                                        pin_memory=True)
    else:
        print("[Warning] You are going to run models on the test set.")
        val_loader= None

    ## Test
    te_data = torchvision.datasets.MNIST(data_path, train=False, transform = transform,
                                            download=True)
    target_idx = torch.arange(10).unsqueeze(1)
    target_map = (target_idx == te_data.targets).float()
    idx_5 = (target_map[5].nonzero(as_tuple=True)[0])
    idx_6 = (target_map[6].nonzero(as_tuple=True)[0])
    idx_7 = (target_map[7].nonzero(as_tuple=True)[0])
    idx_8 = (target_map[8].nonzero(as_tuple=True)[0])
    idx_9 = (target_map[9].nonzero(as_tuple=True)[0])
    idx = torch.cat([idx_5, idx_6, idx_7, idx_8, idx_9])
    te_data_down = [(te_data[i][0], te_data[i][1]-5) for i in idx] # change 5~9 labels to 0~4
    
    te_loader = torch.utils.data.DataLoader(te_data_down,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        pin_memory=True)    
                                    
    num_classes = 5

    return tr_loader, val_loader, te_loader, num_classes



def get_cifar10(data_path='/DATA1/lsj9862/cifar10', batch_size=64, num_workers=4, use_validation=False, val_ratio=0.1):
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
    tr_data = torchvision.datasets.CIFAR10(data_path, train=True, transform=transform_train,
                                            download=True)
    te_data = torchvision.datasets.CIFAR10(data_path, train=False, transform = transform_test,
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


'''
cifar100 입주 예정
'''