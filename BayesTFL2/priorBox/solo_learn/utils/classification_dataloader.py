# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
'''
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder

import torch

def build_custom_pipeline():
    """Builds augmentation pipelines for custom data.
    If you want to do exoteric augmentations, you can just re-write this function.
    Needs to return a dict with the same structure.
    """

    pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
    }
    return pipeline


def prepare_transforms(train_dataset: str, val_dataset: str) -> Tuple[nn.Module, nn.Module]:
    """Prepares pre-defined train and test transformation pipelines for some datasets.

    Args:
        dataset (str): dataset name.

    Returns:
        Tuple[nn.Module, nn.Module]: training and validation transformation pipelines.
    """

    cifar_pipeline = {

        "T_train": transforms.Compose(
            [
                #transforms.Resize(224),
                #transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((224, 224)),  # resize shorter
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    stl_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
    }
    oxford102flowers_pipeline = {
        'T_train':transforms.Compose([
        transforms.Resize((230,230)),
        transforms.RandomRotation(30,),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]),
     'T_val':
         transforms.Compose([
             transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])
    }
    imagenet_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256, interpolation=3),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
    }

    custom_pipeline = build_custom_pipeline()

    pipelines = {
        "cifar10": cifar_pipeline,
        "cifar100": cifar_pipeline,
        "stl10": stl_pipeline,
        "imagenet100": imagenet_pipeline,
        "imagenet": imagenet_pipeline,
        "custom": custom_pipeline,
        'oxford102flowers': oxford102flowers_pipeline,
        'oxfordpets': custom_pipeline
    }

    assert train_dataset in pipelines

    pipeline = pipelines[train_dataset]
    T_train = pipeline["T_train"]
    pipeline_val = pipelines[val_dataset]
    T_val = pipeline_val["T_val"]

    return T_train, T_val


def prepare_datasets(
    train_dataset_name: str,
    val_dataset_name: str,
    T_train: Callable,
    T_val: Callable,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    val_dir: Optional[Union[str, Path]] = None,
    download: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Prepares train and val datasets.

    Args:
        dataset (str): dataset name.
        T_train (Callable): pipeline of transformations for training dataset.
        T_val (Callable): pipeline of transformations for validation dataset.
        data_dir Optional[Union[str, Path]]: path where to download/locate the dataset.
        train_dir Optional[Union[str, Path]]: subpath where the training data is located.
        val_dir Optional[Union[str, Path]]: subpath where the validation data is located.

    Returns:
        Tuple[Dataset, Dataset]: training dataset and validation dataset.
    """

    if data_dir is None:
        sandbox_dir = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        data_dir = sandbox_dir / "datasets"
    else:
        data_dir = Path(data_dir)

    if train_dir is None:
        train_dir = Path(f"{train_dataset_name}/train")
    else:
        train_dir = Path(train_dir)

    if val_dir is None:
        val_dir = Path(f"{val_dataset_name}/val")
    else:
        val_dir = Path(val_dir)

    assert train_dataset_name in ["cifar10", "cifar100", "stl10", "imagenet", "imagenet100", "custom", 'oxford102flowers', 'oxfordpets']

    if train_dataset_name in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[train_dataset_name.upper()]
        train_dataset = DatasetClass(
            data_dir / train_dir,
            train=True,
            download=download,
            transform=T_train,
        )
    if val_dataset_name in ["cifar10", "cifar100"]:
        val_dataset = DatasetClass(
            data_dir / val_dir,
            train=False,
            download=download,
            transform=T_val,
        )

    if train_dataset_name == "stl10":
        train_dataset = STL10(
            data_dir / train_dir,
            split="train",
            download=True,
            transform=T_train,
        )
    if val_dataset_name == "stl10":
        val_dataset = STL10(
            data_dir / val_dir,
            split="test",
            download=download,
            transform=T_val,
        )

    if train_dataset_name in ["imagenet", "imagenet100", "custom", 'oxford102flowers', 'oxfordpets']:
        train_dir = data_dir / train_dir
        train_dataset = ImageFolder(train_dir, T_train)
    if val_dataset_name in ["imagenet", "imagenet100", "custom",  'oxford102flowers', 'oxfordpets']:

        val_dir = data_dir / val_dir

        val_dataset = ImageFolder(val_dir, T_val)

    return train_dataset, val_dataset


def prepare_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Wraps a train and a validation dataset with a DataLoader.

    Args:
        train_dataset (Dataset): object containing training data.
        val_dataset (Dataset): object containing validation data.
        batch_size (int): batch size.
        num_workers (int): number of parallel workers.
    Returns:
        Tuple[DataLoader, DataLoader]: training dataloader and validation dataloader.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def prepare_data(
    train_dataset: str,
    val_dataset: str,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    val_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Prepares transformations, creates dataset objects and wraps them in dataloaders.

    Args:
        dataset (str): dataset name.
        data_dir (Optional[Union[str, Path]], optional): path where to download/locate the dataset.
            Defaults to None.
        train_dir (Optional[Union[str, Path]], optional): subpath where the
            training data is located. Defaults to None.
        val_dir (Optional[Union[str, Path]], optional): subpath where the
            validation data is located. Defaults to None.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of parallel workers. Defaults to 4.

    Returns:
        Tuple[DataLoader, DataLoader]: prepared training and validation dataloader;.
    """
    

    T_train, T_val = prepare_transforms(train_dataset, val_dataset)
    train_dataset, val_dataset = prepare_datasets(
        train_dataset,
        val_dataset,
        T_train, #transform
        T_val,
        data_dir=data_dir,
        train_dir=train_dir, # -no_aug 로도 해봤으나 결과는 같게 나왔다...
        val_dir=val_dir,
        download=download,
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    train_loader = torch.load("/home/emforce77/BayesianTransferLearning-main/cifar10_10shot/tr_loader_seed0.pth",     #/home/emforce77/BayesianTransferLearning-main/cifar10_10shot/tr_loader_seed0.pth <= cifar10
                         map_location=torch.device('cpu'))

    '''
# %%
import sys, os
import torch
import numpy as np
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from timm.data.transforms_factory import create_transform

import argparse


import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Any

##

# # %%
# dataset='cifar10'
# data_path=f'/home/emforce77/BayesTFL/priorBox/solo_learn/datasets'
# #batch_size=256
# use_validation=True
# aug=True
# val_ratio=0.1

# dataset = "dtd"


# seed=2        #0,1,2
# dat_per_cls=10   #1, 10 ,100, 1000, -1


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

def get_dataset_dassl(args):
    from my_dassl.data import DataManager
    from my_dassl.config import get_cfg_default
    
    import my_dassl.datasets.oxford_pets
    import my_dassl.datasets.oxford_flowers
    import my_dassl.datasets.fgvc_aircraft
    import my_dassl.datasets.dtd
    import my_dassl.datasets.eurosat
    import my_dassl.datasets.stanford_cars
    import my_dassl.datasets.food101
    import my_dassl.datasets.sun397
    import my_dassl.datasets.caltech101
    import my_dassl.datasets.ucf101
    import my_dassl.datasets.imagenet
    import my_dassl.datasets.svhn
    import my_dassl.datasets.resisc45
    import my_dassl.datasets.clevr

    import my_dassl.datasets.locmnist
    
    cfg = get_cfg_default()
    
    dataset_config_file = f'./my_dassl/datasets/config/{args.dataset}.yaml'
    cfg.SEED = args.seed
    cfg.merge_from_file(dataset_config_file)
    cfg.DATASET.ROOT = args.data_path
    cfg.DATASET.NUM_SHOTS = args.dat_per_cls
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    
    dm = DataManager(cfg)
    tr_loader = dm.train_loader_x
    val_loader = dm.val_loader
    te_loader = dm.test_loader
    num_classes = dm.num_classes
    
    return tr_loader, val_loader, te_loader, num_classes


# if dataset == 'cifar10' or dataset == 'cifar100':

#     def prepare_data(
#         train_dataset: str,
#         val_dataset: str,
#         data_dir: Optional[Union[str, Path]] = None,
#         train_dir: Optional[Union[str, Path]] = None,
#         val_dir: Optional[Union[str, Path]] = None,
#         batch_size: int = 256,
#         num_workers: int = 4,
#         download: bool = True,
#     ) -> Tuple[DataLoader, DataLoader, int, int]:

#         #dataset='cifar100'
#         data_path=f'/home/emforce77/BayesTFL/priorBox/solo_learn/datasets'
#         batch_size=64
#         use_validation=True
#         aug=True
#         val_ratio=0.1

#         set_seed(seed)
#         print(seed)


#         ## Get transform
#         if aug:
#             transform_train = create_transform(224, is_training=True)
#             transform_test = create_transform(224)
#         else:
#             transform_train = transforms.Compose([
#                             transforms.RandomCrop(32, padding=4),
#                             transforms.RandomHorizontalFlip(),
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                         ])

#             transform_test = transforms.Compose([
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                         ])        

#         ## Load Data
#         if dataset=='cifar10':
#             tr_data = CIFAR10(data_path, train=True, transform=transform_train,
#                                                     download=True)
#             te_data = CIFAR10(data_path, train=False, transform = transform_test,
#                                                     download=True)
#         elif dataset=='cifar100':
#             tr_data = CIFAR100(data_path, train=True, transform=transform_train,
#                                                     download=True)
#             te_data = CIFAR100(data_path, train=False, transform = transform_test,
#                                                     download=True)
            
#         num_classes = max(te_data.targets) + 1
#         print(f"Number of classes : {num_classes} :: Data point per class : {dat_per_cls}")


        


#         ## Make Loader

#         file_path = f"/home/emforce77/BayesTFL/fewshotset/{dataset}_{dat_per_cls}_shot_tr_loader_seed{seed}.pth"

#         if os.path.exists(file_path):

#             train_loader  = torch.load(f"/home/emforce77/BayesTFL/fewshotset/{dataset}_{dat_per_cls}_shot_tr_loader_seed{seed}.pth")

#             val_loader  = torch.load(f"/home/emforce77/BayesTFL/fewshotset/{dataset}_{dat_per_cls}_shot_val_loader_seed{seed}.pth")
            
#             test_loader  = torch.load(f"/home/emforce77/BayesTFL/fewshotset/{dataset}_{dat_per_cls}_shot_test_loader_seed{seed}.pth")


#         else:

#             ## Split Data
#             val_len = int(len(tr_data) * val_ratio)
#             tr_len = len(tr_data) - val_len

#             tr_data, val_data = random_split(tr_data, [tr_len, val_len])

#             ## Pre-Setting for Few-shot Setting
#             class_indices = [[] for _ in range(num_classes)]
#             for idx, (_, target) in enumerate(tr_data):
#                 class_indices[target].append(idx)

#             few_shot_indices = []
#             for indices in class_indices:
#                 few_shot_indices.extend(indices[:dat_per_cls])

#             sampler = SubsetRandomSampler(few_shot_indices)


#             train_loader = tr_loader = DataLoader(tr_data,
#                             batch_size=batch_size,
#                             num_workers=4,
#                             pin_memory=True,
#                             sampler=sampler)

#             val_loader =  DataLoader(val_data, # 
#                                     batch_size=batch_size,
#                                     shuffle=False,
#                                     num_workers=num_workers,
#                                     pin_memory=True,
#                                     )
            
#             test_loader =  DataLoader(te_data, # 
#                                     batch_size=batch_size,
#                                     shuffle=False,
#                                     num_workers=num_workers,
#                                     pin_memory=True,
#                                     )
#             print(train_loader.batch_size)

#             torch.save(tr_loader, f'/home/emforce77/BayesTFL/fewshotset/{dataset}_{dat_per_cls}_shot_tr_loader_seed{seed}.pth')
#             torch.save(val_loader, f'/home/emforce77/BayesTFL/fewshotset/{dataset}_{dat_per_cls}_shot_val_loader_seed{seed}.pth')
#             torch.save(test_loader, f'/home/emforce77/BayesTFL/fewshotset/{dataset}_{dat_per_cls}_shot_test_loader_seed{seed}.pth')
            

#         return train_loader, val_loader, len(tr_data),  len(train_loader), test_loader 
    
def prepare_data(
    train_dataset: str,
    val_dataset: str,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    val_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 256,
    num_workers: int = 4,
    download: bool = True,
    args: Any = None,
) -> Tuple[DataLoader, DataLoader, int, int]:
    
    # class Args:
    #     def __init__(self, dataset, seed, data_path, dat_per_cls): #
    #         self.dataset = dataset
    #         self.seed = seed
    #         self.data_path = data_path
    #         self.dat_per_cls = dat_per_cls


    # args = Args(
    #     dataset="dtd",
    #     seed=0,
    #     data_path="/home/emforce77/BayesTFL/data",
    #     dat_per_cls=47)


    
    tr_loader, val_loader, te_loader, num_classes = get_dataset_dassl(args)

    return tr_loader, val_loader, len(tr_loader.dataset),  len(tr_loader), te_loader


    
