# %%
import torch
import torch.nn as nn
import os
import utils.utils as utils
import numpy as np

# %%
## Settings
MODEL_NAME = 'resnet18-noBN'
DATASET = 'cifar100'
DATA_PATH = f'/data2/lsj9862/data/{DATASET}'
BATCH_SIZE = 256
RESUME = "/mlainas/lsj9862/exp_result/cifar100/resnet18-noBN/dnn-sgd/swag_lr/dnn-sgd_best_val.pt"
SAVE_PATH = "/mlainas/lsj9862/exp_result/cifar100/resnet18-noBN/lswag-sam/swag_lr/fe_dat"
DAT_PER_CLS = -1
SEED = 0
AUG = False


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
utils.set_seed(SEED)


# %%
## Load data
tr_loader, val_loader, te_loader, num_classes = utils.get_dataset(dataset=DATASET,
                                                            data_path=DATA_PATH,
                                                            batch_size=BATCH_SIZE,
                                                            num_workers=4,
                                                            use_validation=True,
                                                            aug=AUG,
                                                            dat_per_cls = DAT_PER_CLS,
                                                            seed = SEED)
# %%
## Define model
model = utils.get_backbone(MODEL_NAME,
                    num_classes = num_classes,
                    device = device,
                    pre_trained = True,
                    last_layer = False)

if RESUME is not None:
    checkpoint = torch.load(RESUME)
    model.load_state_dict(checkpoint['state_dict'])
        
    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()
            
        def forward(self, x):
            return x
        
    if MODEL_NAME == 'vitb16-i21k':
        model.head = Identity()
    elif MODEL_NAME in ['resnet18', 'resnet18-noBN', 'resnet50', 'resnet50-noBN', 'resnet101', 'resnet101-noBN']:
        model.fc = Identity()
    elif MODEL_NAME in ["wideresnet28x10", "wideresnet28x10-noBN"]:
        raise ValueError("Add code for this")


# %%
## Set save path
import os
os.makedirs(SAVE_PATH, exist_ok=True)


# %%
## Training data
feature_list = [] 
target_list = []

model.eval()
with torch.no_grad():
    for _, (input, target) in enumerate(tr_loader):
        input, target = input.to(device), target.to(device)
        pred = model(input)
        feature_list.append(pred)
        target_list.append(target)
    
feature_list = torch.concat(feature_list, dim=0)
target_list = torch.concat(target_list, dim=0)

torch.save(feature_list, f"{SAVE_PATH}/tr_fe_x.pt")
torch.save(target_list, f"{SAVE_PATH}/tr_fe_y.pt")

# %%
## Validation data
feature_list = [] 
target_list = []

model.eval()
with torch.no_grad():
    for _, (input, target) in enumerate(val_loader):
        input, target = input.to(device), target.to(device)
        pred = model(input)
        feature_list.append(pred)
        target_list.append(target)
    
feature_list = torch.concat(feature_list, dim=0)
target_list = torch.concat(target_list, dim=0)

torch.save(feature_list, f"{SAVE_PATH}/val_fe_x.pt")
torch.save(target_list, f"{SAVE_PATH}/val_fe_y.pt")

# %%
## Test data
feature_list = [] 
target_list = []

model.eval()
with torch.no_grad():
    for _, (input, target) in enumerate(te_loader):
        input, target = input.to(device), target.to(device)
        pred = model(input)
        feature_list.append(pred)
        target_list.append(target)
    
feature_list = torch.concat(feature_list, dim=0)
target_list = torch.concat(target_list, dim=0)

torch.save(feature_list, f"{SAVE_PATH}/te_fe_x.pt")
torch.save(target_list, f"{SAVE_PATH}/te_fe_y.pt")

# %%
