# %%
import torch
import torch.nn as nn
import os
import utils.utils as utils
import numpy as np

# %%
## Settings
MODEL_NAME = 'vitb16-i21k'
DATASET = 'nabirds'
DATA_PATH = f'/data1/lsj9862/data/{DATASET}'
BATCH_SIZE = 256

# %%
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
utils.set_seed(0)


# %%
## Load data
tr_loader, val_loader, te_loader, num_classes = utils.get_dataset(dataset=DATASET,
                                                            data_path=DATA_PATH,
                                                            batch_size=BATCH_SIZE,
                                                            num_workers=4,
                                                            use_validation=True,
                                                            aug=True,
                                                            fe_dat=None)
# %%
## Define model
model = utils.get_backbone(MODEL_NAME, num_classes, device, pre_trained=True)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

if MODEL_NAME == 'vitb16-i21k':
    model.head = Identity()
elif MODEL_NAME in ['resnet18', 'resnet18-noBN', 'resnet50', 'resnet50-noBN']:
    model.fc = Identity()

# %%
## Set save path
import os
os.makedirs(f"/mlainas/lsj9862/data/{DATASET}_{MODEL_NAME}_fe/", exist_ok=True)

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

torch.save(feature_list, f"/mlainas/lsj9862/data/{DATASET}_{MODEL_NAME}_fe/tr_x.pt")
torch.save(target_list, f"/mlainas/lsj9862/data/{DATASET}_{MODEL_NAME}_fe/tr_y.pt")

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

torch.save(feature_list, f"/mlainas/lsj9862/data/{DATASET}_{MODEL_NAME}_fe/val_x.pt")
torch.save(target_list, f"/mlainas/lsj9862/data/{DATASET}_{MODEL_NAME}_fe/val_y.pt")

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

torch.save(feature_list, f"/mlainas/lsj9862/data/{DATASET}_{MODEL_NAME}_fe/te_x.pt")
torch.save(target_list, f"/mlainas/lsj9862/data/{DATASET}_{MODEL_NAME}_fe/te_y.pt")

# %%
