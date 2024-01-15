# %%
import torch
import utils.utils as utils

# %%
MODEL='resnet18'
DATASET = "cifar100"
DAT_PER_CLS = 100
PRE_TRAINED=True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED=2

utils.set_seed(SEED)
# %%
## Load Data
tr_loader, val_loader, te_loader, num_classes = utils.get_dataset(dataset=DATASET,
                                                        data_path='/data1/lsj9862/data/cifar10',
                                                        dat_per_cls=DAT_PER_CLS,
                                                        use_validation=True,
                                                        batch_size=256,
                                                        num_workers=4,
                                                        seed=0,
                                                        aug=True,
                                                        )


# %%
## Load Model
model = utils.get_backbone(MODEL, num_classes, DEVICE, PRE_TRAINED)



# %%
from laplace import Laplace

# Laplace
la = Laplace(model, 'classification',
             subset_of_weights='all',
             hessian_structure='lowrank')
la.fit(tr_loader)
# la.optimize_prior_precision(method='marglik')

# %%
from utils.swag.swag_utils import flatten

# mean value 뽑아내기
mean_list = []
for param in model.parameters():
    mean_list.append(param.cpu())
mean = flatten(mean_list)

# variance value 뽑아내기
variance = 1 / la.posterior_precision[1]

# covariacne value 뽑아내기 (low-rank)
cov_sqrt = torch.diag(torch.sqrt(1/la.posterior_precision[0][1])).cpu()
cov_sqrt = cov_sqrt.matmul((1/la.posterior_precision[0][0].cpu()).t())

# %%
# BMA test code 넣기
import numpy as np
def predict(loader, model):
    preds = list()
    targets = list()

    offset = 0
    with torch.no_grad():
        for input, target in loader:
            input = input.cuda(non_blocking=True)
            output = model(input)

            batch_size = input.size(0)
            preds.append(output)
            targets.append(target.cpu().numpy())
            offset += batch_size

    return {"predictions": np.vstack(preds), "targets": np.concatenate(targets)}