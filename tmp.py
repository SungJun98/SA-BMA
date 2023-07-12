# %%
import torch
import utils.utils as utils

# %%
MODEL='resnet18'
NUM_CLASSES=10
PRE_TRAINED=True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
model = utils.get_backbone(MODEL, NUM_CLASSES, DEVICE, PRE_TRAINED)
"""
전체 paramter 개수 : 11181642
"""


# %%
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
const_bnn_prior_parameters = {
    "prior_mu": 0.0,
    "prior_sigma": 1.0,
    "posterior_mu_init": 0.0,
    "posterior_rho_init": -3.0,
    "type": "Reparameterization",
    "moped_enable": True,
    "moped_delta": 0.2,
}
dnn_to_bnn(model, const_bnn_prior_parameters)
model.to(DEVICE)
    
# %%
mu_param=0; rho_param=0; res_param=0
for name, param in model.named_parameters():
    if ("rho" not in name) and ("mu" not in name):
        res_param += param.numel()   # 9600
    elif "mu" in name:
        mu_param += param.numel()   # 11172042
    elif "rho" in name:
        rho_param += param.numel()  # 11172042
      

# %%        
"""
그럼 res_param에 대해서 mean, var은 어떻게 해줄까?
mean은 그냥 weight, bias 값 그대로 가져가면 되고
var은 rand_initialization하자.
근데 flatten하는 순서도 중요..!
즉, mean, var을 만들고 flatten해야 swag에서의 flatten과 일치할 것 같다!
-----
저장할 떄
1. weight, bias로 되어있는 애들(res_param)의 mean, var를 다 만들어놓자
2. weight를 list로 받기 (swag_utils.flatten 사용 목적)
3. flatten
4. mean, var 저장
"""

# %% 
DELTA = 0.2

mean_param_list = list(); var_param_list = list()

for name, param in model.named_parameters():
    if "mu" in name:
        mean_param_list.append(param)
    elif "rho" in name:
        var_param_list.append(param)
    else:
        mean_param_list.append(param)
        var_param_list.append(torch.log(torch.expm1(DELTA*torch.abs(param)) + 1e-20))

# %%
