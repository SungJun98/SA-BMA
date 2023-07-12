import torch
from ..swag.swag_utils import flatten

def get_vi_mean_vector(model):
    mean_list = []
    for name, param in model.named_parameters():
        if "rho" not in name:
            mean_list.append(param)
            
    return flatten(mean_list)
            
            
def get_vi_variance_vector(model, delta=0.2):
    # get rho
    var_list = []
    for name, param in model.named_parameters():
        if "rho" in name:
            var_list.append(param)
        elif ("mu" not in name) and ("rho" not in name):
            var_list.append(torch.log(torch.expm1(delta*torch.abs(param)) + 1e-20))
    var_list = flatten(var_list)
    
    # rho to variance
    var_list = torch.log(1+torch.exp(var_list))
    
    return var_list
