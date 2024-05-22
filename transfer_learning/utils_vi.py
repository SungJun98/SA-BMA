import numpy as np
import torch


def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def get_vi_mean_vector(model):
    """
    Get Mean parameters in Variational Inference model
    """
    mean_list = []
    for name, param in model.named_parameters():
        if "rho" not in name:
            mean_list.append(param.cpu())
    return flatten(mean_list)
            
            
def get_vi_variance_vector(model):
    """
    Get (Diagonal) Variance Parameters in Variatioanl Inference model
    """
    var_list = []
    for name, param in model.named_parameters():
        if "rho" in name:            
            var_list.append(torch.log(1+torch.exp(param.cpu())))  # rho to variance
        elif ("mu" not in name) and ("rho" not in name):
            var_list.append(torch.zeros_like(param.cpu()))
    return flatten(var_list)


def make_ll_vi(args, model):
    from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
    bayesian_last_layer = torch.nn.Sequential(list(model.children())[-1])
    const_bnn_prior_parameters = {
        "prior_mu": args.vi_prior_mu,
        "prior_sigma": args.vi_prior_sigma,
        "posterior_mu_init": args.vi_posterior_mu_init,
        "posterior_rho_init": args.vi_posterior_rho_init,
        "type": args.vi_type,
        "moped_enable": True,
        "moped_delta": args.vi_moped_delta,
    }
    dnn_to_bnn(bayesian_last_layer, const_bnn_prior_parameters)
    if 'resnet' in args.model:
        raise NotImplementedError("Please check the name(module) of last layer and correct this code")
        model.fc = bayesian_last_layer.to(args.device)
    elif 'vitb16-i1k' == args.model:
        raise NotImplementedError("Please check the name(module) of last layer and correct this code")
        model.head = bayesian_last_layer.to(args.device)
    else:
        raise NotImplementedError()