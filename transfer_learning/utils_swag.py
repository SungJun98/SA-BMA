import numpy as np
import pickle, wandb
import os, itertools, tqdm

import matplotlib.pyplot as plt
import collections, tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)

def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    
    return outList
    
def unflatten_like_size(vector, likeTensorSize):
    """
    Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    Input
     - vector : flattened parameters
     - likeTensorSize : list of torch.Size
    """
    outList = []
    i = 0
    for layer_size in likeTensorSize:
        n = layer_size.numel()
        outList.append(vector[i : i + n].view(layer_size))
        i += n

    return outList



class SWAG(torch.nn.Module):
    def __init__(
        self, base, no_cov_mat=False, max_num_models=20, var_clamp=1e-30,
        last_layer=False
    ):
        super(SWAG, self).__init__()

        self.register_buffer("n_models", torch.zeros([1], dtype=torch.long))
        self.params = list()

        self.no_cov_mat = no_cov_mat
        self.max_num_models = max_num_models

        self.var_clamp = var_clamp
        self.last_layer= last_layer

        for name, _ in base.named_modules():
            self.last_layer_name = name
        
        self.num_params = 0
        for name, param in base.named_parameters():
            if name.split('.')[0] in self.last_layer_name:
                self.num_params += param.numel()
        
        self.base = base
        self.init_swag_parameters(params=self.params)


    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)


    def init_swag_parameters(self, params):
        for mod_name, module in self.base.named_modules():
            if not self.last_layer:
                for name in list(module._parameters.keys()):
                    if module._parameters[name] is None:
                        continue

                    name_full = f"{mod_name}.{name}".replace(".", "-")
                    data = module._parameters[name].data
                    module._parameters.pop(name)
                    module.register_buffer("%s_mean" % name_full, data.new(data.size()).zero_())
                    module.register_buffer("%s_sq_mean" % name_full, data.new(data.size()).zero_())

                    if self.no_cov_mat is False:
                        module.register_buffer("%s_cov_mat_sqrt" % name_full, data.new_empty((0, data.numel())).zero_())

                    params.append((module, name_full))
            
            else:
                for name in list(module._parameters.keys()):
                    if module._parameters[name] is None:
                        continue
                    
                    if mod_name == self.last_layer_name:
                        name_full = f"{mod_name}.{name}".replace(".", "-")
                        data = module._parameters[name].data
                        module._parameters.pop(name)
                        module.register_buffer("%s_mean" % name_full, data.new(data.size()).zero_())
                        module.register_buffer("%s_sq_mean" % name_full, data.new(data.size()).zero_())

                        if self.no_cov_mat is False:
                            module.register_buffer("%s_cov_mat_sqrt" % name_full, data.new_empty((0, data.numel())).zero_())

                        params.append((module, name_full))
                         
                
    
    def get_mean_vector(self):
        mean_list = []
        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)
            mean_list.append(mean.cpu())
        return flatten(mean_list)


    def get_variance_vector(self):
        mean_list = []
        sq_mean_list = []

        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            mean_list.append(mean.cpu())
            sq_mean_list.append(sq_mean.cpu())

        mean = flatten(mean_list)
        sq_mean = flatten(sq_mean_list)

        variances = torch.clamp(sq_mean - mean ** 2, self.var_clamp)

        return variances


    def get_covariance_matrix(self, eps=1e-10):
        if self.no_cov_mat:
            raise RuntimeError("No covariance matrix was estimated!")

        cov_mat_sqrt_list = []
        for module, name in self.params:
            cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
            cov_mat_sqrt_list.append(cov_mat_sqrt.cpu())

        return cov_mat_sqrt_list


    def sample(self, scale=0.5, cov=True, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        mean_list = []
        sq_mean_list = []
        if cov:
            cov_mat_sqrt_list = []

        for (module, name) in self.params:
            mean_list.append(module.__getattr__("%s_mean" % name).cpu())
            sq_mean_list.append(module.__getattr__("%s_sq_mean" % name).cpu())
            if cov:
                cov_mat_sqrt_list.append(module.__getattr__("%s_cov_mat_sqrt" % name).cpu())

        mean = flatten(mean_list)
        sq_mean = flatten(sq_mean_list)

        # draw diagonal variance sample
        var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)
        if self.last_layer:
            var[:-self.num_params] = 0.0
        rand_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        # if covariance draw low rank sample
        if cov:
            cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)
            if self.last_layer:
                cov_mat_sqrt[:, :-self.num_params] = 0.0
            eps = cov_mat_sqrt.new_empty((cov_mat_sqrt.size(0),), requires_grad=False).normal_()
            cov_sample = cov_mat_sqrt.t().matmul(eps)
            cov_sample /= (self.max_num_models - 1) ** 0.5
            rand_sample += cov_sample

        # update sample with mean and scale
        sample = (mean + scale**0.5 * rand_sample).unsqueeze(0)

        # unflatten new sample like the mean sample
        samples_list = unflatten_like(sample, mean_list)

        self.set_model_parameters(samples_list)

        return samples_list


    def set_model_parameters(self, parameter):
        for (module, name), param in zip(self.params, parameter):
            module.__setattr__(name.split("-")[-1], param.cuda())


    def collect_model(self, base_model):
        if not self.last_layer:
            for (module, name), base_param in zip(self.params, base_model.parameters()):    
                data = base_param.data

                mean = module.__getattr__("%s_mean" % name)
                sq_mean = module.__getattr__("%s_sq_mean" % name)
                
                # first moment  
                mean = mean * self.n_models.item() / (
                    self.n_models.item() + 1.0
                ) + data / (self.n_models.item() + 1.0)

                # second moment
                sq_mean = sq_mean * self.n_models.item() / (
                    self.n_models.item() + 1.0
                ) + data ** 2 / (self.n_models.item() + 1.0)

                # square root of covariance matrix
                if self.no_cov_mat is False:
                    cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)

                    # block covariance matrices, store deviation from current mean
                    dev = (data - mean)
                    name_full = name.replace("-", ".")
                    cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1, 1).t()), dim=0)

                    # remove first column if we have stored too many models
                    if (self.n_models.item() + 1) > self.max_num_models:
                        cov_mat_sqrt = cov_mat_sqrt[1:, :]
                    module.__setattr__("%s_cov_mat_sqrt" % name, cov_mat_sqrt)

                module.__setattr__("%s_mean" % name, mean)
                module.__setattr__("%s_sq_mean" % name, sq_mean)
        
        else:
            for mod_name, module in base_model.named_modules():
                if mod_name == self.last_layer_name:
                    classifier_params = module._parameters
                    
            for (module, name), base_param in zip(self.params, classifier_params.values()):   
                data = base_param.data

                mean = module.__getattr__("%s_mean" % name)
                sq_mean = module.__getattr__("%s_sq_mean" % name)
                
                # first moment
                mean = mean * self.n_models.item() / (
                    self.n_models.item() + 1.0
                ) + data / (self.n_models.item() + 1.0)

                # second moment
                sq_mean = sq_mean * self.n_models.item() / (
                    self.n_models.item() + 1.0
                ) + data ** 2 / (self.n_models.item() + 1.0)

                # square root of covariance matrix
                if self.no_cov_mat is False:
                    cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)

                    # block covariance matrices, store deviation from current mean
                    dev = (data - mean)
                    name_full = name.replace("-", ".")
                    cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1, 1).t()), dim=0)

                    # remove first column if we have stored too many models
                    if (self.n_models.item() + 1) > self.max_num_models:
                        cov_mat_sqrt = cov_mat_sqrt[1:, :]
                    module.__setattr__("%s_cov_mat_sqrt" % name, cov_mat_sqrt)

                module.__setattr__("%s_mean" % name, mean)
                module.__setattr__("%s_sq_mean" % name, sq_mean)

        self.n_models.add_(1)


    def load_state_dict(self, state_dict, strict=True):
        if not self.no_cov_mat:
            n_models = state_dict["n_models"].item()
            rank = min(n_models, self.max_num_models)
            for module, name in self.params:
                mean = module.__getattr__("%s_mean" % name)
                module.__setattr__(
                    "%s_cov_mat_sqrt" % name,
                    mean.new_empty((rank, mean.numel())).zero_(),
                )
        super(SWAG, self).load_state_dict(state_dict, strict)


            
    def import_numpy_weights(self, w):
        k = 0
        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)
            s = np.prod(mean.shape)
            module.__setattr__(name, mean.new_tensor(w[k : k + s].reshape(mean.shape)))
            k += s




def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)

        for batch in loader:
            input = batch['img']
            
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))

def apply_bn_update(model, momenta):
    model.apply(lambda module: _set_momenta(module, momenta))