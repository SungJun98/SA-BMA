import numpy as np
import pickle, wandb
import os, itertools, tqdm

import matplotlib.pyplot as plt
import collections, tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets.common import maybe_dictionarize

## ------------------------------------------------------------------------------------
## Setting Configs --------------------------------------------------------------------
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



def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


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


def bn_update(loader, model, input_key, verbose=False, subset=None, **kwargs):
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
        if verbose:
            loader = tqdm.tqdm(loader, total=num_batches)
        for batch in loader:
            batch = maybe_dictionarize(batch)
            input = batch[input_key].cuda()
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





class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)


    @torch.no_grad()
    def first_step(self, zero_grad=False, amp=True):
        if amp:
            with torch.cuda.amp.autocast():
                grad_norm = self._grad_norm()
                for group in self.param_groups:
                    scale = group["rho"] / (grad_norm + 1e-12)

                    for p in group["params"]:
                        if p.grad is None: continue
                        self.state[p]["old_p"] = p.data.clone()
                        e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                        p.add_(e_w)  # climb to the local maximum "w + e(w)"
                if zero_grad: self.zero_grad()
        
        else:
            grad_norm = self._grad_norm()
            for group in self.param_groups:
                scale = group["rho"] / (grad_norm + 1e-12)
                
                for p in group["params"]:
                    if p.grad is None: continue
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                    p.add_(e_w)  # climb to the local maximum "w + e(w)"

            if zero_grad: self.zero_grad()


    @torch.no_grad()
    def second_step(self, zero_grad=False, amp=True):
        if amp:
            with torch.cuda.amp.autocast():
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None: continue
                        p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        else:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

            self.base_optimizer.step()  # do the actual "sharpness-aware" update

            if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        self.base_optimizer.step(closure)
        # assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        # closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        # self.first_step(zero_grad=True)
        # closure()
        # self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        
        
        
        
from collections import OrderedDict

import torch, torchvision
import torch.nn as nn
import utils.utils as utils
import utils.sabma.sabma_utils as sabma_utils

import gpytorch
from gpytorch.lazy import RootLazyTensor, DiagLazyTensor, AddedDiagLazyTensor
from gpytorch.distributions import MultivariateNormal


class SABMA(torch.nn.Module):
    def __init__(
        self,
        backbone,
        src_bnn = 'swag',
        w_mean = None,
        diag_only = False,
        w_var = None,
        var_scale = 1,
        low_rank = -1,
        w_cov_sqrt = None,
        cov_scale = 1,
        var_clamp = 1e-16,
        tr_layer="nl_ll",
        pretrained_set = 'source',
        alpha=1e-4
    ):

        super(SABMA, self).__init__()
        
        self.var_clamp = var_clamp
        self.diag_only = diag_only
        self.low_rank = low_rank
        self.tr_layer = tr_layer
        
        self.backbone = backbone
        self.src_bnn = src_bnn
        
        classifier_param = list()
        if isinstance(backbone, torchvision.models.ResNet):
            for param in backbone.fc.parameters():
                classifier_param.extend(torch.flatten(param))
        else:    
            for param in backbone.head.parameters():
                classifier_param.extend(torch.flatten(param))
        classifier_param = torch.tensor(classifier_param)
        self.classifier_param_num = len(classifier_param)
    
        
        ## w_mean
        # random initialization classifier
        if pretrained_set == 'downstream':
            w_mean[-len(classifier_param):] = classifier_param
        elif pretrained_set == 'source':
            # w_mean = torch.cat((w_mean, classifier_param))
            w_mean = torch.cat((w_mean, torch.zeros_like(classifier_param)))
        else:
            raise NotImplementedError("Choose the pretrained_set between 'source' and 'downstream'")
        
        ## diagonal variance (as form of log standard deviation)
        # random initialization classifier
        w_var = torch.clamp(w_var, self.var_clamp)
        if pretrained_set == 'downstream':
            # w_var[-len(classifier_param):] = 1e-2*torch.rand(len(classifier_param))
            w_var[-len(classifier_param):] = alpha*torch.ones(len(classifier_param))
        elif pretrained_set == 'source':
            # w_var = torch.cat((w_var, 1e-2*torch.rand(len(classifier_param))))
            w_var = torch.cat((w_var, alpha*torch.ones(len(classifier_param))))
        
        ## low-ranked covariance matrix
        if src_bnn == 'swag':
            if not self.diag_only:
                if w_cov_sqrt is not None:
                    if type(w_cov_sqrt) == list:
                        # cat blockwise covmat list as full matrix
                        w_cov_sqrt = torch.cat(w_cov_sqrt, dim=1)
                w_cov_sqrt = w_cov_sqrt # * cov_scale
                if pretrained_set == 'downstream':
                    # w_cov_sqrt[:, -len(classifier_param):] = 1e-2*torch.rand((w_cov_sqrt.size(0), len(classifier_param)))
                    w_cov_sqrt[:, -len(classifier_param):] = alpha**0.5*torch.zeros((w_cov_sqrt.size(0), len(classifier_param)))
                else:
                    # w_cov_sqrt = torch.cat((w_cov_sqrt, 1e-2*torch.rand((w_cov_sqrt.size(0), len(classifier_param)))), dim=1)
                    w_cov_sqrt = torch.cat((w_cov_sqrt, alpha**0.5*torch.zeros((w_cov_sqrt.size(0), len(classifier_param)))), dim=1)
                
                self.frz_low_rank = w_cov_sqrt.size(0)
                if self.low_rank < 0:
                    self.low_rank = self.frz_low_rank
                elif self.low_rank != self.frz_low_rank:
                    raise NotImplementedError("No code for different low rank between freezed and trained parameters")
            else:
                print("[Warning] No correlation between parameters")
        ## -----------------------------------------------------------
        
        
        ## calculate the number of params and shape of each layer, set trainable params, and get indices of them
        self.bnn_param = nn.ParameterDict()
        
        self.full_param_shape = OrderedDict(); self.full_param_num = 0
        self.tr_param_shape = OrderedDict(); self.tr_param_num = 0
        self.frz_param_shape = OrderedDict(); self.frz_param_num = 0
        self.tr_param_idx = list(); self.frz_param_idx = list()
        for name, p in backbone.named_parameters():
            p.requires_grad = False
            
            # resnet
            if isinstance(backbone, torchvision.models.ResNet):   
                if ('bn' in name) or ('fc' in name):
                    self.tr_param_shape[name] = p.shape
                    self.tr_param_num += p.shape.numel()
                    self.tr_param_idx.append((self.full_param_num, self.full_param_num + p.shape.numel()))    
                else:
                    self.frz_param_shape[name] = p.shape
                    self.frz_param_num += p.shape.numel()
                    self.frz_param_idx.append((self.full_param_num, self.full_param_num + p.shape.numel()))    
                self.full_param_shape[name] = p.shape
                self.full_param_num += p.shape.numel()      
                
            # Vit
            else:
                if ('norm' in name) or ('head' in name):
                    self.tr_param_shape[name] = p.shape
                    self.tr_param_num += p.shape.numel()
                    self.tr_param_idx.append((self.full_param_num, self.full_param_num + p.shape.numel()))
                else:
                    self.frz_param_shape[name] = p.shape
                    self.frz_param_num += p.shape.numel()
                    self.frz_param_idx.append((self.full_param_num, self.full_param_num + p.shape.numel()))    
                self.full_param_shape[name] = p.shape
                self.full_param_num += p.shape.numel()
    
        self.tr_param_idx = torch.tensor([i for start, end in self.tr_param_idx for i in range(start, end)])
        self.frz_param_idx = torch.tensor([i for start, end in self.frz_param_idx for i in range(start, end)])
        
        self.register_buffer("frz_mean", w_mean[self.frz_param_idx])
        self.bnn_param.update({"mean" :
                nn.Parameter(torch.index_select(w_mean, dim=0, index=self.tr_param_idx))})

        self.register_buffer("frz_log_std", 0.5 * torch.log(w_var[self.frz_param_idx] * var_scale))
        self.bnn_param.update({"log_std" :
                nn.Parameter(0.5*torch.log(torch.index_select(w_var, dim=0, index=self.tr_param_idx) * var_scale))})
        
        self.register_buffer("frz_cov_sqrt", w_cov_sqrt[:, self.frz_param_idx] * cov_scale)
        self.bnn_param.update({"cov_sqrt" :
                nn.Parameter(torch.index_select(w_cov_sqrt, dim=1, index=self.tr_param_idx))[:self.low_rank, :] * cov_scale})    

        assert self.frz_mean.numel() + self.bnn_param['mean'].numel() == self.full_param_num, "division of mean parameters was not right!"
        assert self.frz_log_std.numel() + self.bnn_param['log_std'].numel() == self.full_param_num, "division of variance parameters was not right!"
        assert self.frz_cov_sqrt.numel() + self.bnn_param['cov_sqrt'].numel() == self.full_param_num * self.low_rank,  "division of covariance parameters was not right!"
        
        if self.tr_layer == 'last_layer':
            print("Prior cannot be defined")
        elif self.tr_layer == 'nl_ll':
            self.register_buffer("prior_mean", self.bnn_param['mean'].detach().clone()[:-self.classifier_param_num])
            self.register_buffer("prior_log_std", self.bnn_param['log_std'].detach().clone()[:-self.classifier_param_num])
            self.register_buffer("prior_cov_sqrt", self.bnn_param['cov_sqrt'].detach().clone()[:, :-self.classifier_param_num])
        else:
            raise NotImplementedError("No code for prior except last layer and normalization layer + last layer setting")
        # -----------------------------------------------------------------------------------------------------    


    def forward(self, params, input):
        return nn.utils.stateless.functional_call(self.backbone, params, input)
    
    
    
    def sample(self, z_scale=1.0, sample_param='tr'):
        '''
        Sample weight from bnn params
        '''
        if sample_param == 'frz':
            ## freezed params only
            z_1 = torch.randn_like(self.frz_mean, requires_grad=False)
            rand_sample = torch.exp(self.frz_log_std) * z_1
            if not self.diag_only:
                z_2 = self.frz_cov_sqrt.new_empty((self.frz_low_rank, ), requires_grad=False).normal_(z_scale)
                cov_sample = self.frz_cov_sqrt.t().matmul(z_2)
                if self.low_rank > 1:
                    cov_sample /= (self.frz_low_rank - 1)**0.5
            else:
                z_2 = None
            rand_sample = 0.5**0.5 * (rand_sample + cov_sample)    
            sample = self.frz_mean + rand_sample
            
        elif sample_param == 'tr':
            # trainable params only
            z_1 = torch.randn_like(self.bnn_param['mean'], requires_grad=False)
            rand_sample = torch.exp(self.bnn_param['log_std']) * z_1
            if not self.diag_only:
                z_2 = self.bnn_param['cov_sqrt'].new_empty((self.low_rank, ), requires_grad=False).normal_(z_scale)
                cov_sample = self.bnn_param['cov_sqrt'].t().matmul(z_2)
                if self.low_rank > 1:
                    cov_sample /= (self.low_rank - 1)**0.5
            else:
                z_2 = None
            rand_sample = 0.5**0.5 * (rand_sample + cov_sample)    
            sample = self.bnn_param['mean'] + rand_sample
        
        else:
            raise NotImplementedError("No code for other sampling methods except training params only and frezzed params only")
        
        return sample, z_1, z_2
        
        
    def prior_log_prob(self):
        '''
        calculate prior log_grad
        '''
        if not self.diag_only:
            cov_mat_lt = RootLazyTensor(self.prior_cov_sqrt.t())
            var_lt = DiagLazyTensor(torch.exp(self.prior_log_std))
            covar = AddedDiagLazyTensor(var_lt, cov_mat_lt).add_jitter(1e-10)
        else:
            covar = DiagLazyTensor(torch.exp(self.prior_log_std))
        qdist = MultivariateNormal(self.prior_mean, covar)
        prior_sample = qdist.rsample()
        with gpytorch.settings.num_trace_samples(1) and gpytorch.settings.max_cg_iterations(25):
            prior_log_prob =  qdist.log_prob(prior_sample)
        
        return prior_log_prob


    def posterior_log_prob(self):
        '''
        calculate prior log_grad
        '''
        if self.tr_layer != 'nl_ll':
            raise NotImplementedError("Need to fix indexing except training normalization and last layer")
        
        if not self.diag_only:    
            cov_mat_lt = RootLazyTensor(self.bnn_param['cov_sqrt'][:,:-self.classifier_param_num].t())
            var_lt = DiagLazyTensor(torch.exp(self.bnn_param['log_std'][:-self.classifier_param_num]))
            covar = AddedDiagLazyTensor(var_lt, cov_mat_lt).add_jitter(1e-10)   
        else:
            covar = DiagLazyTensor(torch.exp(self.bnn_param['log_std'][:-self.classifier_param_num]))
        
        qdist = MultivariateNormal(self.bnn_param['mean'][:-self.classifier_param_num], covar)
        posterior_sample = qdist.rsample()
        with gpytorch.settings.num_trace_samples(1) and gpytorch.settings.max_cg_iterations(25):
            posterior_log_prob =  qdist.log_prob(posterior_sample)
        
        return posterior_log_prob


    def log_grad(self, params):
        '''
        Compute gradient of log probability w.r.t bnn params
        '''
        
        if not self.diag_only:
            cov_mat_lt = RootLazyTensor(self.bnn_param['cov_sqrt'].t())
            var_lt = DiagLazyTensor(torch.exp(self.bnn_param['log_std']))
            covar = AddedDiagLazyTensor(var_lt, cov_mat_lt).add_jitter(1e-10)
        else:
            covar = DiagLazyTensor(torch.exp(self.bnn_param['log_std']))
        qdist = MultivariateNormal(self.bnn_param['mean'], covar)
        with gpytorch.settings.num_trace_samples(1) and gpytorch.settings.max_cg_iterations(25):
            log_prob =  qdist.log_prob(params)      

        # gradient of log probability w.r.t. mean
        mean_log_grad = covar.inv_matmul((params - self.bnn_param['mean']))    ## calculate derivative manually (gpytorch version)
        
        # gradient of log probability w.r.t. diagonal variance
        std_log_grad = torch.autograd.grad(log_prob, self.bnn_param['log_std'], retain_graph=True)[0]
        
        # gradient of log probability w.r.t. low-rank covariance
        cov_log_grad = torch.autograd.grad(log_prob, self.bnn_param['cov_sqrt'], retain_graph=True)[0]
        cov_log_grad = torch.flatten(cov_log_grad)
        
        return log_prob, [mean_log_grad, std_log_grad, cov_log_grad]   
    
    
        
    def get_mean_vector(self, unflatten=False):
        '''
        Save mean vector
        '''
        if self.tr_layer != 'full_layer':
            mean_param = torch.zeros(self.full_param_num)
            mean_param[self.tr_param_idx] = self.bnn_param['mean'].cpu()
            mean_param[self.frz_param_idx] = self.frz_mean.cpu()
        else:
            mean_param = self.bnn_param['mean'].cpu()
            
        if unflatten:
            return utils.unflatten_like_size(mean_param, self.full_param_shape.values())
        else:
            return mean_param


    def get_variance_vector(self, unflatten=False):
        '''
        Save variance vector (Not std)
        '''
        if self.tr_layer != 'full_layer':
            var_param = torch.zeros(self.full_param_num)
            var_param[self.tr_param_idx] = self.bnn_param['log_std'].cpu()
            var_param[self.frz_param_idx] = self.frz_log_std.cpu()
        else:
            var_param = self.bnn_param['log_std'].cpu()
        var_param = torch.exp(2*var_param)
        
        if unflatten:
            return utils.unflatten_like_size(var_param, self.full_param_shape.values())
        else:
            return var_param


    def get_covariance_matrix(self):
        '''
        Save covariance vector
        '''
        if self.diag_only:
            return None
        else:
            if self.tr_layer != 'full_layer':
                cov_param = torch.zeros((self.low_rank, self.full_param_num))
                cov_param[:, self.tr_param_idx] = self.bnn_param['cov_sqrt'].cpu()
                cov_param[:, self.frz_param_idx] = self.frz_cov_sqrt.cpu()
            else:
                cov_param = self.bnn_param['cov_sqrt'].cpu()
            return cov_param
                   
    
    
    def load_state_dict(self, state_dict, strict=True):
        '''
        load하는거 만들어놓기
        '''
        super(SABMA, self).load_state_dict(state_dict, strict)

#####################################################################################################################





## BSAM
class SABMA_optim(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SABMA_optim, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        self.shared_device = self.param_groups[0]["params"][0].device


    @torch.no_grad()
    def first_step(self, log_grad, zero_grad=False, amp=True):
        if amp:
            with torch.cuda.amp.autocast():
                for group in self.param_groups:
                    for idx, p in enumerate(group["params"]):
                        if p.grad is None: continue
                        self.state[p]["old_p"] = p.data.clone()
                                           
                        ## Calculate perturbation --------------------------------------
                        Delta_p = group["rho"] / (log_grad[idx].norm(p=2).to(self.shared_device) + 1e-12)   # add small value for numericaly stability
                        Delta_p = Delta_p * log_grad[idx]
                        if idx == 2:
                            Delta_p = Delta_p.reshape((-1, log_grad[0].size(0)))
                        # --------------------------------------------------------------------------- 
                                               
                        ## theta + Delta_theta
                        p.add_(Delta_p) # climb to the local maximum "w + e(w)"
                        # ---------------------------------------------------------------------------
                if zero_grad: self.zero_grad()
                
        else:
            raise NotImplementedError("Need to be fixed")


    def second_sample(self, z_1, z_2, sabma_model):
        '''
        Sample from perturbated bnn parameters with pre-selected z_1, z_2
        '''

        # diagonal variance
        rand_sample = (torch.exp(self.param_groups[0]['params'][1])) * z_1
        
        # covariance
        if not sabma_model.diag_only:
            cov_sample = (self.param_groups[0]['params'][2].t().matmul(z_2[:sabma_model.low_rank]))
            if sabma_model.low_rank > 1:
                cov_sample /= (sabma_model.low_rank - 1)**0.5
            rand_sample = 0.5**0.5 * (rand_sample + cov_sample)
        sample = self.param_groups[0]['params'][0] + rand_sample
    
        return sample


    @torch.no_grad()
    def second_step(self, zero_grad=False, amp=True):
        if amp:
            with torch.cuda.amp.autocast():
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None: continue
                        p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        else:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    p.data = self.state[p]["old_p"]

            self.base_optimizer.step()  # do the actual "sharpness-aware" update

            if zero_grad: self.zero_grad()



    @torch.no_grad()
    def step(self, closure=None):
        self.base_optimizer.step(closure)