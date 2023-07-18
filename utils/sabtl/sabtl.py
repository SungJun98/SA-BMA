import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import pi
import utils.utils as utils
from utils.swag.swag_utils import flatten

import gpytorch
from gpytorch.lazy import RootLazyTensor, DiagLazyTensor, AddedDiagLazyTensor
from gpytorch.distributions import MultivariateNormal

class SABTL(torch.nn.Module):
    def __init__(
        self,
        backbone,
        src_bnn = 'swag',
        w_mean = None,
        diag_only = False,
        w_var=None,
        var_scale = 1e2,
        low_rank = 20,
        w_cov_sqrt=None,
        cov_scale = 1e1,
        var_clamp = 1e-16,
        last_layer=False,
    ):
        super(SABTL, self).__init__()
        
        # setting for covariance
        self.var_clamp = var_clamp
        self.diag_only = diag_only
        self.low_rank = low_rank
        self.last_layer = last_layer
        
        self.backbone = backbone
        
        self.full_model_shape = list()
        for p in backbone.parameters():
            p.requires_grad = False
            self.full_model_shape.append(p.shape)

        # Get Last Layer Name and Last Layer Shape
        if last_layer:
            for name, _ in self.backbone.named_modules():
                self.last_layer_name = name
            self.last_layer_shape = self.full_model_shape[-2:]
            
        ## Load Pre-Trained Model
        if w_mean is not None:
            self.load_backbone(w_mean)
        else:
            raise RuntimeError("We need pre-trained weight to define model")
        
        # Get total number of parameters and backbone shape which are updated during training
        self.num_params = 0
        for name, param in self.backbone.named_parameters():
            if last_layer:
                if name.split('.')[0] == self.last_layer_name:
                    self.num_params += param.numel()
            else:
                self.num_params += param.numel()

        ### Add Mean, Var, Cov layer ---------------------------------------------------------------
        self.bnn_param = nn.ParameterDict()

        ## Mean
        w_mean = w_mean[-self.num_params:]
        self.bnn_param.update({"mean" : nn.Parameter(w_mean)})

        ## Variance
        if w_var is not None:
            w_var = w_var[-self.num_params:]*var_scale
        else:
            w_var = torch.rand(self.num_params)
        w_var = torch.clamp(w_var, self.var_clamp)
        w_std = 0.5*torch.log(w_var)    # log_std       
        self.bnn_param.update({"log_std" : nn.Parameter(w_std)})   
        
        ## Covariance           
        if src_bnn == 'swag':
            if not self.diag_only:
                if w_cov_sqrt is not None:
                    if type(w_cov_sqrt) == list:
                        # cat covmat list as matrix
                        w_cov_sqrt = torch.cat(w_cov_sqrt, dim=1)         
                    w_cov_sqrt = w_cov_sqrt[:, -self.num_params:]*cov_scale
                    if self.low_rank == 0:
                        L = w_cov_sqrt.t().matmul(w_cov_sqrt)
                        L += torch.diag(w_var)
                        L = torch.cholesky(L)
                        L = torch.tril(L, diagonal=-1)
                        self.bnn_param.update({"cov_sqrt" : nn.Parameter(L)})
                        if torch.sum(torch.isnan(self.bnn_param['cov_sqrt'])) != 0:
                            raise RuntimeError("There's NaN value in lower traingular matrix")
                    else:
                        self.low_rank = w_cov_sqrt.size(0)
                        self.bnn_param.update({"cov_sqrt" : nn.Parameter(w_cov_sqrt)})
                else:
                    self.bnn_param.update({"cov_sqrt" : nn.Parameter(torch.randn((low_rank, self.num_params)*1e-2))})

        elif src_bnn == 'la':
            raise RuntimeError("Add Load for Laplace Approximation")
        
        elif src_bnn == 'vi':
            if not self.diag_only:
                self.bnn_param.update({"cov_sqrt" : nn.Parameter(torch.randn((low_rank, self.num_params)*1e-2))})
        
        print(f"Load covariance of weight from pre-trained {src_bnn} model")
        # -----------------------------------------------------------------------------------------------------
        
    

    def load_backbone(self, w_mean):
        '''
        Reform Saved Weight As State Dict
        and Load Pre-Trained Backbone Model
        '''
        if self.last_layer:
            unflatten_mean_list = utils.unflatten_like_size(w_mean, self.last_layer_shape)
            st_dict = dict()
            st_dict[f'{self.last_layer_name}.weight'] = unflatten_mean_list[0]
            st_dict[f'{self.last_layer_name}.bias'] = unflatten_mean_list[1]
            self.backbone.load_state_dict(st_dict, strict=False)
        else:
            unflatten_mean_list = utils.unflatten_like_size(w_mean, self.full_model_shape)
            st_dict = dict()
            for idx, (name, _) in enumerate(self.backbone.named_parameters()):
                st_dict[name] = unflatten_mean_list[idx]
            self.backbone.load_state_dict(st_dict, strict=False)


    def forward(self, params, input):
        return nn.utils.stateless.functional_call(self.backbone, params, input)
    
    
    def sample(self, z_scale=1.0):
        '''
        Sample weight from bnn params
        '''
        if not self.diag_only:
            if self.low_rank == 0:
                z_1 = self.bnn_param['cov_sqrt'].new_empty((self.bnn_param['cov_sqrt'].size(0),), requires_grad=False).normal_(std=z_scale)
                rand_sample = (torch.exp(self.bnn_param['log_std']) + self.var_clamp**0.5) * z_1
                rand_sample += self.bnn_param['cov_sqrt'].matmul(z_1)
                z_2 = None
                sample = self.bnn_param['mean'] + rand_sample

            else:
                z_1 = torch.randn_like(self.bnn_param['log_std'], requires_grad=False) * z_scale
                rand_sample = torch.exp(self.bnn_param['log_std']) * z_1
                z_2 = self.bnn_param['cov_sqrt'].new_empty((self.bnn_param['cov_sqrt'].size(0),), requires_grad=False).normal_(std=z_scale)
                cov_sample = self.bnn_param['cov_sqrt'].t().matmul(z_2)
                cov_sample /= (self.low_rank - 1)**0.5
                rand_sample += cov_sample
                sample = self.bnn_param['mean'] + 0.5**0.5 * rand_sample
                
        else:
            z_1 = torch.randn_like(self.bnn_param['log_std'], requires_grad=False) * z_scale
            rand_sample = torch.exp(self.bnn_param['log_std']) * z_1
            z_2 = None
            sample = self.bnn_param['mean'] + rand_sample
        
        return sample, z_1, z_2
        
        
    def fish_inv(self, params, eta=1.0):
        '''
        Compute gradient of log probability w.r.t bnn params
        '''
        soft_std = torch.exp(self.bnn_param['log_std']) + self.var_clamp**0.5
        if not self.diag_only:
            if self.low_rank == 0:
                covar = torch.tril(self.bnn_param['cov_sqrt'], diagonal=-1) + torch.diag(soft_std)
                covar = covar.matmul(covar.t())
            else:
                cov_mat_lt = RootLazyTensor(self.bnn_param['cov_sqrt'].t())
                var_lt = DiagLazyTensor(soft_std**2) #  + self.var_clamp)
                covar = AddedDiagLazyTensor(var_lt, cov_mat_lt).add_jitter(1e-6)
        else:
            covar = torch.diag(soft_std**2)
        
        qdist = MultivariateNormal(self.bnn_param['mean'], covar)

        with gpytorch.settings.num_trace_samples(1) and gpytorch.settings.max_cg_iterations(25):
            log_prob =  qdist.log_prob(params)
        ## Fisher Inverse w.r.t. mean
        # \nabla_\mean p(w | \theta)
        # mean_fi = torch.autograd.grad(log_prob, self.bnn_param['mean'], retain_graph=True)
        # mean_fi = mean_fi[0]**2
        # mean_fi = 1 / (1 + eta * mean_fi)    
        mean_fi = covar.inv_matmul((params - self.bnn_param['mean']))   ## calculate derivative manually (gpytorch version)
        mean_fi = mean_fi**2
        mean_fi = 1 / (1 + eta * mean_fi)

        ## Fisher Inverse w.r.t. variance
        std_fi = torch.autograd.grad(log_prob, self.bnn_param['log_std'], retain_graph=True)
        std_fi = std_fi[0]**2
        std_fi = 1 / (1 + eta * std_fi)
        
        ## Fisher Inverse w.r.t. covariance
        # \nabla_\cov p(w | \theta)
        if not self.diag_only:
            cov_fi = torch.autograd.grad(log_prob, self.bnn_param['cov_sqrt'], retain_graph=True)
            cov_fi = cov_fi[0]**2
            cov_fi = 1 / (1 + eta * cov_fi)
            if self.low_rank == 0:
                cov_fi = torch.tril(cov_fi, diagonal=-1) 
            return [mean_fi, std_fi, cov_fi]
        else:
            return [mean_fi, std_fi]

    
    
    def get_mean_vector(self, unflatten=False):
        '''
        Load mean vector
        '''
        if unflatten:
            return utils.unflatten_like_size(self.bnn_param['mean'], self.backbone_shape)
        else:
            return self.bnn_param['mean']


    def get_variance_vector(self, unflatten=False):
        '''
        Load variance vector (Not std)
        '''
        # variance = torch.exp(utils.softclip(self.bnn_param['log_std'], std=False))
        variance = torch.exp(2*self.bnn_param['log_std']) + self.var_clamp
        if unflatten:
            return utils.unflatten_like_size(variance, self.backbone_shape)
        else:
            return variance


    def get_covariance_matrix(self):
        '''
        Load covariance vector
        '''
        if self.diag_only:
            return None
        else:
            return self.bnn_param['cov_sqrt']
    
    
    def load_state_dict(self, state_dict, strict=True):
        '''
        load하는거 만들어놓기
        '''
        super(SABTL, self).load_state_dict(state_dict, strict)

#####################################################################################################################





## BSAM
class BSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(BSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    


    @torch.no_grad()
    def first_step(self, fish_inv, zero_grad=False, amp=True):
        if amp:
            with torch.cuda.amp.autocast():
                for group in self.param_groups:
                    for idx, p in enumerate(group["params"]):
                        if p.grad is None: continue
                        self.state[p]["old_p"] = p.data.clone()
                        ## Calculate perturbation Delta_theta --------------------------------------
                        Delta_p = group["rho"] * fish_inv[idx] * p.grad
                        Delta_p = Delta_p / (torch.sqrt(p.grad * fish_inv[idx] * p.grad) + 1e-12) # add small value for numericaly stability
                        # ---------------------------------------------------------------------------                        
                        ## theta + Delta_theta
                        p.add_(Delta_p)  # climb to the local maximum "w + e(w)"
                        # ---------------------------------------------------------------------------
                if zero_grad: self.zero_grad()
                
        else:
            for group in self.param_groups:
                for idx, p in enumerate(group["params"]):
                    if p.grad is None: continue
                    self.state[p]["old_p"] = p.data.clone()
                    ## Calculate perturbation Delta_theta --------------------------------------
                    Delta_p = group["rho"] * fish_inv[idx] * p.grad
                    Delta_p = Delta_p / (torch.sqrt(p.grad * fish_inv[idx] * p.grad) + 1e-12) # add small value for numericaly stability
                    # ---------------------------------------------------------------------------                        
                    ## theta + Delta_theta
                    p.add_(Delta_p)  # climb to the local maximum "w + e(w)"
                    # ---------------------------------------------------------------------------
            if zero_grad: self.zero_grad()




    def second_sample(self, z_1, z_2, sabtl_model):
        '''
        Sample from perturbated bnn parameters with pre-selected z_1, z_2
        '''
        if not sabtl_model.diag_only:
            rand_sample = (torch.exp(self.param_groups[0]['params'][1]) + sabtl_model.var_clamp**0.5) * z_1
            if sabtl_model.low_rank != 0:
                rand_sample += (self.param_groups[0]['params'][2].t().matmul(z_2)) / (sabtl_model.low_rank - 1)**0.5
                sample = self.param_groups[0]['params'][0] + 0.5**0.5 * rand_sample
            else:
                rand_sample += self.param_groups[0]['params'][2].matmul(z_1)
                sample = self.param_groups[0]['params'][0] + rand_sample
            
        else:
            rand_sample = torch.exp(self.param_groups[0]['params'][1]) * z_1
            sample = self.param_groups[0]['params'][0] + rand_sample    
        
        # change sampled weight type list to dict 
        sample = utils.format_weights(sample, sabtl_model)
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
                    p.data = self.state[p]["old_p"]  # get ba

            self.base_optimizer.step()  # do the actual "sharpness-aware" update

            if zero_grad: self.zero_grad()



    @torch.no_grad()
    def step(self, closure=None):
        self.base_optimizer.step(closure)
    

    '''
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
    '''
