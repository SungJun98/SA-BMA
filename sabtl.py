import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils
from baselines.swag import swag_utils

import gpytorch
from gpytorch.lazy import RootLazyTensor, DiagLazyTensor, AddedDiagLazyTensor
from gpytorch.distributions import MultivariateNormal

class SABTL(torch.nn.Module):
    def __init__(
        self,
        backbone,
        src_bnn = 'swag',
        w_mean = None,
        w_var=None,
        prior_var_scale = 1,
        w_cov_sqrt=None,
        prior_cov_scale = 1,
        var_clamp = 1e-4,
    ):
        super(SABTL, self).__init__()
        
        for p in backbone.parameters():
            p.requires_grad = False

        
        # setting for covariance
        self.var_clamp = var_clamp
        self.backbone = backbone        
        self.full_model_shape = [p.shape for p in backbone.parameters()]
        
        ## Load Pre-Trained Model
        if w_mean is not None:
            self.load_full_backbone(w_mean, src_bnn)
        else:
            raise RuntimeError("We need pre-trained weight to define model")
        
        # Set Last Layer Name
        for name, _ in self.backbone.named_modules():
            self.last_layer_name = name
        
        # Get total number of parameters and backbone shape which are updated during training
        self.num_params = 0
        for name, param in self.backbone.named_parameters():
            if name.split('.')[0] == self.last_layer_name:
                self.num_params += param.numel()
        self.backbone_shape = self.full_model_shape[-2:]

        ### Add Mean, Var, Cov layer ---------------------------------------------------------------
        self.bnn_param = nn.ParameterDict()

        if src_bnn == 'swag':
            ## Mean
            w_mean = w_mean[-self.num_params:]
            self.bnn_param.update({"mean" : nn.Parameter(w_mean)})

            ## Variance
            if w_var is not None:
                w_var = w_var[-self.num_params:]
            else:
                w_var = torch.rand(self.num_params)
            w_var = torch.clamp(w_var * prior_var_scale, self.var_clamp)
            w_std = 0.5*torch.log(w_var)    # log_std       
            self.bnn_param.update({"log_std" : nn.Parameter(w_std)})      

            ## Covariance
            if w_cov_sqrt is not None:
                if type(w_cov_sqrt) == list:
                    # cat covmat list as matrix
                    w_cov_sqrt = torch.cat(w_cov_sqrt, dim=1)         
                w_cov_sqrt = w_cov_sqrt[:, -self.num_params:]
                L = w_cov_sqrt.t().matmul(w_cov_sqrt)
                L += torch.diag(w_var)
                L = torch.cholesky(L)
                L = torch.tril(L, diagonal=-1)
                self.bnn_param.update({"cov_sqrt" : nn.Parameter(L)})
                if torch.sum(torch.isnan(self.bnn_param['cov_sqrt'])) != 0:
                    raise RuntimeError("There's NaN value in lower traingular matrix")

        elif src_bnn == 'la':
            raise RuntimeError("Add Load for Laplace Approximation")
        
        elif src_bnn == 'vi':
            raise RuntimeError("Add Load for Variational Inference")
        
        print(f"Load covariance of weight from pre-trained {src_bnn} model")
        # -----------------------------------------------------------------------------------------------------
        
    

    def load_full_backbone(self, w_mean, src_bnn):
        '''
        Reform Saved Weight As State Dict
        and Load Pre-Trained Backbone Model
        '''
        if src_bnn == 'swag':
            unflatten_mean_list = utils.unflatten_like_size(w_mean, self.full_model_shape)
            st_dict = dict()
            for (name, _), w in zip(self.backbone.named_parameters(), unflatten_mean_list):
                st_dict[name] = w
                
        self.backbone.load_state_dict(st_dict, strict=False)
        ## bn의 running_mean, running_var는 trainable parameter가 아니라 save할 때 제대로 안 됨


    def forward(self, params, input):
        return nn.utils.stateless.functional_call(self.backbone, params, input)
    
    
    def sample(self, scale=1.0):
        '''
        Sample weight from bnn params
        '''
        # z_ = self.bnn_param['cov_sqrt'].new_empty((self.bnn_param['cov_sqrt'].size(0),), requires_grad=False).normal_(std=1e-4)
        z_ = self.bnn_param['cov_sqrt'].new_empty((self.bnn_param['cov_sqrt'].size(0),), requires_grad=False).normal_(std=1e-2) ###
        
        rand_sample = (torch.exp(self.bnn_param['log_std']) + self.var_clamp**0.5) * z_
        rand_sample += self.bnn_param['cov_sqrt'].matmul(z_)
        
        # update sample with mean and scale
        sample = self.bnn_param['mean'] + scale**0.5 * rand_sample
        return sample, z_
        
        
    def fish_inv(self, params, eta=1.0):
        '''
        Compute gradient of log probability w.r.t bnn params
        '''
        # soft_std = torch.exp(utils.softclip(self.bnn_param['log_std'])) + 1e-1
        soft_std = torch.exp(self.bnn_param['log_std']) + self.var_clamp**0.5
        covar = torch.tril(self.bnn_param['cov_sqrt'], diagonal=-1) + torch.diag(soft_std)
        covar = covar.matmul(covar.t())
        
        qdist = MultivariateNormal(self.bnn_param['mean'], covar)
        with gpytorch.settings.num_trace_samples(10) and gpytorch.settings.max_cg_iterations(25):
            log_prob =  qdist.log_prob(params)
        
        ## Fisher Inverse w.r.t. mean
        # \nabla_\mean p(w | \theta)
        mean_fi = torch.autograd.grad(log_prob, self.bnn_param['mean'], retain_graph=True)
        mean_fi = mean_fi[0]**2
        mean_fi = 1 / (1 + eta * mean_fi)

        # # ## Fisher Inverse w.r.t. variance
        # std_fi = torch.autograd.grad(log_prob, soft_std, retain_graph=True)
        std_fi = torch.autograd.grad(log_prob, self.bnn_param['log_std'], retain_graph=True)
        std_fi = std_fi[0]**2
        std_fi = 1 / (1 + eta * std_fi)
        
        ## Fisher Inverse w.r.t. covariance
        # \nabla_\cov p(w | \theta)
        cov_fi = torch.autograd.grad(log_prob, self.bnn_param['cov_sqrt'], retain_graph=True)
        cov_fi = cov_fi[0]**2
        cov_fi = 1 / (1 + eta * cov_fi)
        cov_fi = torch.tril(cov_fi, diagonal=-1)
        
        return [mean_fi, std_fi, cov_fi]

    
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
        variance = torch.exp(2*self.bnn_param['log_std']) + self.var_clamp**0.5
        if unflatten:
            return utils.unflatten_like_size(variance, self.backbone_shape)
        else:
            return variance


    def get_covariance_matrix(self):
        '''
        Load covariance vector
        '''
        # if self.diag_only:
        #     raise RuntimeError("No covariance matrix was estimated!")        
    
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
    def first_step(self, fish_inv, zero_grad=False):
        with torch.cuda.amp.autocast():
            for group in self.param_groups:
                for idx, p in enumerate(group["params"]):
                    if p.grad is None: continue
                    self.state[p]["old_p"] = p.data.clone()
                    ### Calculate perturbation Delta_theta --------------------------------------
                    # print(f"p  nan count : {torch.sum(torch.isnan(p))}")
                    Delta_p = group["rho"] * fish_inv[idx] * p.grad
                    # print(f"Delta p nan count : {torch.sum(torch.isnan(Delta_p))}")
                    Delta_p = Delta_p / (torch.sqrt(p.grad * fish_inv[idx] * p.grad) + 1e-6) # add small value for numericaly stability
                    # print(f"Full Delta nan count : {torch.sum(torch.isnan(Delta_p))}")
                    # ---------------------------------------------------------------------------    
                            
                    ### theta + Delta_theta
                    p.add_(Delta_p)  # climb to the local maximum "w + e(w)"
                    # print(f"perturbated p nan count : {torch.sum(torch.isnan(p))}")
                    # ---------------------------------------------------------------------------

            if zero_grad: self.zero_grad()


    def second_sample(self, z_, sabtl_model, scale=1.0):
        '''
        Sample from perturbated bnn parameters with pre-selected z_1, z_2
        '''
        # rand_sample = torch.tril(self.param_groups[0]['params'][2], diagonal=-1)
        # rand_sample += torch.diag(torch.exp(utils.softclip(self.param_groups[0]['params'][1]))+ sabtl_model.var_clamp**0.5)
        # rand_sample = rand_sample.matmul(z_)
        # rand_sample = (torch.exp(utils.softclip(self.param_groups[0]['params'][1])) + 1e-1) * z_
        rand_sample = (torch.exp(self.param_groups[0]['params'][1]) + sabtl_model.var_clamp**0.5) * z_
        rand_sample += self.param_groups[0]['params'][2].matmul(z_)
        
        # update sample with mean and scale
        sample = self.param_groups[0]['params'][0] + scale**0.5 * rand_sample
        # change sampled weight type list to dict 
        sample = utils.format_weights(sample, sabtl_model)
        return sample



    @torch.no_grad()
    def second_step(self, zero_grad=False):
        with torch.cuda.amp.autocast():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        # self.base_optimizer.step()  # do the actual "sharpness-aware" update
        # if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        self.base_optimizer.step(closure)
    

    '''
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
    '''
