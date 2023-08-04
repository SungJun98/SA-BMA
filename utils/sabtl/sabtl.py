import torch
import torch.nn as nn
import utils.utils as utils

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
        w_var = None,
        var_scale = 1,
        low_rank = 3,
        w_cov_sqrt = None,
        cov_scale = 1,
        var_clamp = 1e-16,
        last_layer=True,
    ):
        super(SABTL, self).__init__()
        
        self.var_clamp = var_clamp
        self.diag_only = diag_only
        self.low_rank = low_rank
        self.last_layer = last_layer
        
        self.backbone = backbone
        self.src_bnn = src_bnn
        
        self.full_model_shape = list()
        for p in backbone.parameters():
            p.requires_grad = False
            self.full_model_shape.append(p.shape)

        # Get Last Layer Name and Feature Extractor / Last Layer Shape
        for name, _ in self.backbone.named_modules():
            self.last_layer_name = name
        self.fe_layer_shape = self.full_model_shape[:-2]
        self.last_layer_shape = self.full_model_shape[-2:]

        if w_mean is None:
            raise NotImplementedError("We need Pre-trained weight to define model")
        
        # Get the number of total / last_layer / Feature Extractor parameters
        self.total_num_params = 0 ; self.ll_num_params = 0
        for name, param in self.backbone.named_parameters():
            self.total_num_params += param.numel()
            if name.split('.')[0] == self.last_layer_name:
                self.ll_num_params += param.numel()

        ### Add Mean, Var, Cov layer ---------------------------------------------------------------
        self.bnn_param = nn.ParameterDict()

        ## Mean
        self.register_buffer('fe_mean', w_mean[:-self.ll_num_params])
        self.bnn_param.update({"mean" : nn.Parameter(w_mean[-self.ll_num_params:])})

        ## Variance
        w_var = torch.clamp(w_var, self.var_clamp)
        w_log_std = 0.5 * torch.log(w_var)              # log_std
        self.register_buffer('fe_log_std', w_log_std[:-self.ll_num_params])
        self.bnn_param.update({"log_std" : nn.Parameter(w_log_std[-self.ll_num_params:] * var_scale)})
        
        ## Covariance           
        if src_bnn == 'swag':
            if not self.diag_only:
                if w_cov_sqrt is not None:
                    if type(w_cov_sqrt) == list:
                        # cat covmat list as matrix
                        w_cov_sqrt = torch.cat(w_cov_sqrt, dim=1) 
                    self.register_buffer('fe_cov_sqrt', w_cov_sqrt[:, :-self.ll_num_params])
                    self.bnn_param.update({"cov_sqrt" : nn.Parameter(w_cov_sqrt[:,-self.ll_num_params:] * cov_scale)})
                    self.low_rank = w_cov_sqrt.size(0)
                else:
                    self.bnn_param.update({"cov_sqrt" : nn.Parameter(torch.randn((self.low_rank, self.ll_num_params))*1e-2)})

        elif src_bnn == 'la':
            raise RuntimeError("Add Load for Laplace Approximation")
        
        elif src_bnn == 'vi':
            if not self.diag_only:
                self.bnn_param.update({"cov_sqrt" : nn.Parameter(torch.randn((self.low_rank, self.ll_num_params))*1e-2)})
        
        print(f"Load covariance of weight from pre-trained {src_bnn} model")
        # -----------------------------------------------------------------------------------------------------    


    def forward(self, params, input):
        return nn.utils.stateless.functional_call(self.backbone, params, input)
    
    
    def sample(self, z_scale=1.0, last_only=False):
        '''
        Sample weight from bnn params
        '''
        if not last_only:
            sample = torch.cat((self.fe_mean, self.bnn_param.mean))
            z_1 = torch.randn_like(sample, requires_grad=False)
            if not self.diag_only:
                z_2 = self.bnn_param.cov_sqrt.new_empty((self.bnn_param.cov_sqrt.size(0), ), requires_grad=False).normal_(z_scale)
            else:
                z_2 = None
            sample += 0.5**0.5 * (torch.exp(torch.cat((self.fe_log_std, self.bnn_param.log_std))) * z_1 + torch.cat((self.fe_cov_sqrt, self.bnn_param.cov_sqrt)).t().matmul(z_2))
        
        else:
            z_1 = torch.randn_likfe(self.bnn_param.mean, requires_grad=False)
            if not self.diag_only:
                z_2 = self.bnn_param.cov_sqrt.new_empty((self.bnn_param.cov_sqrt.size(0), ), requires_grad=False).normal_(z_scale)
            else:
                z_2 = None
            sample = self.bnn_param.mean + 0.5**0.5 * (torch.exp(self.bnn_param.log_std)*z_1 + self.bnn_param.cov_sqrt.t().matmul(z_2))            
            
        return sample, z_1, z_2

        """
        if not last_only:
            # feature extractor -------------------------------
            z_1_fe = torch.randn_like(self.fe_mean, requires_grad=False)
            rand_sample_fe = torch.exp(self.fe_log_std) * z_1_fe
            if not self.diag_only:
                z_2 = self.bnn_param.cov_sqrt.new_empty((self.bnn_param.cov_sqrt.size(0), ), requires_grad=False).normal_(z_scale)
                if hasattr(self, "fe_cov_sqrt"):
                    cov_sample_fe = self.fe_cov_sqrt.t().matmul(z_2)
                    cov_sample_fe /= (self.fe_cov_sqrt.size(0) - 1) ** 0.5
                    rand_sample_fe = 0.5**0.5 * (rand_sample_fe + cov_sample_fe)
                            
            sample_fe = self.fe_mean + rand_sample_fe
            # -------------------------------------------------

        ## last layer --------------------------------------
        z_1_ll = torch.randn_like(self.bnn_param.mean, requires_grad=False)
        rand_sample_ll = torch.exp(self.bnn_param['log_std']) * z_1_ll
        if not self.diag_only:            
            z_2 = self.bnn_param.cov_sqrt.new_empty((self.bnn_param.cov_sqrt.size(0), ), requires_grad=False).normal_(z_scale)
            cov_sample_ll = self.bnn_param['cov_sqrt'].t().matmul(z_2)
            cov_sample_ll /= (self.low_rank - 1)**0.5
            rand_sample_ll = 0.5**0.5 * (rand_sample_ll + cov_sample_ll)
        else:
            z_2 = None
        sample_ll = self.bnn_param['mean'] + rand_sample_ll
        ## -------------------------------------------------

        ## concatenate -------------------------------------
        if not last_only:
            sample = torch.cat((sample_fe, sample_ll))
            z_1 = torch.cat((z_1_fe, z_1_ll))
        else:
            sample = sample_ll
            z_1 = z_1_ll
        ## -------------------------------------------------

        return sample, z_1, z_2
        """
    
    def fish_inv(self, params, eta=1.0):
        '''
        Compute gradient of log probability w.r.t bnn params
        '''
        if self.last_layer:
            params = params[-self.ll_num_params:]

        soft_std = torch.exp(self.bnn_param['log_std']) # + self.var_clamp**0.5
        if not self.diag_only:
            cov_mat_lt = RootLazyTensor(self.bnn_param['cov_sqrt'].t())
            var_lt = DiagLazyTensor(soft_std**2)
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
            return [mean_fi, std_fi, cov_fi]
        else:
            return [mean_fi, std_fi]

        
    def get_mean_vector(self, unflatten=False):
        '''
        Load mean vector
        '''
        mean_param = torch.cat((self.fe_mean, self.bnn_param['mean']))
        if unflatten:
            return utils.unflatten_like_size(mean_param, self.backbone_shape)
        else:
            return mean_param


    def get_variance_vector(self, unflatten=False):
        '''
        Load variance vector (Not std)
        '''
        var_param = torch.cat((self.fe_log_std, self.bnn_param['log_std']))
        var_param = torch.exp(2*var_param)
        if unflatten:
            return utils.unflatten_like_size(var_param, self.backbone_shape)
        else:
            return var_param


    def get_covariance_matrix(self):
        '''
        Load covariance vector
        '''
        if self.diag_only:
            return None
        else:
            if hasattr(self, "fe_cov_sqrt"):
                cov_param = torch.cat((self.fe_cov_sqrt, self.bnn_param['cov_sqrt']), dim=1)
            else:
                cov_param = self.bnn_param['cov_sqrt']
            return cov_param
    
    
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
        if sabtl_model.last_layer:
            z_1 = z_1[-sabtl_model.ll_num_params:]

        rand_sample = (torch.exp(self.param_groups[0]['params'][1])) * z_1
        if not sabtl_model.diag_only:
            cov_sample = (self.param_groups[0]['params'][2].t().matmul(z_2)) / (sabtl_model.low_rank - 1)**0.5
            rand_sample = 0.5**0.5 * (rand_sample + cov_sample)
        sample = self.param_groups[0]['params'][0] + rand_sample
        
        # change sampled weight type list to dict 
        sample = utils.format_weights(sample, sabtl_model, sabtl_model.last_layer)
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
