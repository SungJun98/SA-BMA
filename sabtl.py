import torch
import torch.nn as nn
import numpy as np

import utils
from baselines.swag import swag_utils



class SABTL(torch.nn.Module):
    def __init__(
        self, backbone, low_rank=20,
        w_mean=None, prior_mu_init = 0.0,
        w_var=None, prior_var_scale = 0.1,
        diag_only=False, w_cov=None, prior_cov_scale = 1e-3,
        var_clamp=1e-30, sampling_scale=1.0
    ):
        super(SABTL, self).__init__()

        # setting for covariance
        self.diag_only = diag_only
        self.low_rank = low_rank

        # prior hyper-parameters
        self.prior_mu_init = prior_mu_init
        self.prior_var_scale = prior_var_scale
        self.prior_cov_scale = prior_cov_scale

        self.var_clamp = var_clamp
        self.sampling_scale = sampling_scale

        self.backbone = backbone
        self.num_params = sum(p.numel() for p in backbone.parameters())
        self.backbone_shape = [p.shape for p in backbone.parameters()]

        ## Add Mean, Var, Cov layer before Backbone Model ----------------------------------------------------
        self.bnn_param = nn.ParameterDict()

        # Mean
        if w_mean is not None:
            self.mean_param = nn.Parameter(w_mean)       
        else:
            self.mean_param = nn.Parameter(torch.ones(self.num_params) * self.prior_mu_init)

        self.bnn_param.update({"mean" : self.mean_param})

        # Variance (Diagonal Covariance)
        if w_var is not None:
            self.var_param = nn.Parameter(w_var)
        else:
            self.var_param = nn.Parameter(torch.ones(self.num_params) * self.prior_sigma_diag_scale)

        self.bnn_param.update({"var" : self.var_param})

        # Covariance (Off-Diagonal Covariance)
        if diag_only is False:
            if w_cov is not None:
                if type(w_cov) == list:
                    w_cov = torch.cat(w_cov, dim=1)         # cat covmat list as matrix
                    #TODO: change self.low_rank to w_cov
                    
                self.cov_param = nn.Parameter(w_cov)
                self.low_rank = w_cov.shape[0]
                
            else:
                self.cov_param = nn.Parameter(torch.ones(self.low_rank, self.num_params) * self.prior_sigma_off_diag_scale)

            self.bnn_param.update({"cov" : self.cov_param})
        # -----------------------------------------------------------------------------------------------------
    


    def forward(self, *args, **kwargs):
        # forward backbone model
        return self.backbone(*args, **kwargs)


    def sample(self, scale=1.0, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        # draw diagonal variance sample
        z_1 = torch.randn_like(self.bnn_param['mean'], requires_grad=False)
        rand_sample = self.bnn_param['var'] * z_1

        # if covariance draw low rank sample
        if not self.diag_only:
            z_2 = self.bnn_param['cov'].new_empty((self.bnn_param['cov'].size(0),), requires_grad=False).normal_()
            cov_sample = self.bnn_param['cov'].t().matmul(z_2)
            cov_sample /= (self.low_rank - 1) ** 0.5
            rand_sample += cov_sample
        else:
            z_2 = None

        # update sample with mean and scale
        sample = self.bnn_param['mean'] + scale**0.5 * rand_sample

        # unflatten like DNN model
        sample = utils.unflatten_like_size(sample, self.backbone_shape)
             
        # change sampled weight type list to dict 
        sample = self.format_weights(sample)
      
        return sample, z_1, z_2
         

    def format_weights(self, sample_w):
        state_dict = dict()
        for (name, _), w in zip(self.backbone.named_parameters(), sample_w):
            state_dict[name] = w
        return state_dict
    
    
    def get_mean_vector(self, unflatten=False):
        if unflatten:
            return utils.unflatten_like_size(self.bnn_param['mean'], self.backbone_shape)
        else:
            return self.bnn_param['mean']


    def get_variance_vector(self, unflatten=False):
        variances = torch.clamp(self.bnn_param['var'], self.var_clamp)
        if unflatten:
            return utils.unflatten_like_size(variances, self.backbone_shape)
        else:
            return variances


    def get_covariance_matrix(self, unflatten=False, eps=1e-10):
        if self.diag_only:
            raise RuntimeError("No covariance matrix was estimated!")        
        
        if unflatten:
            cov_mat = torch.matmul(self.bnn_param['cov'].t(), self.bnn_param['cov'])
            cov_mat /= (self.low_rank - 1)
            print(cov_mat.shape)

            # obtain covariance matrix by adding variances (+ eps for numerical stability) to diagonal and scaling
            var = torch.flatten(self.get_variance_vector()) + eps
            cov_mat.add_(torch.diag(var)).mul_(0.5)
        
            return cov_mat
        
        else:
            return self.bnn_param['cov']
    
    
    def load_state_dict(self, state_dict, strict=True):
        # if not self.diag_only:
        #     for module, name in self.params:
        #         mean = module.__getattr__("%s_mean" % name)
        #         module.__setattr__(
        #             "%s_cov_mat_sqrt" % name,
        #             mean.new_empty((rank, mean.numel())).zero_(),
        #         )
        super(SABTL, self).load_state_dict(state_dict, strict)


def second_sample(bnn_params, z_1, z_2, sabtl_model, scale=1.0):
    rand_sample = bnn_params[1] * z_1

    if sabtl_model.diag_only == False:
        cov_sample = bnn_params[2].t().matmul(z_2)
        cov_sample /= (sabtl_model.low_rank - 1) ** 0.5
        rand_sample += cov_sample
    
    # update sample with mean and scale
    sample = bnn_params[0] + scale**0.5 * rand_sample
    
    # unflatten like DNN model
    sample = utils.unflatten_like_size(sample, sabtl_model.backbone_shape)
             
    # change sampled weight type list to dict 
    sample = sabtl_model.format_weights(sample)
    
    return sample
      

## BSAM
class BSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, sabtl_model, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(BSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        
        self.num_params = sabtl_model.num_params
        self.backbone_shape = sabtl_model.backbone_shape
        
        self.diag_only = sabtl_model.diag_only
        self.low_rank = sabtl_model.low_rank
        

    @torch.no_grad()
    def first_step(self, eta=1.0, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                
                self.state[p]["old_p"] = p.data.clone()
                        
                ### Calculate fisher inverse ------------------------------------------------
                flat_grad = (p.grad)**2 + 1e-8      # add small value for numerical stability
                fish_inv = 1 / (1 + eta*flat_grad)
                # ---------------------------------------------------------------------------
                
                ### Calculate perturbation Delta_theta --------------------------------------
                Delta_p = group["rho"] * fish_inv * p.grad
                Delta_p = Delta_p / (torch.sqrt(p.grad * fish_inv * p.grad) + 1e-8) # add small value for numericaly stability
                # ---------------------------------------------------------------------------
                
                ### theta + Delta_theta
                p.add_(Delta_p)  # climb to the local maximum "w + e(w)"
                # ---------------------------------------------------------------------------
                
        if zero_grad: self.zero_grad()
        
        return self.param_groups[0]["params"]


    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()


    '''
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
    '''
