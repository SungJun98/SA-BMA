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
                if w_cov == list:
                    w_cov = torch.cat(w_cov, dim=1)         # cat covmat list as matrix
                self.cov_param = nn.Parameter(w_cov)
            else:
                self.cov_param = nn.Parameter(torch.ones(self.num_params) * self.prior_sigma_off_diag_scale)

            self.bnn_param.update({"cov" : self.cov_param})
        # -----------------------------------------------------------------------------------------------------
    


    def forward(self, *args, **kwargs):
        # parameter, z_1, z_2 = self.sample(scale=self.sampling_scale)
        # self.set_sampled_parameters(parameter=parameter)
        # forward backbone model
        return self.backbone(*args, **kwargs) # , z_1, z_2


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
        


    def get_mean_vector(self):
        return utils.unflatten_like_size(self.bnn_param['mean'], self.backbone_shape)


    def get_variance_vector(self):
        variances = torch.clamp(self.bnn_param['var'], self.var_clamp)
        return utils.unflatten_like_size(variances, self.backbone_shape)


    def get_covariance_matrix(self, full_cov_mat=False, eps=1e-10):
        if self.diag_only:
            raise RuntimeError("No covariance matrix was estimated!")        
        
        if full_cov_mat:
            cov_mat = torch.matmul(self.bnn_param['cov'].t(), self.bnn_param['cov'])
            cov_mat /= (self.low_rank - 1)
            print(cov_mat.shape)

            # obtain covariance matrix by adding variances (+ eps for numerical stability) to diagonal and scaling
            var = torch.flatten(self.get_variance_vector()) + eps
            cov_mat.add_(torch.diag(var)).mul_(0.5)
        
            return cov_mat
        
        else:
            self.bnn_param['cov']
    
    
    
    




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
        self.low_rank = sabtl_model.low_rank

    @torch.no_grad()
    def first_step(self, z_1, z_2, eta=1.0, zero_grad=False):
        for group in self.param_groups:
            bnn_grad = list()
            dnn_grad = list()
            for p in group["params"]:
                if p.grad is None: continue
                
            ### Save BNN weight and Flatten DNN grad --------------------------------------
                ## Save BNN weight
                # Mean & Variance
                if len(p) == self.num_params:
                    self.state[p]["old_p"] = p.data.clone()
                    bnn_grad.append(p.grad.view(-1))
                # Covariance
                elif p.shape  == torch.Size([self.low_rank, self.num_params]):
                    set_cov = True
                    self.state[p]["old_p"] = p.data.clone()
                    bnn_grad.append(p.grad.view(-1))
                
                ## Flatten DNN grad
                else:
                    dnn_grad.append(p.grad.view(-1))
            
            bnn_grad = torch.cat(bnn_grad)                  # [2p+Kp]
            dnn_grad = torch.cat(dnn_grad)                  # [p]
            print(f"BNN gradient shape : {bnn_grad.shape}")
            print(f"DNN gardient shape : {dnn_grad.shape}")
            break
            # ---------------------------------------------------------------------------
                    
            ### Calculate gradient^T * A ------------------------------------------------
            A_var = dnn_grad * z_1      # [p]
            l_A = torch.cat([dnn_grad, A_var])
            
            if set_cov:
                # draw low-rank covariance
                A_cov = torch.outer(dnn_grad, z_2).view(-1)            # [Kp]
                l_A =  torch.cat([l_A, A_cov])                         # [2p+Kp]
            # ---------------------------------------------------------------------------
                
        
            ### Calculate fisher inverse ------------------------------------------------
            bnn_grad += + 1e-8      # add small value for numerical stability
        
            fish_inv = 1 / (1 + eta*(bnn_grad**2))                      # [2p+Kp]
            # ---------------------------------------------------------------------------
            
            ### Calculate perturbation Delta_theta --------------------------------------
            Delta_theta = group["rho"] * fish_inv * l_A                         # [2p+Kp]
            Delta_theta = Delta_theta / torch.sqrt(torch.dot(l_A * fish_inv, l_A.t()))  # [2p+Kp]
            # ---------------------------------------------------------------------------
            
            ### theta + Delta_theta
            bnn_grad.add_(Delta_theta)  # climb to the local maximum "w + e(w)"
            # ---------------------------------------------------------------------------
            
        if zero_grad: self.zero_grad()



    '''
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
