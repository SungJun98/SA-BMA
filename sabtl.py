import torch
import torch.nn as nn
import numpy as np

import utils
from baselines.swag import swag_utils

"""
class SABTL(torch.nn.Module):
    def __init__(
        self, base, low_rank=20,
        w_mean=None, prior_mu_init = 0.0,
        w_var=None, prior_sigma_diag_scale = 0.1,
        diag_only=False, w_cov=None, prior_sigma_off_diag_scale = 0.1,
        var_clamp=1e-30
    ):
        super(SABTL, self).__init__()

        self.params = list()

        self.diag_only = diag_only
        self.low_rank = low_rank

        self.prior_mu_init = prior_mu_init
        self.prior_sigma_diag_scale = prior_sigma_diag_scale
        self.prior_sigma_off_diag_scale = prior_sigma_off_diag_scale

        self.var_clamp = var_clamp

        self.base = base

        self.init_sabtl_parameters(params=self.params, w_mean=w_mean, w_var=w_var,
                                diag_only=self.diag_only, w_cov=w_cov)
    


    def forward(self, *args, **kwargs):
        sample_param = self.sample(1.0)
        self.set_sampled_parameters(self.base_model, sample_param)
        '''
        base model에서 forward
        '''
        return self.base_model


    def init_sabtl_parameters(self, params, w_mean=None, w_var=None,
                            diag_only=True, w_cov=None):
        
        if (w_cov is not None) and (w_cov == list):
            # cat covmat list as matrix
            w_cov = torch.cat(w_cov, dim=1)
        
        k_mean = 0
        k_var = 0
        k_cov = 0

        for mod_name, module in self.base.named_modules():
            for name in list(module._parameters.keys()):
                if module._parameters[name] is None:
                    continue
                
                name_full = f"{mod_name}.{name}".replace(".", "-")
                data = module._parameters[name].data
                module._parameters.pop(name)

                ## Mean --------------------------------------------------------------------------------------------------------------------------------
                if w_mean is not None:
                    s_mean = torch.prod(torch.tensor(data.shape))
                    module.register_parameter("%s_mean" % name_full,
                                    nn.Parameter(w_mean[k_mean : k_mean + s_mean].reshape(data.shape))
                                    )
                    k_mean += s_mean
                else:
                    module.register_parameter("%s_mean" % name_full,
                                    nn.Parameter(self.prior_mu_init * data.new_ones(data).reshape(data.shape))
                                    )
                # --------------------------------------------------------------------------------------------------------------------------------------

                ## Variance ----------------------------------------------------------------------------------------------------------------------------
                if w_var is not None:
                    s_var = torch.prod(torch.tensor(data.shape))
                    module.register_parameter("%s_diag_cov_sqrt" % name_full,
                                    nn.Parameter(w_var[k_var : k_var + s_var].reshape(data.shape))
                                    )
                    k_var += s_var
                else:
                    module.register_parameter("%s_diag_cov_sqrt" % name_full,
                                    nn.Parameter(self.prior_sigma_diag_scale * data.new_ones(data.size()).reshape(data.shape))
                                    )
                # -------------------------------------------------------------------------------------------------------------------------------------

                ## Covariance -------------------------------------------------------------------------------------------------------------------------
                if diag_only is False:
                    s_cov = torch.prod(torch.tensor(data.shape))

                    if w_cov is not None:
                        off_diag_cov_sqrt_shape = [w_cov.size(0)] + list(data.shape)
                        off_diag_cov_sqrt_shape[0], off_diag_cov_sqrt_shape[1] = off_diag_cov_sqrt_shape[1], off_diag_cov_sqrt_shape[0] # permute

                        module.register_parameter("%s_off_diag_cov_sqrt" % name_full,
                                    nn.Parameter(w_cov[:, k_cov : k_cov + s_cov].reshape(off_diag_cov_sqrt_shape))
                                    )
                    else:
                        off_diag_cov_sqrt_shape = [self.low_rank] + list(data.shape)
                        off_diag_cov_sqrt_shape[0], off_diag_cov_sqrt_shape[1] = off_diag_cov_sqrt_shape[1], off_diag_cov_sqrt_shape[0] # permute

                        module.register_parameter("%s_off_diag_cov_sqrt" % name_full,
                                    nn.Parameter(self.prior_sigma_off_diag_scale * data.new_ones((self.low_rank, s_cov)).reshape())
                                    )
                    
                    k_cov += s_cov
                # -------------------------------------------------------------------------------------------------------------------------------------

                params.append((module, name_full))
    


    def sample(self, scale=1.0, cov=False, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        mean_list = []
        var_list = []
        if cov:
            cov_mat_sqrt_list = []

        for (module, name) in self.params:
            mean_list.append(module.__getattr__("%s_mean" % name))
            var_list.append(module.__getattr__("%s_diag_cov_sqrt" % name))
            if cov:
                cov_mat_sqrt_list.append(module.__getattr__("%s_off_diag_cov_sqrt" % name))

        mean = swag_utils.flatten(mean_list)
        var = swag_utils.flatten(var_list)
        var = torch.clamp(var, self.var_clamp)
        
        z = torch.randn_like(var, requires_grad=False)
        # draw diagonal variance sample
        rand_sample = var * z

        # if covariance draw low rank sample
        if cov:
            cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)
            eps = cov_mat_sqrt.new_empty((cov_mat_sqrt.size(0),), requires_grad=False).normal_()
            cov_sample = cov_mat_sqrt.t().matmul(eps)
            cov_sample /= (self.low_rank - 1) ** 0.5
            rand_sample += cov_sample

        # update sample with mean and scale
        sample = (mean + scale**0.5 * rand_sample).unsqueeze(0)

        # unflatten new sample like the mean sample
        samples_list = swag_utils.unflatten_like(sample, mean_list)

        return samples_list
 
##############################################################################


    def get_mean_vector(self):
        mean_list = []
        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)
            mean_list.append(mean.cpu())
        return swag_utils.flatten(mean_list)


    def get_variance_vector(self):
        var_list = []
        for module, name in self.params:
            var = module.__getattr__("%s_diag_cov_sqrt" % name)
            var_list.append(var.cpu())

        variances = swag_utils.flatten(var_list)
        variances = torch.clamp(variances, self.var_clamp)
        return variances


    def get_covariance_matrix(self, eps=1e-10):
        if self.diag_only:
            raise RuntimeError("No covariance matrix was estimated!")

        cov_mat_sqrt_list = []
        for module, name in self.params:
            cov_mat_sqrt = module.__getattr__("%s_off_diag_cov_sqrt" % name)
            cov_mat_sqrt_list.append(cov_mat_sqrt.cpu())

        # build low-rank covariance matrix
        cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)
        print(f"Lower Triangular Matrix Shape : {cov_mat_sqrt.shape} (low_rank, # of params)")
        '''
        cov_mat = torch.matmul(cov_mat_sqrt.t(), cov_mat_sqrt)
        cov_mat /= (self.low_rank - 1)
        print(cov_mat.shape)

        # obtain covariance matrix by adding variances (+ eps for numerical stability) to diagonal and scaling
        var = self.get_variance_vector() + eps
        cov_mat.add_(torch.diag(var)).mul_(0.5)
        '''
        return cov_mat_sqrt


    # def load_state_dict(self, state_dict, strict=True):
    #     if not self.diag_only:
    #         for module, name in self.params:
    #             mean = module.__getattr__("%s_mean" % name)
    #             '''
    #             diag_cov_sqrt load 
    #             off_diag_cov_sqrt load
    #             '''
    #     super(SABTL, self).load_state_dict(state_dict, strict)


def set_sampled_parameters(base_model, parameter):
    for base_param, param in zip(base_model.parameters(), parameter):
        base_param.data = param.cuda()
"""


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
        self.dnn_shape = [p.shape for p in backbone.parameters()]

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
        parameter, z_1, z_2 = self.sample(scale=self.sampling_scale)
        self.set_sampled_parameters(parameter=parameter)
        # forward backbone model
        return self.backbone(*args, **kwargs), z_1, z_2


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
        sample = utils.unflatten_like_size(sample, self.dnn_shape)
        
        return sample, z_1, z_2
 

    def set_sampled_parameters(self, parameter):
        for base_param, param in zip(self.backbone.parameters(), parameter):
            base_param.data = param.cuda()


    def get_mean_vector(self):
        return utils.unflatten_like_size(self.bnn_param['mean'], self.dnn_shape)


    def get_variance_vector(self):
        variances = torch.clamp(self.bnn_param['var'], self.var_clamp)
        return utils.unflatten_like_size(variances, self.dnn_shape)


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
        self.dnn_shape = sabtl_model.dnn_shape
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
