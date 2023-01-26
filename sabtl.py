import torch
import torch.nn as nn
import numpy as np

from baselines.swag import swag_utils


class SABTL(torch.nn.Module):
    def __init__(
        self, base, w_mean=None, w_var=None,
        no_cov_mat=False, w_cov=None,
        var_clamp=1e-30
    ):
        super(SABTL, self).__init__()

        self.params = list()

        self.no_cov_mat = no_cov_mat
        
        self.var_clamp = var_clamp

        self.base = base
        self.init_sabtl_parameters(params=self.params, w_mean=w_mean, w_var=w_var,
                                no_cov_mat=self.no_cov_mat, w_cov=w_cov)


    def forward(self, *args, **kwargs):
        '''
        base에서 forward가 아니라
        새로 정의된 모델에서 forward를 가야되는거 아닌가
        cf > bayesian-torch에서는 어떻게 했는지 함 보자
        '''
        return self.base(*args, **kwargs)


    def init_sabtl_parameters(self, params, w_mean=None, w_var=None,
                            no_cov_mat=True, w_cov=None):
        k_mean = 0
        k_var = 0
        
        for mod_name, module in self.base.named_modules():
            for name in list(module._parameters.keys()):
                if module._parameters[name] is None:
                    continue
                
                name_full = f"{mod_name}.{name}".replace(".", "-")
                data = module._parameters[name].data
                module._parameters.pop(name)

                ## Mean        
                if w_mean is not None:
                    s_mean = torch.prod(torch.tensor(data.shape))
                    module.register_parameter("%s_mean" % name_full,
                                    nn.Parameter(w_mean[k_mean : k_mean + s_mean].reshape(data.shape))
                                    )
                    k_mean += s_mean
                else:
                    module.register_parameter("%s_mean" % name_full,
                                    nn.Parameter(data.new_tensor(data))
                                    )

                ## Variance
                if w_var is not None:
                    s_var = torch.prod(torch.tensor(data.shape))
                    module.register_parameter("%s_diag_cov_sqrt" % name_full,
                                    nn.Parameter(w_var[k_var : k_var + s_var].reshape(data.shape))
                                    )
                    k_var += s_var
                else:
                    module.register_parameter("%s_diag_cov_sqrt" % name_full,
                                    nn.Parameter(data.new_ones(data.size()))
                                    )
                
                ## Covariance
                """
                if no_cov_mat is False:
                    '''
                    memory issue
                    '''
                    p = data.numel()    # number of parameters
                    
                    module.register_parameter("%s_off_diag_cov_sqrt" % name,
                    nn.Parameter(torch.sparse_coo_tensor(torch.tril_indices(p, p, dtype=torch.int),
                                torch.rand(int(p*(p+1)/2)),
                                (p, p)))
                    )
                """
                 
                params.append((module, name_full))
    


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
        if self.no_cov_mat:
            raise RuntimeError("No covariance matrix was estimated!")

        cov_mat_sqrt_list = []
        for module, name in self.params:
            cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
            cov_mat_sqrt_list.append(cov_mat_sqrt.cpu())
        '''
        # build low-rank covariance matrix
        cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)
        print(cov_mat_sqrt.shape)
        cov_mat = torch.matmul(cov_mat_sqrt.t(), cov_mat_sqrt)
        cov_mat /= (self.max_num_models - 1)
        print(cov_mat.shape)

        # obtain covariance matrix by adding variances (+ eps for numerical stability) to diagonal and scaling
        var = self.get_variance_vector() + eps
        cov_mat.add_(torch.diag(var)).mul_(0.5)

        return cov_mat
        '''
        return cov_mat_sqrt_list

    
    
    def sample(self, scale=1.0, cov=False, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        mean_list = []
        var_list = []
        if cov:
            cov_mat_sqrt_list = []

        for (module, name) in self.params:
            mean_list.append(module.__getattr__("%s_mean" % name).cpu())
            var_list.append(module.__getattr__("%s_diag_cov_sqrt" % name).cpu())
            if cov:
                cov_mat_sqrt_list.append(module.__getattr__("%s_off_diag_cov_sqrt" % name).cpu())

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
            cov_sample /= (self.max_num_models - 1) ** 0.5
            rand_sample += cov_sample

        # update sample with mean and scale
        sample = (mean + scale**0.5 * rand_sample).unsqueeze(0)

        # unflatten new sample like the mean sample
        samples_list = swag_utils.unflatten_like(sample, mean_list)
        self.set_model_parameters(samples_list)

        return samples_list #, z
 
 
    def set_model_parameters(self, parameter):
        for (name, module), param in zip(self.base.named_parameters(), parameter):
            module.__setattr__(name.split("-")[-1], param.cuda())



    # def load_state_dict(self, state_dict, strict=True):
    #     if not self.no_cov_mat:
    #         for module, name in self.params:
    #             mean = module.__getattr__("%s_mean" % name)
    #             '''
    #             diag_cov_sqrt load 
    #             off_diag_cov_sqrt load
    #             '''
    #     super(SABTL, self).load_state_dict(state_dict, strict)
