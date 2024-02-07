# https://github.com/runame/laplace-redux/blob/main/baselines/swag/swag.py
import torch
import numpy as np

from . import swag_utils


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
        return swag_utils.flatten(mean_list)


    def get_variance_vector(self):
        mean_list = []
        sq_mean_list = []

        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            mean_list.append(mean.cpu())
            sq_mean_list.append(sq_mean.cpu())

        mean = swag_utils.flatten(mean_list)
        sq_mean = swag_utils.flatten(sq_mean_list)

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

        mean = swag_utils.flatten(mean_list)
        sq_mean = swag_utils.flatten(sq_mean_list)

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
        samples_list = swag_utils.unflatten_like(sample, mean_list)

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
    
