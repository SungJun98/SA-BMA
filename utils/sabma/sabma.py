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
    