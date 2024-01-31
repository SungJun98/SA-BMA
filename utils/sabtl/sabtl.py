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
        tr_layer="last_layer",
    ):
        """
        TODO : last layer random initialization 코드 추가 (argument 받아서)
        """
        super(SABTL, self).__init__()
        
        self.var_clamp = var_clamp
        self.diag_only = diag_only
        self.low_rank = low_rank
        self.tr_layer = tr_layer
        
        self.backbone = backbone
        self.src_bnn = src_bnn
        
        self.full_model_shape = list()
        for p in backbone.parameters():
            p.requires_grad = False
            self.full_model_shape.append(p.shape)

        # Get Last Layer Name and Feature Extractor / Last Layer Shape
        # fe_layer -> non_tr_layer, last_layer -> tr_layer
        if tr_layer == "last_layer":
            for name, _ in self.backbone.named_modules():
                self.tr_layer_name = name
            self.non_tr_layer_shape = self.full_model_shape[:-2]
            self.tr_layer_shape = self.full_model_shape[-2:]
        elif tr_layer == "last_block":
            raise NotImplementedError("Need code for last block SA-BMA")
        elif tr_layer == "full_layer":
            self.tr_layer_shape = self.full_model_shape
        
        if w_mean is None:
            raise NotImplementedError("We need Pre-trained weight to define model")
        
        # Get the number of total / last_layer / Feature Extractor parameters
        self.total_num_params = 0 ; self.tr_num_params = 0
        for name, param in self.backbone.named_parameters():
            self.total_num_params += param.numel()
            if name.split('.')[0] == self.tr_layer_name:
                self.tr_num_params += param.numel()

        ### Add Mean, Var, Cov layer ---------------------------------------------------------------
        self.bnn_param = nn.ParameterDict()

        ## Mean
        """Load Mean Params"""
        if self.tr_layer == "last_layer":
            self.register_buffer('fe_mean', w_mean[:-self.tr_num_params])
            self.bnn_param.update({"mean" : nn.Parameter(w_mean[-self.tr_num_params:])})
        elif self.tr_layer == "last_block":
            raise NotImplementedError("Need code for last block SA-BMA")
        elif self.tr_layer == "full_layer":
            self.bnn_param.update({"mean" : nn.Parameter(w_mean)})

        ## Variance
        """Load Diagonal Variance Params"""
        w_var = torch.clamp(w_var, self.var_clamp)
        w_log_std = 0.5 * torch.log(w_var)              # log_std
        if self.tr_layer == "last_layer":
            self.register_buffer('fe_log_std', w_log_std[:-self.tr_num_params])
            self.bnn_param.update({"log_std" : nn.Parameter(w_log_std[-self.tr_num_params:] * var_scale)})
        elif self.tr_layer == "last_block":
            raise NotImplementedError("Need code for last block SA-BMA")
        elif self.tr_layer == "full_layer":
            self.bnn_param.update({"log_std" : nn.Parameter(w_log_std * var_scale)})
        
        ## Covariance
        """Load Covariance Params"""   
        if src_bnn == 'swag':
            if not self.diag_only:
                if w_cov_sqrt is not None:
                    if type(w_cov_sqrt) == list:
                        # cat covmat list as matrix
                        w_cov_sqrt = torch.cat(w_cov_sqrt, dim=1) 
                    
                    self.fe_low_rank = w_cov_sqrt.size(0)
                    if self.tr_layer == "last_layer":
                        self.register_buffer('fe_cov_sqrt', w_cov_sqrt[:, :-self.tr_num_params])
                        if self.low_rank <= w_cov_sqrt.size(0):
                            w_cov_sqrt =w_cov_sqrt[:self.low_rank,-self.tr_num_params:]
                        else:
                            raise NotImplementedError("SABTL set lower low-rank than Pre-trained BNNs")
                        self.bnn_param.update({"cov_sqrt" : nn.Parameter(w_cov_sqrt * cov_scale)})                        
                    elif self.tr_layer == "last_block":
                        raise NotImplementedError("Need code for last block SA-BMA")
                    elif self.tr_layer == "full_layer":
                        if self.low_rank <= w_cov_sqrt.size(0):
                            w_cov_sqrt =w_cov_sqrt[:self.low_rank,:]
                        else:
                            raise NotImplementedError("SABTL set lower low-rank than Pre-trained BNNs")
                        self.bnn_param.update({"cov_sqrt" : nn.Parameter(w_cov_sqrt * cov_scale)})
                else:
                    # Random Initialization Covariance
                    self.fe_low_rank = self.low_rank
                    self.bnn_param.update({"cov_sqrt" : nn.Parameter(torch.randn((self.low_rank, self.tr_num_params))*cov_scale)})
        # elif src_bnn == 'la':
        #     raise RuntimeError("Add Load for Laplace Approximation")
        
        # elif src_bnn == 'vi':
        #     if not self.diag_only:
        #         self.bnn_param.update({"cov_sqrt" : nn.Parameter(torch.randn((self.low_rank, self.ll_num_params))*cov_scale)})
        
        # print(f"Load covariance of weight from pre-trained {src_bnn} model")
        # -----------------------------------------------------------------------------------------------------    


    def forward(self, params, input):
        return nn.utils.stateless.functional_call(self.backbone, params, input)
    
    
    def sample(self, z_scale=1.0, tr_param_only=False):
        '''
        Sample weight from bnn params
        '''
        if (not tr_param_only) and (self.tr_layer != "full_layer"):
            # sampling full params
            if self.tr_layer == "last_layer":
                # in case of training last layer
                # freezed layer ------------------------------------------------------
                z_1_fe = torch.randn_like(self.fe_mean, requires_grad=False)
                rand_sample_fe = torch.exp(self.fe_log_std) * z_1_fe
                if not self.diag_only:
                    z_2_fe = self.bnn_param.cov_sqrt.new_empty((self.fe_low_rank, ), requires_grad=False).normal_(z_scale)
                    cov_sample_fe = self.fe_cov_sqrt.t().matmul(z_2_fe)    
                    if self.fe_low_rank > 1:
                        cov_sample_fe /= (self.fe_low_rank - 1) ** 0.5
                    rand_sample_fe = 0.5**0.5 * (rand_sample_fe + cov_sample_fe)
                    
                sample_fe = self.fe_mean + rand_sample_fe
                # ---------------------------------------------------------------------
                
                # training layer ------------------------------------------------------
                z_1_tr = torch.randn_like(self.bnn_param.mean, requires_grad=False)
                rand_sample_tr = torch.exp(self.bnn_param.log_std) * z_1_tr
                if not self.diag_only:
                    z_2_tr = z_2_fe[:self.low_rank]
                    # z_2_ll = self.bnn_param.cov_sqrt.new_empty((self.bnn_param.cov_sqrt.size(0), ), requires_grad=False).normal_(z_scale)
                    cov_sample_tr = self.bnn_param.cov_sqrt.t().matmul(z_2_tr)
                    if self.low_rank > 1:
                        cov_sample_tr /= (self.low_rank - 1)**0.5
                    
                    rand_sample_tr = 0.5**0.5 * (rand_sample_tr + cov_sample_tr)
                    
                sample_ll = self.bnn_param['mean'] + rand_sample_tr
                # --------------------------------------------------------------------- 
                
                ## concatenate -------------------------------------
                sample = torch.cat((sample_fe.detach(), sample_ll))
                z_1 = torch.cat((z_1_fe, z_1_tr))
                z_2 = z_2_fe
                ## -------------------------------------------------
                
                
            elif self.tr_layer == "last_block":
                # in case of training last block
                raise NotImplementedError("Need code for last block SA-BMA")
                
        else:
            # sampling only trainig params
            z_1 = torch.randn_like(self.bnn_param['mean'], requires_grad=False)
            rand_sample = torch.exp(self.bn_param['log_std']) * z_1
            
            if not self.diag_only:
                z_2 = self.bnn_param['cov_sqrt'].new_empty((self.low_rank, ), requires_grad=False).normal_(z_scale)
                cov_sample = self.bnn_param['cov_sqrt'].t().matmul(z_2)
                if self.low_rank > 1:
                    cov_sample /= (self.low_rank - 1) ** 0.5
                rand_sample = 0.5**0.5 * (rand_sample + cov_sample)
            
            sample = self.bnn_param['mean'] + rand_sample 
        
        
        if self.diag_only:
            z_2 = None
        
        return sample, z_1, z_2



    def log_grad(self, params, approx='full', eta=1.0):
        '''
        Compute gradient of log probability w.r.t bnn params
        '''
        if self.tr_layer in ["last_layer", "last_block"]:
            params = params[-self.tr_num_params:]

        soft_std = torch.exp(self.bnn_param['log_std'])
        if not self.diag_only:
            cov_mat_lt = RootLazyTensor(self.bnn_param['cov_sqrt'].t())
            var_lt = DiagLazyTensor(soft_std**2)
            covar = AddedDiagLazyTensor(var_lt, cov_mat_lt).add_jitter(1e-8)
        else:
            covar = DiagLazyTensor(soft_std**2)
        qdist = MultivariateNormal(self.bnn_param['mean'], covar)
        with gpytorch.settings.num_trace_samples(1) and gpytorch.settings.max_cg_iterations(25):
            log_prob =  qdist.log_prob(params)
        
        # mean
        mean_log_grad = covar.inv_matmul((params - self.bnn_param['mean']))   ## calculate derivative manually (gpytorch version)
        # print(f"Mean FI / nan : {torch.sum(torch.isnan(mean_fi))} / max {torch.max(mean_fi)}")
        
        # diagonal variance
        std_log_grad = torch.autograd.grad(log_prob, self.bnn_param['log_std'], retain_graph=True)[0]
        # print(f"log_std FI / nan : {torch.sum(torch.isnan(std_fi))} / max {torch.max(std_fi)}")
        
        # off-diagonal covariance
        cov_log_grad = torch.autograd.grad(log_prob, self.bnn_param['cov_sqrt'], retain_graph=True)[0]
        if approx == 'full':
            cov_log_grad = torch.flatten(cov_log_grad)
        elif approx == 'diag':
            cov_log_grad = torch.pow(torch.flatten(cov_log_grad), 2)
            cov_log_grad = 1 / (1 + eta*cov_log_grad)
            
        # inf_mask = (cov_fi == float('inf'))
        # cov_fi[inf_mask] = 1
        
        # print(f"Cov FI / nan : {torch.sum(torch.isnan(cov_fi))} / max {torch.max(cov_fi)}")
        
        return [mean_log_grad, std_log_grad, cov_log_grad]
    

        
    def get_mean_vector(self, unflatten=False):
        '''
        Load mean vector
        '''
        if self.tr_layer in ["last_layer", "last_block"]:
            mean_param = torch.cat((self.fe_mean, self.bnn_param.mean))
        else:
            mean_param = self.bnn_param.mean
            
        if unflatten:
            return utils.unflatten_like_size(mean_param, self.backbone_shape)
        else:
            return mean_param


    def get_variance_vector(self, unflatten=False):
        '''
        Load variance vector (Not std)
        '''
        if self.tr_layer in ["last_layer", "last_block"]:
            var_param = torch.cat((self.fe_log_std, self.bnn_param['log_std']))
        else:
            var_param = self.bnn_param.log_std
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
                fe_cov_param = self.fe_cov_sqrt
                cov_param = self.bnn_param['cov_sqrt']
                return {"fe_cov_sqrt" : fe_cov_param, "cov_sqrt" : cov_param}
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
                        # print(f"p / nan : {torch.sum(torch.isnan(p))} / max : {torch.max(p)}")
                        # print(f"Delta_p / nan : {torch.sum(torch.isnan(Delta_p))} / max : {torch.max(Delta_p)}")
                        print(f"p : {p}")
                        print(f"Delta_p : {Delta_p}")
                        # ---------------------------------------------------------------------------
                if zero_grad: self.zero_grad()
                
        else:
            raise NotImplementedError("Need to be fixed")
            """
            for group in self.param_groups:
                for idx, p in enumerate(group["params"]):
                    if p.grad is None: continue
                    self.state[p]["old_p"] = p.data.clone()
                    ## Calculate perturbation Delta_theta --------------------------------------
                    # Delta_p = group["rho"] * log_grad[idx].to('cuda') * p.grad
                    # Delta_p = Delta_p / (torch.sqrt(p.grad * log_grad[idx].to('cuda') * p.grad) + 1e-12) # add small value for numericaly stability
                    # ---------------------------------------------------------------------------                        
                    ## theta + Delta_theta
                    p.add_(Delta_p)  # climb to the local maximum "w + e(w)"
                    # ---------------------------------------------------------------------------
                    del Delta_p
            if zero_grad: self.zero_grad()
            """


    def second_sample(self, z_1, z_2, sabtl_model):
        '''
        Sample from perturbated bnn parameters with pre-selected z_1, z_2
        '''
        if sabtl_model.tr_layer in ["last_layer", "last_block"]:
            z_1 = z_1[-sabtl_model.tr_num_params:]

        # diagonal variance
        rand_sample = (torch.exp(self.param_groups[0]['params'][1])) * z_1
        
        # covariance
        if not sabtl_model.diag_only:
            cov_sample = (self.param_groups[0]['params'][2].t().matmul(z_2[:sabtl_model.low_rank]))
            if sabtl_model.low_rank > 1:
                cov_sample /= (sabtl_model.low_rank - 1)**0.5
            rand_sample = 0.5**0.5 * (rand_sample + cov_sample)
        sample = self.param_groups[0]['params'][0] + rand_sample
        
        # change sampled weight type list to dict 
        sample = utils.format_weights(sample, sabtl_model, sabtl_model.tr_layer)
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
    

    '''
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
    '''
