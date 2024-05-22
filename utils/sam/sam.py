import torch
import utils.utils as utils

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)


    @torch.no_grad()
    def first_step(self, zero_grad=False, amp=True):
        if amp:
            with torch.cuda.amp.autocast():
                grad_norm = self._grad_norm()
                for group in self.param_groups:
                    scale = group["rho"] / (grad_norm + 1e-12)

                    for p in group["params"]:
                        if p.grad is None: continue
                        self.state[p]["old_p"] = p.data.clone()
                        e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                        p.add_(e_w)  # climb to the local maximum "w + e(w)"
                if zero_grad: self.zero_grad()
        
        else:
            grad_norm = self._grad_norm()
            for group in self.param_groups:
                scale = group["rho"] / (grad_norm + 1e-12)
                
                for p in group["params"]:
                    if p.grad is None: continue
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                    p.add_(e_w)  # climb to the local maximum "w + e(w)"

            if zero_grad: self.zero_grad()


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
                    p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

            self.base_optimizer.step()  # do the actual "sharpness-aware" update

            if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        self.base_optimizer.step(closure)
        # assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        # closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        # self.first_step(zero_grad=True)
        # closure()
        # self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


    def second_sample(self, z_1, z_2, sabtl_model):
        '''
        Sample from perturbated bnn parameters with pre-selected z_1, z_2
        '''
        if sabtl_model.last_layer:
            z_1 = z_1[-sabtl_model.ll_num_params:]

        # diagonal variance
        rand_sample = (torch.exp(self.param_groups[0]['params'][1])) * z_1
        
        # covariance
        if not sabtl_model.diag_only:
            cov_sample = (self.param_groups[0]['params'][2].t().matmul(z_2)) / (sabtl_model.low_rank - 1)**0.5
            rand_sample = 0.5**0.5 * (rand_sample + cov_sample)
        sample = self.param_groups[0]['params'][0] + rand_sample
        
        # change sampled weight type list to dict 
        sample = utils.format_weights(sample, sabtl_model, sabtl_model.last_layer)
        return sample




## FisherSAM
class FSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(FSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, eta=1.0, zero_grad=False):
        with torch.cuda.amp.autocast():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    self.state[p]["old_p"] = p.data.clone()

                    flat_grad = (p.grad.view(-1))**2 + 1e-8
                
                    fish_inv = 1 / (1 + eta*flat_grad)
                    e_w = group["rho"] * torch.mul(fish_inv, flat_grad) / torch.sqrt(torch.dot(fish_inv, (flat_grad**2)))

                    # unflatten fish_inv like p
                    e_w = e_w.view(p.grad.shape)

                    p.add_(e_w)  # climb to the local maximum "w + e(w)"

            if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        with torch.cuda.amp.autocast():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

    @torch.no_grad()
    def step(self, closure=None):
        self.base_optimizer.step(closure)