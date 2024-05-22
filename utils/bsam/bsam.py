import torch

class bSAM(torch.optim.Optimizer): 
    def __init__(self, params, Ndata, lr=1e-3, betas=(0.9, 0.999), 
                 rho=0.05, s_init=1, damping=0.1, weight_decay=0,
                 noise_scale=1e-4, **kwargs): 
        defaults = dict(Ndata=Ndata, rho=rho, betas=betas, s_init=s_init,
                        damping=damping, lr=lr, noise_scale=noise_scale, **kwargs)
        super(bSAM, self).__init__(params, defaults) 
        self.gamma = weight_decay
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["step"] = 0
                self.state[p]["gm"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                self.state[p]["s"] = s_init * torch.ones_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def add_noise(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                # noisy_param : p + e
                self.state[p]["old_p"] = p.data.clone()
                # get a random gaussian variable for every parameter
                noise = torch.normal(0, 1, size=p.shape)
                p.add_(torch.sqrt(1.0 / (group["Ndata"] * self.state[p]["s"])) * noise.to(p),
                       alpha=group["noise_scale"])

                # ## another implementation
                # noise = torch.normal(mean=0.0, 
                #                      std=torch.sqrt(1.0 / (group["Ndata"] * 1e20 * self.state[p]["s"])))
                # p.add_(noise)

        if zero_grad: self.zero_grad()
         
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        self.noisy_grad = [p.grad for group in self.param_groups for p in group['params']]

        for group in self.param_groups: 
            for p in group["params"]:
                if p.grad is None: continue
                # get back to "p" from "p + e (noisy params)"
                p.data = self.state[p]["old_p"]
                self.state[p]["old_p"] = p.data.clone()

                # perturbed params : p + eps
                # climb to the local maximum
                eps = group["rho"] * p.grad / (self.state[p]["s"] + 1e-12)
                p.add_(eps)

                # ## another implementation
                # scale = group["rho"] / (self.state[p]["s"] + 1e-12)
                # eps = p.grad * scale.to(p)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                # get back to "p" from "p + eps"
                p.data = self.state[p]["old_p"]

        self.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None): 
        for group in self.param_groups:
            for (p, g) in zip(group["params"], self.noisy_grad):
                if p.grad is None: continue
                self.state[p]["step"] += 1
                
                gs = torch.sqrt(self.state[p]["s"].mul(torch.pow(g, 2)))

                beta1, beta2 = group["betas"]

                # Decay the first and second moment running average coefficient
                self.state[p]["gm"].mul_(beta1).add_(p.grad+group['damping']*p, alpha=1-beta1)
                self.state[p]["s"].mul_(beta2).add_(gs+group['damping']+self.gamma)

                gm = self.state[p]["gm"]
                s = self.state[p]["s"]

                p.data.addcdiv_(gm, s, value=-group["lr"])