"""Losses for sgld/sghmc networks"""
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
import torch
import torch.nn as nn


class GaussianPriorCELossShifted(nn.Module):
    """Scaled CrossEntropy + Gaussian prior"""

    def __init__(self, params, constant=1e6):
        super().__init__()
        means = params['mean']
        variance = params['variance']
        cov_mat_sqr = params['cov_mat_sqr']
        # Computes the Gaussian prior log-density.
        self.mvn = LowRankMultivariateNormal(means, cov_mat_sqr.t(), variance)
        self.constant = constant
        self.ce = nn.CrossEntropyLoss()
    
    def log_prob(self, params):
        return self.mvn.log_prob(params)

    def forward(self, logits, Y, N=1, params=None):
        nll = self.ce(logits, Y)
        log_prior_value = self.log_prob(params).sum() / N
        log_prior_value = torch.clamp(log_prior_value, min=-1e20, max=1e20)

        ne_en = nll - log_prior_value
        matrices = {'loss': ne_en, 'nll': nll, 'prior': log_prior_value}
        return matrices