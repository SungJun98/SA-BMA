"""Losses for sgld/sghmc networks"""
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
import torch
import torch.nn as nn
from ..solo_learn.utils.classification_dataloader import prepare_data as prepare_data_classification
from torchvision.models import resnet50
from ..prior.prior_ssl_utils.utils import load_weights_simclr
from torchmetrics.classification import MulticlassCalibrationError

class CustomCEN(nn.Module):
    """Wrapper for Cross entropy loss that gets also N and params.""" #

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.ece = MulticlassCalibrationError(num_classes=10, n_bins=50, norm='l1')

    def forward(self, logits, Y, N=1, params=1):
        energy = self.ce(logits, Y)
        energy_ece = self.ece(logits, Y)
        matrices = {'loss': energy, 'nll': energy, 'ece':energy_ece, 'prior': torch.Tensor([0])}
        return matrices


class GaussianPriorCELossShifted(nn.Module):
    """Scaled CrossEntropy + Gaussian prior"""

    def __init__(self, params, dataset_name, constant=1e6):
        super().__init__()
        means = params['mean']
        variance = params['variance']
        cov_mat_sqr = params['cov_mat_sqr']
        # Computes the Gaussian prior log-density.
        self.mvn = LowRankMultivariateNormal(means, cov_mat_sqr.t(), variance)
        self.constant = constant
        self.ce = nn.CrossEntropyLoss()

        dataset_labels = {
            "eurosat": 10,
            "dtd": 47,
            "oxford_flowers": 102,
            "oxford_pets": 37,
            "food101": 101,
            "ucf101": 101,
            "fgvc_aircraft": 100,
            "cifar100" : 100,
            "cifar10" : 10
        }

        print(dataset_labels[dataset_name])

        self.ece = MulticlassCalibrationError(num_classes=dataset_labels[dataset_name], n_bins=50, norm='l1')


    
    def log_prob(self, params):
        return self.mvn.log_prob(params)

    def forward(self, logits, Y, N=1, params=None):
        nll = self.ce(logits, Y)
        energy_ece = self.ece(logits, Y)
        
        log_prior_value = self.log_prob(params).sum() / N
        log_prior_value = torch.clamp(log_prior_value, min=-1e20, max=1e20)

        ne_en = nll - log_prior_value
        matrices = {'loss': ne_en, 'nll': nll,'ece': energy_ece, 'prior': log_prior_value}
        return matrices


class LaplaceApproxPrior(nn.Module):
    def __init__(self, path, config):
        super().__init__()
        from laplace import Laplace
        model = resnet50().cuda().eval()
        model.relu = nn.ReLU(inplace=False)
        load_weights_simclr(model, path=path, eval_model=False)
        train_loader, test_loader, N, num_of_batches = prepare_data_classification(
            train_dataset=config['prior_train_dataset'],
            val_dataset=config['prior_val_dataset'],
            data_dir=config['prior_data_dir'],
            train_dir=config['prior_train_dir'],
            val_dir=config['prior_val_dir'],
            batch_size=32,
        )
        la = Laplace(model, 'classification',
                     subset_of_weights='last_layer',
                     hessian_structure='diag')
        la.fit(train_loader)
        self.log_prior_i = la
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, Y, N=1, params=None):
        nll = self.ce(logits, Y)
        # TODO: add log_prob function to the class
        log_prior_val = self.log_prob(params).sum() / N
        ne_en = nll - log_prior_val
        matrices = {'loss': ne_en, 'nll': nll, 'prior': log_prior_val}
        return matrices