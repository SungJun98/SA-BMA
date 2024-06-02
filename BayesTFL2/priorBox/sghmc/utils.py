import csv
import os
from pathlib import Path
import torch
from ray import tune
from ..prior.prior_ssl_utils.utils import get_backbone
from typing import Dict
from torch.utils.data import DataLoader
import numpy as np
from torchmetrics.classification import MulticlassCalibrationError


import torch
from torch import nn, optim
from torch.nn import functional as F



def run_and_log_bma(net, test_loaders, samples_dir, config, loggers, num_classes):
    """Run Bayesian model averaging  and log it to file and loggers"""
    dict_res = {}
    test_loader_name = "real"
    bma_test_metrics = test_bma(net, test_loaders, samples_dir, num_classes)
    print(f"BMA Test {test_loader_name}: test_acc: {bma_test_metrics['test_acc']:.4f}")
    print(f"BMA Test {test_loader_name}: test_nll: {bma_test_metrics['test_nll']:.4f}")
    print(f"BMA Test {test_loader_name}: test_ece: {bma_test_metrics['test_ece']:.4f}")
    
    dict_res[f'BMA acc {test_loader_name}'] = bma_test_metrics['test_acc']

    for logger in loggers:
        logger.log_metrics(dict_res)
    if config['use_tune']:
        write_files(dict_res)


def load_prior(path: str, config: dict, number_of_samples_prior: int = 4) -> Dict[str, torch.Tensor]:
    """Return the networks prior"""
    prior_type = config['prior_type']
    if prior_type == 'shifted_gaussian':
        # TODO fix to load from ssl_prior
        # load mean, diagonal variance and the low rank covariance

        mean = torch.load(path + '_mean.pt')
        variance = torch.load(path + '_variance.pt')
        cov_factor = torch.load(path + '_covmat.pt')

        
        variance = config['prior_scale'] * variance + config['prior_eps']
        if number_of_samples_prior > 0:
            if config['scale_low_rank']:
                cov_mat_sqrt = config['prior_scale'] * (cov_factor[:number_of_samples_prior])
            else:
                cov_mat_sqrt = cov_factor[:number_of_samples_prior]
        else:
            cov_mat_sqrt = torch.zeros_like(cov_factor[:1])
    elif prior_type == 'normal':
        # Normal gaussian prior
        backbone = get_backbone(config)
        mean = torch.flatten(torch.cat([torch.flatten(p) for p in backbone.parameters()]))
        mean = torch.zeros_like(mean)
        variance = config['prior_scale'] * torch.ones_like(mean)
        cov_mat_sqrt = torch.zeros_like(mean[None, :])
    elif prior_type == 'shifted_gaussian_diag':
        model = torch.load(path)
        if 'resnet' in model:
            model = model['resnet']
            del model['fc.weight']
            del model['fc.bias']
        mean = torch.flatten(torch.cat([torch.flatten(model[p]) for p in model]))
        variance = torch.sqrt(torch.abs(mean) +config['prior_eps'])
        variance = config['prior_scale'] * variance
        cov_mat_sqrt = torch.zeros((2, variance.shape[0]))

    else:
        raise NotImplementedError(f"The prior {prior_type} is not supported")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    prior_params = {'mean': mean.to(device), 'variance': variance.to(device), 'cov_mat_sqr': cov_mat_sqrt.to(device)}
    # prior_params = {'mean': mean.cuda(), 'variance': variance.cuda(), 'cov_mat_sqr': cov_mat_sqrt.cuda()}
    # prior_params = {'mean': mean, 'variance': variance, 'cov_mat_sqr': cov_mat_sqrt}


    return prior_params


@torch.no_grad()
def test_bma(net, data_loader: DataLoader, samples_dir: str, num_classes:int, device: str = None) -> Dict[str, float]:
    """Calculate the Bayesian model averaging based on all the samples """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = net.to(device)
    net.eval()
    ens_logits = []
    ce = nn.CrossEntropyLoss()
    ece = MulticlassCalibrationError(num_classes=num_classes, n_bins=50, norm='l1')
    
    
    es = 0
    lat_box = []
    results = []  # 추가된 결과 리스트

    for sample_path in Path(samples_dir).rglob('*.pt'):
        acc_ = 0.0  # 초기화

        # sample_path, acc 정보 저장
        result = {"sample_path": str(sample_path), "acc": None, "original_position": None}

        print(f"[{es}] {sample_path}")
        try:
            net.load_state_dict(torch.load(sample_path))

            all_logits = []
            all_Y = []
            for batch in data_loader:

                if len(batch) > 2:
                    X, Y = batch['img'] , batch['label']
                else:
                    #print("tt")
                    X, Y = batch #batch['img'] , batch['label'] #batch

                X, Y = X.to(device), Y.to(device)
                all_logits.append(net(X))
                all_Y.append(Y)

            all_logits = torch.cat(all_logits)
            all_Y = torch.cat(all_Y)
            ens_logits.append(all_logits)

            Y_pred_ = all_logits.argmax(dim=-1)
            acc_ = (Y_pred_ == all_Y).sum().item() / Y_pred_.size(0)
            print(f"acc: {acc_:.4f}")

        except Exception as e:
            print(f"Error loading {sample_path}: {e}")

        finally:
            # 결과 저장
            result["acc"] = acc_
            result["original_position"] = es
            results.append(result)
            lat_box.append(acc_)
            es += 1

    # 정확도 순서대로 정렬 (acc 기준 내림차순)
    sorted_results = sorted(results, key=lambda x: x["acc"], reverse=True)

    ens_logits = torch.stack(ens_logits).mean(dim=0)
    print("\n정렬된 결과:")
    for result in sorted_results:
        print(f"sample_path: {result['sample_path']}, acc: {result['acc']:.4f}, 원래 위치: {result['original_position']}")

    print("\nacc 리스트:")
    print(lat_box)
    print(f"\n정확도 순위 (acc 기준 내림차순):")
    print(torch.argsort(torch.tensor(lat_box)))
    
    

    
    energy = ce(ens_logits, all_Y)
    energy_ece = ece(ens_logits, all_Y)

    
    Y_pred = ens_logits.argmax(dim=-1)
    acc = (Y_pred == all_Y).sum().item() / Y_pred.size(0)
    return {'test_acc': acc, 'test_nll':energy, 'test_ece': energy_ece} #acc 였음 원래는


def write_files(dict_res: dict):
    """Write the results of tune to files during learning"""
    with tune.checkpoint_dir(step=0) as checkpoint_dir:
        dir_to_write = checkpoint_dir.split('checkpoint')[0]
        progress_file = os.path.join(dir_to_write, 'text.csv')
        with open(progress_file, 'w', newline='') as file:
            writer = csv.writer(file)
            header = ','.join(dict_res.keys())
            writer.writerow(header)
            writer.writerow(dict_res.values())

