# https://github.com/wjmaddox/swa_gaussian/blob/ed5fd56e34083b42630239e59076952dee44daf4/swag/utils.py
import itertools
import torch
import math
import numpy as np
import tqdm
import torch.nn.functional as F

import utils.utils as utils
from utils import temperature_scaling as ts

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    
    return outList


def LogSumExp(x, dim=0):
    m, _ = torch.max(x, dim=dim, keepdim=True)
    return m + torch.log((x - m).exp().sum(dim=dim, keepdim=True))


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_best_swag_model(args, best_epoch, model, swag_model, optimizer, scaler, first_step_scaler, second_step_scaler):
    if args.optim in ["sgd", "adam"]:
        if not args.no_amp:
            utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                epoch = best_epoch,
                                state_dict = swag_model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                # scheduler = scheduler.state_dict(),
                                scaler = scaler.state_dict()
                                )
        else:
            utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                epoch = best_epoch,
                                state_dict = swag_model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                # scheduler = scheduler.state_dict(),
                                )
    elif args.optim in ["sam", "bsam"]:
        if not args.no_amp:
            utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                epoch = best_epoch,
                                state_dict = swag_model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                # scheduler = scheduler.state_dict(),
                                first_step_scaler = first_step_scaler.state_dict(),
                                second_step_scaler = second_step_scaler.state_dict()
                                )
        else:
            utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                epoch = best_epoch,
                                state_dict = swag_model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                # scheduler = scheduler.state_dict(),
                                )
            
    torch.save(model.state_dict(),f'{args.save_path}/{args.method}-{args.optim}_best_val_model.pt')
    
    # Save Mean, variance, Covariance matrix
    mean = swag_model.get_mean_vector()
    torch.save(mean,f'{args.save_path}/{args.method}-{args.optim}_best_val_mean.pt')
    
    variance = swag_model.get_variance_vector()
    torch.save(variance, f'{args.save_path}/{args.method}-{args.optim}_best_val_variance.pt')
    
    cov_mat_list = swag_model.get_covariance_matrix()
    torch.save(cov_mat_list, f'{args.save_path}/{args.method}-{args.optim}_best_val_covmat.pt')    



def predict(loader, model, temperature=None):
    logits = list()
    preds = list()
    targets = list()

    model.eval()
    
    offset = 0
    with torch.no_grad():
        for input, target in loader:
            input = input.to("cuda")
            output = model(input)

            batch_size = input.size(0)
            if temperature is not None:
                temperature.unsqueeze(1).expand(output.size(0), output.size(1))
                output = output / temperature           
            logits.append(output.cpu().numpy())
            preds.append(F.softmax(output, dim=1).cpu().numpy())
            targets.append(target.cpu().numpy())
            offset += batch_size

    return {"logits" : np.vstack(logits), "predictions": np.vstack(preds), "targets": np.concatenate(targets)}
    


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:
            loader = tqdm.tqdm(loader, total=num_batches)
        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))

def apply_bn_update(model, momenta):
    model.apply(lambda module: _set_momenta(module, momenta))

    
def inv_softmax(x, eps=1e-10):
    return torch.log(x / (1.0 - x + eps))


def predictions(test_loader, model, device, seed=None, **kwargs):
    # will assume that model is already in eval mode
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        if seed is not None:
            torch.manual_seed(seed)

        input = input.to(device, non_blocking=True)
        output = model(input, **kwargs)
        
        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def schedule(epoch, lr_init, epochs, swa, swa_start=None, swa_lr=None):
    try:
      t = (epoch) / (swa_start if swa else epochs)
    except:
      t = (epoch) / (swa_start+1 if swa else epochs)
    lr_ratio = swa_lr / lr_init if swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_init * factor



def bma_swag(tr_loader, te_loader,  model, num_classes, criterion, bma_num_models=30, bma_save_path=None, eps=1e-8, batch_norm=True, seed=None, num_bins=15):
    '''
    run bayesian model averaging in test step
    '''
    if seed is not None:
        utils.set_seed(seed)
    bma_logits = np.zeros((len(te_loader.dataset), num_classes))
    bma_predictions = np.zeros((len(te_loader.dataset), num_classes))
    with torch.no_grad():
        for i in range(bma_num_models):          
            model.sample(1.0, cov=True, seed=seed)
            if batch_norm:
                bn_update(tr_loader, model, verbose=False, subset=1.0)
                            
            # save sampled weight for bma
            if bma_save_path is not None:
                torch.save(model, f'{bma_save_path}/bma_model-{i}.pt')

            res = predict(te_loader, model)
            logits = res["logits"]; predictions = res["predictions"]; targets = res["targets"]
            
            accuracy = np.mean(np.argmax(predictions, axis=1) == targets)
            nll = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), targets] + eps))
            print(f"Sample {i+1}/{bma_num_models}. Accuracy: {accuracy*100:.2f}% NLL: {nll:.4f}")
            
            bma_logits += logits
            bma_predictions += predictions
            
            ens_accuracy = np.mean(np.argmax(bma_predictions, axis=1) == targets)
            ens_nll = -np.mean(
                np.log(
                    bma_predictions[np.arange(bma_predictions.shape[0]), targets] / (i + 1)
                    + eps
                )
            )
            print(f"Ensemble {i+1}/{bma_num_models}. Accuracy: {ens_accuracy*100:.2f}% NLL: {ens_nll:.4f}")

        bma_logits /= bma_num_models
        bma_predictions /= bma_num_models

        bma_loss = criterion(torch.tensor(bma_predictions), torch.tensor(targets)).item()
        bma_accuracy = np.mean(np.argmax(bma_predictions, axis=1) == targets)
        bma_nll = -np.mean(
            np.log(bma_predictions[np.arange(bma_predictions.shape[0]), targets] + eps)
        )
        bma_unc = utils.calibration_curve(bma_predictions, targets, num_bins)
        
    print(f"bma Accuracy using {bma_num_models} model : {bma_accuracy * 100:.2f}% / NLL : {bma_nll:.4f}")
    return {"logits" : bma_logits,
            "predictions" : bma_predictions,
            "targets" : targets,
            "accuracy" : bma_accuracy * 100,
            "nll" : bma_nll,
            "unc" : bma_unc, 
            "ece" : bma_unc['ece'],
    }