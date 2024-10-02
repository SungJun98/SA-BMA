'''
References
[1] https://github.com/lblaoke/EMCMC/blob/master/utils.py
[2] https://github.com/lblaoke/EMCMC/blob/master/exp/cifar10_sgld.py
[3] https://github.com/lblaoke/EMCMC/blob/master/exp/cifar10_emcmc.py
'''
import torch
import numpy as np
import random
from math import *
from tqdm import tqdm

import utils.utils as utils
import utils.swag.swag_utils as swag_utils


def lr_decay(args, opt, epoch, batch_idx, num_batch, T, M):
    lr0, lr1 = args.lr_init, args.lr_end

    if args.decay_scheme=='cyclical':
        rcounter = epoch*num_batch + batch_idx
        cos_inner = pi*(rcounter%(T//M))
        cos_inner /= T//M
        cos_out = cos(cos_inner)+1
        lr = lr1+(lr0-lr1)/2*cos_out
    elif args.decay_scheme=='exp':
        lr = lr0*((lr1/lr0)**(epoch/args.epochs))
    elif args.decay_scheme=='linear':
        lr = lr1+(args.epoch-epoch)*(lr0-lr1)/args.epochs
    elif args.decay_scheme=='step':
        if epoch<=args.epochs-40:
            lr = lr0
        elif epoch<=args.epochs-20:
            lr = lr0/5
        else:
            lr = lr0/25
    else:
        lr = lr0

    if opt:
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    return lr



def noise(net,coeff):
    _noise = 0
    for param in net.parameters():
        _noise += torch.sum(param*torch.randn_like(param.data)*coeff)
    return _noise



def train_sgld(args, epoch, dataloader, model, criterion, optimizer, scaler=None):
        loss_sum = 0.0
        correct = 0.0

        num_objects_current = 0
        num_batch = len(dataloader)
        T = args.epochs * num_batch
        
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            try:
                X = batch["img"].to(args.device)
                y = batch["label"].to(args.device)
            except:
                X, y = batch[0].to(args.device), batch[1].to(args.device)

            lr = lr_decay(args, optimizer, epoch-1, batch_idx, num_batch, T, args.n_cycle)
            
            pred = model(X)
            noise_coeff = sqrt(2/lr/50000*args.temp)
            loss = criterion(pred, y) + noise(model, noise_coeff)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            loss_sum += loss.data.item() * X.size(0)
            num_objects_current += X.size(0)
        
        return {
                "loss" : loss_sum / num_objects_current,
                "accuracy" : correct / num_objects_current * 100.0,
                "lr" : lr
                }
    
# resample model1 from model2
def resample(model1, model2, eta=0.0):
    model1.load_state_dict(model2.state_dict())

    if eta > 0:
        for param in model1.parameters():
            param.data += sqrt(eta) * torch.randn_like(param.data)
    elif eta < 0:
        assert False, 'Invalid eta!'

    return model1

# for BN
def additional_forward(args, dataloader, model):
    model.train()
    with torch.no_grad():
        for _,(inputs,_) in enumerate(dataloader):
            model(inputs.to(args.device))


# smooth regularization & random noise
def reg_noise(net1, net2, datasize, alpha, eta, temperature):
    reg_coeff = 0.5/(eta*datasize)
    noise_coeff = sqrt(2/alpha/datasize*temperature)
    loss = 0

    for param1,param2 in zip(net1.parameters(),net2.parameters()):
        sub = param1-param2
        reg = sub*sub*reg_coeff
        noise1 = param1*torch.randn_like(param1.data)*noise_coeff
        noise2 = param2*torch.randn_like(param2.data)*noise_coeff
        loss += torch.sum(reg-noise1-noise2)

    return loss



def train_emcmc(args, epoch, dataloader, model_s, model_a, criterion, optimizer, scaler=None):
    loss_sum = 0.0
    correct = 0.0

    num_objects_current = 0
    num_batch = len(dataloader)
    T = args.epochs * num_batch
    
    model_s.train()
    for batch_idx, batch in enumerate(dataloader):
        try:
            X = batch["img"].to(args.device)
            y = batch["label"].to(args.device)
        except:
            X, y = batch[0].to(args.device), batch[1].to(args.device)

        lr = lr_decay(args, optimizer, epoch-1, batch_idx, num_batch, T, args.n_cycle)
        
        pred = model_s(X)
        loss = criterion(pred, y) + reg_noise(model_s, model_a, len(dataloader.dataset), lr, args.eta, args.temp)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss_sum += loss.data.item() * X.size(0)
        num_objects_current += X.size(0)
    
    return {
            "loss" : loss_sum / num_objects_current,
            "accuracy" : correct / num_objects_current * 100.0,
            "lr" : lr
            }

    
def save_sample(args, model, save_path):
    model.cpu()
    torch.save(model.state_dict(), save_path)
    model.to(args.device)
    return save_path



def bma_mcmc(args, te_loader, num_classes, w_list, model, criterion):
    bma_logits = np.zeros((len(te_loader.dataset), num_classes))
    bma_predictions = np.zeros((len(te_loader.dataset), num_classes))
    with torch.no_grad():
        for idx, w in tqdm(enumerate(w_list)):
        
            ## load w
            model.load_state_dict(torch.load(w))
            model.to(args.device)
            
            ## eval    
            res = swag_utils.predict(te_loader, model)
            logits = res["logits"]; predictions = res["predictions"]; targets = res["targets"]
                
            accuracy = np.mean(np.argmax(predictions, axis=1) == targets)
            nll = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), targets] + args.eps))
            print(f"Sample {idx+1}/{len(w_list)}. Accuracy: {accuracy*100:.2f}% NLL: {nll:.4f}")
                
            bma_logits += logits
            bma_predictions += predictions
                
            ens_accuracy = np.mean(np.argmax(bma_predictions, axis=1) == targets)
            ens_nll = -np.mean(
                np.log(
                    bma_predictions[np.arange(bma_predictions.shape[0]), targets] / (idx + 1)
                    + args.eps
                )
            )
            print(f"Ensemble {idx+1}/{len(w_list)}. Accuracy: {ens_accuracy*100:.2f}% NLL: {ens_nll:.4f}")

        bma_logits /= len(w_list)
        bma_predictions /= len(w_list)

        bma_accuracy = np.mean(np.argmax(bma_predictions, axis=1) == targets)
        bma_nll = -np.mean(
            np.log(bma_predictions[np.arange(bma_predictions.shape[0]), targets] + args.eps)
        )
        bma_unc = utils.calibration_curve(bma_predictions, targets, args.num_bins)
        
    print(f"bma Accuracy using {len(w_list)} model : {bma_accuracy * 100:.2f}% / NLL : {bma_nll:.4f}")
    return {"logits" : bma_logits,
            "predictions" : bma_predictions,
            "targets" : targets,
            "accuracy" : bma_accuracy * 100,
            "nll" : bma_nll,
            "unc" : bma_unc, 
            "ece" : bma_unc['ece'],
    }