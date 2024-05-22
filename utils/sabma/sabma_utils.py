import numpy as np
import pickle, os, collections
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sam import sam, sam_utils
import utils.sabma.sabma as sabma
import utils.utils as utils

def get_optimizer(args, sabma_model):
    '''
    Define optimizer
    '''
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(sabma_model.bnn_param.values(),
                        lr=args.lr_init, weight_decay=args.wd,
                        momentum=args.momentum)
    
    elif args.optim == "sam":
        base_optimizer = torch.optim.SGD
        optimizer = sam.SAM(sabma_model.bnn_param.values(), base_optimizer, rho=args.rho, lr=args.lr_init, momentum=args.momentum,
                        weight_decay=args.wd)
        
    elif args.optim == "sabma":
        base_optimizer = torch.optim.SGD
        optimizer = sabma.SABMA_optim(sabma_model.bnn_param.values(), base_optimizer, rho=args.rho, lr=args.lr_init, momentum=args.momentum,
                        weight_decay=args.wd)
    
    return optimizer

def list_to_state_dict(sabma_model, tr_sample, frz_sample):
    '''
    Change sample list to state dict
    '''
    ordDict = collections.OrderedDict()
    tr_idx = 0; frz_idx = 0
    for name in sabma_model.full_param_shape.keys():
        if name in sabma_model.tr_param_shape.keys():
            ordDict[name] = tr_sample[tr_idx]
            tr_idx += 1
        elif name in sabma_model.frz_param_shape.keys():
            ordDict[name] = frz_sample[frz_idx]
            frz_idx += 1
    assert tr_idx == len(sabma_model.tr_param_shape.keys()), "Check the process to convert trainable parameter sample to state dict"
    assert frz_idx == len(sabma_model.frz_param_shape.keys()), "Check the process to convert freezed parameter sample to state dict"

    return ordDict


def format_weights(tr_sample, frz_sample, sabma_model):
    '''
    Format sampled vector to state dict
    '''  
    tr_sample = utils.unflatten_like_size(tr_sample, sabma_model.tr_param_shape.values())
    frz_sample = utils.unflatten_like_size(frz_sample, sabma_model.frz_param_shape.values())
    state_dict = list_to_state_dict(sabma_model, tr_sample, frz_sample)
    return state_dict




def train_sabma_sabma(dataloader, sabma_model, criterion, optimizer, device, first_step_scaler, second_step_scaler, kl_eta=1.0):
    loss_sum = 0.0
    correct = 0.0
    num_objects_current = 0
    
    sabma_model.train()
    sabma_model.backbone.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Sample weight
        tr_params, z_1, z_2 = sabma_model.sample(z_scale = 1.0, sample_param='tr')    
        frz_params, _, _ = sabma_model.sample(z_scale = 1.0, sample_param='frz')

        # compute log probability and gradient of log probability w.r.t. model parameters
        _, log_grad = sabma_model.log_grad(tr_params)
        
        # Change weight sample shape to input model
        params = format_weights(tr_params, frz_params, sabma_model)
        
        if first_step_scaler is not None:
            ## first forward & backward
            with torch.cuda.amp.autocast():
                pred = sabma_model(params, X)
                loss = criterion(pred, y)

            first_step_scaler.scale(loss).backward()
            first_step_scaler.unscale_(optimizer)
            
            optimizer_state = first_step_scaler._per_optimizer_states[id(optimizer)]
            
            inf_grad_cnt = sum(v.item() for v in optimizer_state["found_inf_per_device"].values())      # Check if any gradients are inf/nan
            if inf_grad_cnt == 0:
                # if valid graident, apply sam_first_step
                optimizer.first_step(log_grad, zero_grad=True)
                sam_first_step_applied = True
            else:
                # if invalid graident, skip sam and revert to single optimization step
                optimizer.zero_grad()
                sam_first_step_applied = False
            first_step_scaler.update()
            
            ## second forward-backward pass
            tr_params = optimizer.second_sample(z_1, z_2, sabma_model)
            params = format_weights(tr_params, frz_params, sabma_model)
            
            prior_log_prob = sabma_model.prior_log_prob()
            posterior_log_prob = sabma_model.posterior_log_prob()
            kld_loss = (posterior_log_prob - prior_log_prob).mean()

            with torch.cuda.amp.autocast():
                pred = sabma_model(params, X)
                loss = criterion(pred, y) + kl_eta * kld_loss
            
            second_step_scaler.scale(loss).backward()
            if sam_first_step_applied:
                optimizer.second_step()  
            second_step_scaler.step(optimizer)
            second_step_scaler.update()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            loss_sum += loss.data.item() * X.size(0)
            num_objects_current += X.size(0)
            
            # print(f"Batch : {batch} / Tr loss : {loss:.4f} / KL loss : {kld_loss:.4f} / Pred NaN : {torch.sum(torch.isnan(pred))/100} ")
        else:
            ## first forward & backward
            pred = sabma_model(params, X)
            loss = criterion(pred, y)        
            loss.backward()
            optimizer.first_step(log_grad, zero_grad=True, amp=False)
                      
            ## second forward-backward pass
            tr_params = optimizer.second_sample(z_1, z_2, sabma_model)
            params = format_weights(tr_params, frz_params, sabma_model)
            
            pred = sabma_model(params, X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.second_step(zero_grad=True, amp=False)  

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            loss_sum += loss.data.item() * X.size(0)
            num_objects_current += X.size(0)
    return{
        "loss" : loss_sum / num_objects_current,
        "accuracy" : correct / num_objects_current * 100.0,
    }
    
    
def eval_sabma(loader, sabma_model, params, criterion, device, num_bins=15, eps=1e-8):
    '''
    get loss, accuracy, nll and ece for every eval step
    '''

    loss_sum = 0.0
    num_objects_total = len(loader.dataset)

    logits = list()
    preds = list()
    targets = list()

    sabma_model.eval()
    sabma_model.backbone.eval()
    offset = 0
    with torch.no_grad():
        for _, (input, target) in enumerate(loader):
            input, target = input.to(device), target.to(device)
            pred = sabma_model(params, input)
            loss = criterion(pred, target)
            loss_sum += loss.item() * input.size(0)
            
            logits.append(pred.cpu().numpy())    
            preds.append(F.softmax(pred, dim=1).cpu().numpy())
            targets.append(target.cpu().numpy())
            offset += input.size(0)
    
    logits = np.vstack((logits))
    preds = np.vstack(preds)
    targets = np.concatenate(targets)

    accuracy = np.mean(np.argmax(preds, axis=1) == targets)
    nll = -np.mean(np.log(preds[np.arange(preds.shape[0]), targets] + eps))
    ece = utils.calibration_curve(preds, targets, num_bins)['ece']
    
    return {
        "logits" : logits,
        "predictions" : preds,
        "targets" : targets,
        "loss" : loss_sum / num_objects_total,
        "accuracy" : accuracy * 100.0,
        "nll" : nll,
        "ece" : ece
    }
    
    
def bma_sabma(te_loader, sabma_model, bma_num_models,
            num_classes, criterion, device,
            bma_save_path=None, eps=1e-8, num_bins=15,
            validation=False, tr_layer="nl_ll",
            ood_loader=None
            ):
    '''
    run bayesian model averaging in test step
    '''
    sabma_logits = np.zeros((len(te_loader.dataset), num_classes))
    sabma_predictions = np.zeros((len(te_loader.dataset), num_classes))
    if ood_loader is not None:
        ood_sabma_predictions =np.zeros((len(ood_loader.dataset), num_classes))
    with torch.no_grad():
        for i in range(bma_num_models):
            tr_params, _, _  = sabma_model.sample(z_scale=1.0, sample_param='tr')
            frz_params, _, _  = sabma_model.sample(z_scale=1.0, sample_param='frz')
            params = format_weights(tr_params, frz_params, sabma_model)
                
            # save sampled weight for bma
            if (bma_save_path is not None) and (not validation):
                torch.save(params, f'{bma_save_path}/bma_model-{i}.pt')
            
            res = eval_sabma(te_loader, sabma_model, params, criterion, device, num_bins, eps)
            if ood_loader is not None:
                ood_res = eval_sabma(ood_loader, sabma_model, params, criterion, device, num_bins, eps)

            if not validation:
                print(f"SA-BMA Sample {i+1}/{bma_num_models}. Accuracy: {res['accuracy']:.2f}%  NLL: {res['nll']:.4f}")

            sabma_logits += res["logits"]
            sabma_predictions += res["predictions"]
            
            if ood_loader is not None:
                ood_sabma_predictions += ood_res["predictions"]

            ens_accuracy = np.mean(np.argmax(sabma_predictions, axis=1) == res["targets"]) * 100
            ens_nll = -np.mean(np.log(sabma_predictions[np.arange(sabma_predictions.shape[0]), res["targets"]] / (i + 1) + eps))
            
            if not validation:
                print(f"Ensemble {i+1}/{bma_num_models}. Accuracy: {ens_accuracy:.2f}% NLL: {ens_nll:.4f}")

        sabma_logits /= bma_num_models
        sabma_predictions /= bma_num_models

        sabma_loss = criterion(torch.tensor(sabma_predictions), torch.tensor(res['targets'])).item()
        sabma_accuracy = np.mean(np.argmax(sabma_predictions, axis=1) == res["targets"]) * 100
        sabma_nll = -np.mean(np.log(sabma_predictions[np.arange(sabma_predictions.shape[0]), res["targets"]] + eps))        
        unc = utils.calibration_curve(sabma_predictions, res["targets"], num_bins)
        sabma_ece = unc['ece']
        
        
    if not validation:
        print(f"bma Accuracy using {bma_num_models} model : {sabma_accuracy:.2f}% / NLL : {sabma_nll:.4f}")
        
    if ood_loader is not None:    
        return {"logits" : sabma_logits,
            "predictions" : sabma_predictions,
            "targets" : res["targets"],
            "loss" : sabma_loss,
            "accuracy" : sabma_accuracy,
            "nll" : sabma_nll,
            "ece" : sabma_ece,
            }
    else:
        return {"logits" : sabma_logits,
            "predictions" : sabma_predictions,
            "targets" : res["targets"],
            "loss" : sabma_loss,
            "accuracy" : sabma_accuracy,
            "nll" : sabma_nll,
            "ece" : sabma_ece,
            }   
    
    
    
def save_best_sabma_model(args, best_epoch, sabma_model, optimizer, scaler, first_step_scaler, second_step_scaler):
    if args.optim == "sgd":
        if not args.no_amp:
            utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                            epoch = best_epoch,
                            state_dict =sabma_model.state_dict(),
                            optimizer = optimizer.state_dict(),
                            # scheduler = scheduler.state_dict(),
                            scaler = scaler.state_dict(),
                            )
        else:
            utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                            epoch = best_epoch,
                            state_dict =sabma_model.state_dict(),
                            optimizer = optimizer.state_dict(),
                            # scheduler = scheduler.state_dict(),
                            )
    elif args.optim in ["sam", "sabma"]:
        if not args.no_amp:
            utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                epoch = best_epoch,
                                state_dict = sabma_model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                # scheduler = scheduler.state_dict(),
                                first_step_scaler = first_step_scaler.state_dict(),
                                second_step_scaler = second_step_scaler.state_dict()
                                )
        else:
            utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                epoch = best_epoch,
                                state_dict = sabma_model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                # scheduler = scheduler.state_dict(),
                                )
    torch.save(sabma_model, f"{args.save_path}/{args.method}-{args.optim}_best_val_model.pt")
    
    # Save Mean, variance, Covariance matrix
    mean = sabma_model.get_mean_vector()
    torch.save(mean,f'{args.save_path}/{args.method}-{args.optim}_best_val_mean.pt')
    
    variance = sabma_model.get_variance_vector()
    torch.save(variance, f'{args.save_path}/{args.method}-{args.optim}_best_val_variance.pt')
    
    cov_mat_list = sabma_model.get_covariance_matrix()
    torch.save(cov_mat_list, f'{args.save_path}/{args.method}-{args.optim}_best_val_covmat.pt')    