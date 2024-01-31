import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sam import sam, sam_utils
import utils.sabtl.sabtl as sabtl
import utils.utils as utils

def get_optimizer(args, sabtl_model):
    '''
    Define optimizer
    '''
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(sabtl_model.bnn_param.values(),
                        lr=args.lr_init, weight_decay=args.wd,
                        momentum=args.momentum)
    
    elif args.optim == "sam":
        base_optimizer = torch.optim.SGD
        optimizer = sam.SAM(sabtl_model.bnn_param.values(), base_optimizer, rho=args.rho, lr=args.lr_init, momentum=args.momentum,
                        weight_decay=args.wd)
        
    elif args.optim == "bsam":
        base_optimizer = torch.optim.SGD
        optimizer = sabtl.BSAM(sabtl_model.bnn_param.values(), base_optimizer, rho=args.rho, lr=args.lr_init, momentum=args.momentum,
                        weight_decay=args.wd)
    
    return optimizer



def train_sabtl_sgd(dataloader, sabtl_model, criterion, optimizer, device, scaler):
    loss_sum = 0.0
    correct = 0.0
    num_objects_current = 0
    
    sabtl_model.backbone.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
           
        # Sample weight
        params, _, _ = sabtl_model.sample(1.0)
        
        # Change weight sample shape to input model
        params = utils.format_weights(params, sabtl_model)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = sabtl_model(params, X)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            
            # # gradient clipping (useless....)
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(sabtl_model.bnn_param.log_std, 0.05) # max_norm=0.1

            scaler.step(optimizer)  # optimizer.step()
            scaler.update()
            optimizer.zero_grad()
        else:
            pred = sabtl_model(params, X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss_sum += loss.data.item() * X.size(0)
        num_objects_current += X.size(0)
        
    return{
        "loss" : loss_sum / num_objects_current,
        "accuracy" : correct / num_objects_current * 100.0,
    }




def train_sabtl_sam(dataloader, sabtl_model, criterion, optimizer, device, first_step_scaler, second_step_scaler):
    loss_sum = 0.0
    correct = 0.0
    num_objects_current = 0
    
    sabtl_model.backbone.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
           
        # Sample weight
        params, z_1, z_2 = sabtl_model.sample(1.0)        
        # Change weight sample shape to input model
        params = utils.format_weights(params, sabtl_model)

        if first_step_scaler is not None:
            ## first forward & backward
            with torch.cuda.amp.autocast():
                pred = sabtl_model(params, X)
                loss = criterion(pred, y)        
            first_step_scaler.scale(loss).backward()
            first_step_scaler.unscale_(optimizer)
            
            optimizer_state = first_step_scaler._per_optimizer_states[id(optimizer)]
            
            inf_grad_cnt = sum(v.item() for v in optimizer_state["found_inf_per_device"].values())      # Check if any gradients are inf/nan
            if inf_grad_cnt == 0:
                # if valid graident, apply sam_first_step
                optimizer.first_step(zero_grad=True)
                sam_first_step_applied = True
            else:
                # if invalid graident, skip sam and revert to single optimization step
                optimizer.zero_grad()
                sam_first_step_applied = False  
            first_step_scaler.update()

            ## second forward-backward pass
            params = optimizer.second_sample(z_1, z_2, sabtl_model)
            with torch.cuda.amp.autocast():
                pred = sabtl_model(params, X)
                loss = criterion(pred, y)
            second_step_scaler.scale(loss).backward()
            
            if sam_first_step_applied:
                optimizer.second_step()  
            second_step_scaler.step(optimizer)
            second_step_scaler.update()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            loss_sum += loss.data.item() * X.size(0)
            num_objects_current += X.size(0)
            
        else:
            ## first forward & backward
            pred = sabtl_model(params, X)
            loss = criterion(pred, y)        
            loss.backward()
            optimizer.first_step(zero_grad=True, amp=False)
                      
            ## second forward-backward pass
            params = optimizer.second_sample(z_1, z_2, sabtl_model)
            params = utils.format_weights(params, sabtl_model)
            
            pred = sabtl_model(params, X)
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



def train_sabtl_bsam(dataloader, sabtl_model, criterion, optimizer, device, eta, first_step_scaler, second_step_scaler, tr_layer):
    loss_sum = 0.0
    correct = 0.0
    num_objects_current = 0
    
    sabtl_model.backbone.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        params, z_1, z_2 = sabtl_model.sample(1.0)    # Sample weight

        log_grad = sabtl_model.log_grad(params, approx='full', eta=eta)             # compute gradient of log probability w.r.t. model parameters
        params = utils.format_weights(params, sabtl_model, tr_layer)       # Change weight sample shape to input model

        if first_step_scaler is not None:
            ## first forward & backward
            with torch.cuda.amp.autocast():
                pred = sabtl_model(params, X)
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
            params = optimizer.second_sample(z_1, z_2, sabtl_model)
            with torch.cuda.amp.autocast():
                pred = sabtl_model(params, X)
                loss = criterion(pred, y)

            second_step_scaler.scale(loss).backward()
            if sam_first_step_applied:
                optimizer.second_step()  
            second_step_scaler.step(optimizer)
            second_step_scaler.update()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            loss_sum += loss.data.item() * X.size(0)
            num_objects_current += X.size(0)
            
            print(f"Batch : {batch} / Tr loss : {loss} / Pred NaN : {torch.sum(torch.isnan(pred))/100} ")
        else:
            ## first forward & backward
            pred = sabtl_model(params, X)
            loss = criterion(pred, y)        
            loss.backward()
            optimizer.first_step(log_grad, zero_grad=True, amp=False)
                      
            ## second forward-backward pass
            params = optimizer.second_sample(z_1, z_2, sabtl_model)
            pred = sabtl_model(params, X)
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
    
    
def eval_sabtl(loader, sabtl_model, params, criterion, device, num_bins=15, eps=1e-8):
    '''
    get loss, accuracy, nll and ece for every eval step
    '''

    loss_sum = 0.0
    num_objects_total = len(loader.dataset)

    preds = list()
    targets = list()

    sabtl_model.backbone.eval()
    offset = 0
    with torch.no_grad():
        for _, (input, target) in enumerate(loader):
            input, target = input.to(device), target.to(device)
            pred = sabtl_model(params, input)
            loss = criterion(pred, target)
            loss_sum += loss.item() * input.size(0)
            
            preds.append(F.softmax(pred, dim=1).cpu().numpy())
            targets.append(target.cpu().numpy())
            offset += input.size(0)
    
    preds = np.vstack(preds)
    targets = np.concatenate(targets)

    accuracy = np.mean(np.argmax(preds, axis=1) == targets)
    nll = -np.mean(np.log(preds[np.arange(preds.shape[0]), targets] + eps))
    ece = utils.calibration_curve(preds, targets, num_bins)['ece']
    
    return {
        "predictions" : preds,
        "targets" : targets,
        "loss" : loss_sum / num_objects_total,
        "accuracy" : accuracy * 100.0,
        "nll" : nll,
        "ece" : ece
    }
    
    
def bma_sabtl(te_loader, sabtl_model, bma_num_models,
            num_classes, criterion, device,
            bma_save_path=None, eps=1e-8, num_bins=15,
            validation=False, tr_layer="last_layer"
            ):
    '''
    run bayesian model averaging in test step
    '''
    sabtl_predictions = np.zeros((len(te_loader.dataset), num_classes))
    with torch.no_grad():
        for i in range(bma_num_models):
            
            if i == 0:
                params, _, _ = sabtl_model.sample(z_scale=0, tr_param_only=False)
            else:
                params, _, _  = sabtl_model.sample(z_scale=1.0, tr_param_only=False)
            
            # save sampled weight for bma
            if (bma_save_path is not None) and (not validation):
                torch.save(params, f'{bma_save_path}/bma_model-{i}.pt')
            
            # params = utils.format_weights(params, sabtl_model)
            params = utils.format_weights(params, sabtl_model, tr_layer)
            res = eval_sabtl(te_loader, sabtl_model, params, criterion, device, num_bins, eps)

            if not validation:
                print(f"SABTL Sample {i+1}/{bma_num_models}. Accuracy: {res['accuracy']:.2f}%  NLL: {res['nll']:.4f}")

            sabtl_predictions += res["predictions"]

            ens_accuracy = np.mean(np.argmax(sabtl_predictions, axis=1) == res["targets"]) * 100
            ens_nll = -np.mean(np.log(sabtl_predictions[np.arange(sabtl_predictions.shape[0]), res["targets"]] / (i + 1) + eps))
            
            if not validation:
                print(f"Ensemble {i+1}/{bma_num_models}. Accuracy: {ens_accuracy:.2f}% NLL: {ens_nll:.4f}")

        sabtl_predictions /= bma_num_models

        sabtl_loss = criterion(torch.tensor(sabtl_predictions), torch.tensor(res['targets'])).item()
        sabtl_accuracy = np.mean(np.argmax(sabtl_predictions, axis=1) == res["targets"]) * 100
        sabtl_nll = -np.mean(np.log(sabtl_predictions[np.arange(sabtl_predictions.shape[0]), res["targets"]] + eps))
        
        unc = utils.calibration_curve(sabtl_predictions, res["targets"], num_bins)
        sabtl_ece = unc['ece']
        
    if not validation:
        print(f"bma Accuracy using {bma_num_models} model : {sabtl_accuracy:.2f}% / NLL : {sabtl_nll:.4f}")
    
    return {"predictions" : sabtl_predictions,
            "targets" : res["targets"],
            "loss" : sabtl_loss,
            "accuracy" : sabtl_accuracy,
            "nll" : sabtl_nll,
            "ece" : sabtl_ece
            }
    
    
    
def save_best_sabtl_model(args, best_epoch, sabtl_model, optimizer, scaler, first_step_scaler, second_step_scaler):
    if args.optim == "sgd":
        if not args.no_amp:
            utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                            epoch = best_epoch,
                            state_dict =sabtl_model.state_dict(),
                            optimizer = optimizer.state_dict(),
                            # scheduler = scheduler.state_dict(),
                            scaler = scaler.state_dict(),
                            )
        else:
            utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                            epoch = best_epoch,
                            state_dict =sabtl_model.state_dict(),
                            optimizer = optimizer.state_dict(),
                            # scheduler = scheduler.state_dict(),
                            )
    elif args.optim in ["sam", "bsam"]:
        if not args.no_amp:
            utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                epoch = best_epoch,
                                state_dict = sabtl_model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                # scheduler = scheduler.state_dict(),
                                first_step_scaler = first_step_scaler.state_dict(),
                                second_step_scaler = second_step_scaler.state_dict()
                                )
        else:
            utils.save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                epoch = best_epoch,
                                state_dict = sabtl_model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                # scheduler = scheduler.state_dict(),
                                )
    # Save Mean, variance, Covariance matrix
    mean = sabtl_model.get_mean_vector()
    torch.save(mean,f'{args.save_path}/{args.method}-{args.optim}_best_val_mean.pt')
    
    variance = sabtl_model.get_variance_vector()
    torch.save(variance, f'{args.save_path}/{args.method}-{args.optim}_best_val_variance.pt')
    
    cov_mat_list = sabtl_model.get_covariance_matrix()
    torch.save(cov_mat_list, f'{args.save_path}/{args.method}-{args.optim}_best_val_covmat.pt')    