"""
TODO
utils를 세분화해서 나눠야 할 것 같음
1. sabtl 관련 utils
2. swag 관련 utils (swag_utils 통폐합)
3. 나머지 utils
+ la, vi도 들어오면 얘네 관련 utils로 쪼개는게 관리하기 더 편할 듯
"""

import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.swag.swag_utils import flatten, bn_update, predict
from baselines.swag import swag

from baselines.sam import sam_utils

import sabtl

## ------------------------------------------------------------------------------------
## Setting Configs --------------------------------------------------------------------
def set_seed(RANDOM_SEED=0):
    '''
    Set seed for reproduction
    '''
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random
    random.seed(RANDOM_SEED)



def set_save_path(args):
    '''
    Set save path following the method / model / dataset / optimizer / hyperparameters
    '''
    ### scheduler part
    if args.scheduler == "cos_anneal":
        save_path_ = f"{args.save_path}/{args.dataset}/{args.model}/{args.method}-{args.optim}/{args.scheduler}_{args.t_max}"
    elif args.scheduler == "swag_lr":
        save_path_ = f"{args.save_path}/{args.dataset}/{args.model}/{args.method}-{args.optim}/{args.scheduler}_{args.swa_lr}"
    elif args.scheduler == "cos_decay":
        # save_path_ = f"{args.save_path}/{args.dataset}/{args.model}/{args.method}-{args.optim}/{args.scheduler}({args.first_cycle_steps}/{args.cycle_mult}/{args.min_lr}/{args.warmup_steps}/{args.decay_ratio})"
        save_path_ = f"{args.save_path}/{args.dataset}/{args.model}/{args.method}-{args.optim}/{args.scheduler}_{args.t_initial}_{args.lr_min}/{args.cycle_mul}/{args.cycle_decay}/{args.warmup_t}"
    
    ## learning hyperparameter part
    if args.method in ["swag", "last_swag"]:
        save_path_ = f"{save_path_}/{args.lr_init}_{args.wd}_{args.max_num_models}_{args.swa_start}_{args.swa_c_epochs}"
    else:
        save_path_ = f"{save_path_}/{args.lr_init}_{args.wd}_{args.momentum}"
        
    if args.optim != "sgd":
        save_path_ = f"{save_path_}_{args.rho}"
    
    return save_path_
    


def set_wandb_runname(args):
    '''
    Set wandb run name following the method / model / dataset / optimizer / hyperparameters
    '''
    ### scheduler part
    if args.scheduler == "cos_anneal":
        run_name_ = f"{args.method}-{args.optim}_{args.model}_{args.dataset}_{args.scheduler}({args.t_max})"
    elif args.scheduler == "swag_lr":
        run_name_ = f"{args.method}-{args.optim}_{args.model}_{args.dataset}_{args.scheduler}({args.swa_lr})"
    elif args.scheduler == "cos_decay":
        # run_name_ = f"{args.method}-{args.optim}_{args.model}_{args.dataset}_{args.scheduler}({args.first_cycle_steps}/{args.cycle_mult}/{args.min_lr}/{args.warmup_steps}/{args.decay_ratio})"
        run_name_ = f"{args.method}-{args.optim}_{args.model}_{args.dataset}_{args.scheduler}({args.t_initial}/{args.lr_min}/{args.cycle_mul}/{args.cycle_decay}/{args.warmup_t})"
    
    ## learning hyperparameter part
    if args.method in ["swag", "last_swag"]:
        run_name_ = f"{run_name_}_{args.lr_init}_{args.wd}_{args.max_num_models}_{args.swa_start}_{args.swa_c_epochs}"
    else:
        run_name_ = f"{run_name_}/{args.lr_init}_{args.wd}_{args.momentum}"
        
    if args.optim != "sgd":
        run_name_ = f"{run_name_}_{args.rho}"
    
    return run_name_




def get_dataset(dataset, data_path, batch_size, num_workers, use_validation):
    '''
    Load Dataset
    '''
    import data
    if dataset == 'cifar10':
        tr_loader, val_loader, te_loader, num_classes = data.get_cifar10(data_path, batch_size,
                                                                        num_workers,
                                                                        use_validation = use_validation)
    elif dataset == 'cifar100':
        tr_loader, val_loader, te_loader, num_classes = data.get_cifar100(data_path, batch_size,
                                                                        num_workers,
                                                                        use_validation = use_validation)
    elif dataset == 'oxfordflower':
        raise RuntimeError("You need to add code for this dataset")
    
    elif dataset == 'oxfordcars':
        raise RuntimeError("You need to add code for this dataset")


    if not use_validation:
        val_loader = te_loader
    
    return tr_loader, val_loader, te_loader, num_classes



def get_backbone(model_name, num_classes, device, pre_trained=False):
    '''
    Define Backbone Model
    '''
    from models import mlp, resnet_noBN, wide_resnet, wide_resnet_noBN
    from torchvision.models import resnet18, resnet50
    if model_name == "mlp":
        model = mlp.MLP(output_size=num_classes)

    elif model_name == "resnet18":
        if pre_trained:
            model = resnet18(pretrained=True)
            freeze_fe(model)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        else:
            model = resnet18(pretrained=False, num_classes=num_classes)
    elif model_name == "resnet18-noBN":
        model = resnet_noBN.resnet18(num_classes=num_classes)

    elif model_name == "resnet50":
        if pre_trained:
            model = resnet50(pretrained=True)
            freeze_fe(model)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        else:
            model = resnet50(pretrained=False, num_classes=num_classes)
    elif model_name == "resnet50-noBN":
        model = resnet_noBN.resnet50(num_classes=num_classes)


        """
        add code for pre-trained model
        """
    elif model_name == "wideresnet28x10":
        model_cfg = getattr(wide_resnet, "WideResNet28x10")
        model = model_cfg.base(num_classes=num_classes)
    elif model_name == "wideresnet28x10-noBN":
        model_cfg = getattr(wide_resnet_noBN, "WideResNet28x10")
        model = model_cfg.base(num_classes=num_classes)

    elif model_name == "wideresnet40x10":
        model_cfg = getattr(wide_resnet, "WideResNet40x10")
        model = model_cfg.base(num_classes=num_classes)
    elif model_name == "wideresnet40x10-noBN":
        model_cfg = getattr(wide_resnet_noBN, "WideResNet40x10")
        model = model_cfg.base(num_classes=num_classes)
    
    model.to(device)
    
    print(f"Preparing model {model_name}")
    
    return model

## ------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------

def freeze_fe(model):
    '''
    Freezing Feature Extractor
    '''
    for name, param in model.named_parameters():
        if name.split('.')[0] in ['fc', 'linear', 'classifier']:
            continue
        param.requires_grad = False
    
    print("Freeze Feature Extractor for last-layer Training")



def save_checkpoint(file_path, epoch, **kwargs):
    '''
    Save Model Checkpoint
    '''
    state = {"epoch": epoch}
    state.update(kwargs)
    torch.save(state, file_path)

                    

# parameter list to state_dict(ordered Dict)
def list_to_state_dict(model, sample_list, last=False):
    '''
    Change sample list to state dict
    '''
    ordDict = collections.OrderedDict()
    if last:
        ordDict["fc.weight"] = sample_list[0]
        ordDict["fc.bias"] = sample_list[1]   
    else:        
        for sample, (name, param) in zip(sample_list, model.named_parameters()):
                ordDict[name] = sample
    return ordDict


def unflatten_like_size(vector, likeTensorSize):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    # Input
    #  - vector : flattened parameters
    #  - likeTensorSize : list of torch.Size
    outList = []
    i = 0
    for layer_size in likeTensorSize:
        n = layer_size.numel()
        outList.append(vector[i : i + n].view(layer_size))
        i += n

    return outList


def format_weights(sample, sabtl_model):
    '''
    Format sampled vector to state dict
    '''  
    sample = unflatten_like_size(sample, sabtl_model.backbone_shape)
    
    if True: # last_layer만
        state_dict = sabtl_model.backbone.state_dict()
        state_dict[f"{sabtl_model.last_layer_name}.weight"] = sample[0]
        state_dict[f"{sabtl_model.last_layer_name}.bias"] = sample[1]    
    else:
        state_dict = dict()
        for (name, _), w in zip(sabtl_model.backbone.named_parameters(), sample):
                state_dict[name] = w
    return state_dict

    
# NLL
# https://github.com/wjmaddox/swa_gaussian/blob/master/experiments/uncertainty/uncertainty.py#L78
def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.sum(np.log(ps))
    return nll



# train SGD
def train_sgd(dataloader, model, criterion, optimizer, device, scaler):
    loss_sum = 0.0
    correct = 0.0

    num_objects_current = 0
    num_batches = len(dataloader)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        with torch.cuda.amp.autocast():
            pred = model(X)
            loss = criterion(pred, y)
        
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
        # Backprop
        scaler.scale(loss).backward()
        scaler.step(optimizer)  # optimizer.step()
        scaler.update()
        optimizer.zero_grad()
        
        loss_sum += loss.data.item() * X.size(0)
        num_objects_current += X.size(0)
    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": correct / num_objects_current * 100.0,
    }



# train SAM
def train_sam(dataloader, model, criterion, optimizer, device, first_step_scaler, second_step_scaler):
    # https://github.com/davda54/sam/issues/7
    loss_sum = 0.0
    correct = 0.0

    num_objects_current = 0
    num_batches = len(dataloader)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        sam_utils.enable_running_stats(model)
        ### first forward-backward pass
        with torch.cuda.amp.autocast():
            pred = model(X)
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

        sam_utils.disable_running_stats(model)
        
        ### second forward-backward pass
        with torch.cuda.amp.autocast():
            pred = model(X)
            loss = criterion(pred, y)
        second_step_scaler.scale(loss).backward()

        if sam_first_step_applied:
            optimizer.second_step()
        
        second_step_scaler.step(optimizer)
        second_step_scaler.update()
        
        # Calculate loss and accuracy
        correct += (model(X).argmax(1) == y).type(torch.float).sum().item()
        loss_sum += loss.data.item() * X.size(0)
        num_objects_current += X.size(0)
    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": correct / num_objects_current * 100.0,
    }



def train_sabtl_sgd(dataloader, sabtl_model, criterion, optimizer, device, scaler):
    loss_sum = 0.0
    correct = 0.0
    num_objects_current = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
           
        # Sample weight
        params, _, _ = sabtl_model.sample(1.0)
        # Change weight sample shape to input model
        params = format_weights(params, sabtl_model)

        with torch.cuda.amp.autocast():
            pred = sabtl_model(params, X)
            loss = criterion(pred, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)  # optimizer.step()
        scaler.update()
        optimizer.zero_grad()
        
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss_sum += loss.data.item() * X.size(0)
        num_objects_current += X.size(0)
        
        """
        # ### Checking accuracy with MAP (Mean) solution
        params, _, _ = sabtl_model.sample(0.0)
        params = format_weights(params, sabtl_model)
        pred = sabtl_model(params, X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss_sum += loss.data.item() * X.size(0)
        num_objects_current += X.size(0)
        """
    return{
        "loss" : loss_sum / num_objects_current,
        "accuracy" : correct / num_objects_current * 100.0,
    }


def train_sabtl_sam(dataloader, sabtl_model, criterion, optimizer, device, first_step_scaler, second_step_scaler):
    loss_sum = 0.0
    correct = 0.0
    num_objects_current = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
           
        # Sample weight
        params, z_1, z_2 = sabtl_model.sample(1.0)        
        # Change weight sample shape to input model
        params = format_weights(params, sabtl_model)

        ## first forward & backward
        with torch.cuda.amp.autocast():
            pred = sabtl_model(params, X)
            loss = criterion(pred, y)        
        first_step_scaler.scale(loss).backward()
        first_step_scaler.unscale_(optimizer)
        
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss_sum += loss.data.item() * X.size(0)
        num_objects_current += X.size(0)
        
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
        params = format_weights(params, sabtl_model)
        
        with torch.cuda.amp.autocast():
            pred = sabtl_model(params, X)
            loss = criterion(pred, y)
        second_step_scaler.scale(loss).backward()
        
        if sam_first_step_applied:
            optimizer.second_step()  
        second_step_scaler.step(optimizer)
        second_step_scaler.update()

        """
        # ### Checking accuracy with MAP (Mean) solution
        params, _, _ = sabtl_model.sample(0.0)
        params = format_weights(params, sabtl_model)
        pred = sabtl_model(params, X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss_sum += loss.data.item() * X.size(0)
        num_objects_current += X.size(0)
        """
    return{
        "loss" : loss_sum / num_objects_current,
        "accuracy" : correct / num_objects_current * 100.0,
    }



def train_sabtl_bsam(dataloader, sabtl_model, criterion, optimizer, device, eta, first_step_scaler, second_step_scaler):
    loss_sum = 0.0
    correct = 0.0
    num_objects_current = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Sample weight
        params, z_1, z_2 = sabtl_model.sample(1.0)    

        # compute Fisher inverse
        fish_inv = sabtl_model.fish_inv(params, eta)
        # Change weight sample shape to input model
        params = format_weights(params, sabtl_model)

        ## first forward & backward
        with torch.cuda.amp.autocast():
            pred = sabtl_model(params, X)
            loss = criterion(pred, y)

        first_step_scaler.scale(loss).backward()
        first_step_scaler.unscale_(optimizer)

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss_sum += loss.data.item() * X.size(0)
        num_objects_current += X.size(0)
        
        optimizer_state = first_step_scaler._per_optimizer_states[id(optimizer)]
        
        inf_grad_cnt = sum(v.item() for v in optimizer_state["found_inf_per_device"].values())      # Check if any gradients are inf/nan
        if inf_grad_cnt == 0:
            # if valid graident, apply sam_first_step
            optimizer.first_step(fish_inv, zero_grad=True)
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

        """
        # ### Checking accuracy with MAP (Mean) solution
        params, _, _ = sabtl_model.sample(0.0)
        params = format_weights(params, sabtl_model)
        pred = sabtl_model(params, X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss_sum += loss.data.item() * X.size(0)
        num_objects_current += X.size(0)
        """ 
    return{
        "loss" : loss_sum / num_objects_current,
        "accuracy" : correct / num_objects_current * 100.0,
    }



# Test
def eval(loader, model, criterion, device, num_bins=50, eps=1e-8):
    '''
    get loss, accuracy, nll and ece for every eval step
    '''
    loss_sum = 0.0
    num_objects_total = len(loader.dataset)

    preds = list()
    targets = list()

    model.eval()
    offset = 0
    with torch.no_grad():
        for _, (input, target) in enumerate(loader):
            input, target = input.to(device), target.to(device)
            pred = model(input)
            loss = criterion(pred, target)
            loss_sum += loss.item() * input.size(0)
            
            preds.append(F.softmax(pred, dim=1).cpu().numpy())
            targets.append(target.cpu().numpy())
            offset += input.size(0)
    
    preds = np.vstack(preds)
    targets = np.concatenate(targets)

    accuracy = np.mean(np.argmax(preds, axis=1) == targets)
    nll = -np.mean(np.log(preds[np.arange(preds.shape[0]), targets] + eps))
    ece = calibration_curve(preds, targets, num_bins)['ece']
    
    return {
        "loss" : loss_sum / num_objects_total,
        "accuracy" : accuracy * 100.0,
        "nll" : nll,
        "ece" : ece,
    }



def eval_sabtl(loader, sabtl_model, params, criterion, device, num_bins=50, eps=1e-8):
    '''
    get loss, accuracy, nll and ece for every eval step
    '''

    loss_sum = 0.0
    num_objects_total = len(loader.dataset)

    preds = list()
    targets = list()

    sabtl_model.eval()
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
    ece = calibration_curve(preds, targets, num_bins)['ece']
    
    return {
        "predictions" : preds,
        "targets" : targets,
        "loss" : loss_sum / num_objects_total,
        "accuracy" : accuracy * 100.0,
        "nll" : nll,
        "ece" : ece
    }



def bma(tr_loader, te_loader, model, bma_num_models, num_classes, bma_save_path=None, eps=1e-8, batch_norm=True):
    '''
    run bayesian model averaging in test step
    '''
    swag_predictions = np.zeros((len(te_loader.dataset), num_classes))
    with torch.no_grad():
        for i in range(bma_num_models):
            
            if i == 0:
                sample = model.sample(0)
            else:
                sample = model.sample(1.0, cov=True)
            
            # print("SWAG Sample %d/%d. BN update" % (i + 1, bma_num_models))
            if batch_norm:
                bn_update(tr_loader, model, verbose=False, subset=1.0)
            
            # save sampled weight for bma
            if bma_save_path is not None:
                torch.save(sample, f'{bma_save_path}/bma_model-{i}.pt')
            
            # print("SWAG Sample %d/%d. EVAL" % (i + 1, bma_num_models))
            res = predict(te_loader, model, verbose=False)

            predictions = res["predictions"]
            targets = res["targets"]

            accuracy = np.mean(np.argmax(predictions, axis=1) == targets)
            nll = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), targets] + eps))
            print(
                "SWAG Sample %d/%d. Accuracy: %.2f%% NLL: %.4f"
                % (i + 1, bma_num_models, accuracy * 100, nll)
            )

            swag_predictions += predictions

            ens_accuracy = np.mean(np.argmax(swag_predictions, axis=1) == targets)
            ens_nll = -np.mean(
                np.log(
                    swag_predictions[np.arange(swag_predictions.shape[0]), targets] / (i + 1)
                    + eps
                )
            )
            print(
                "Ensemble %d/%d. Accuracy: %.2f%% NLL: %.4f"
                % (i + 1, bma_num_models, ens_accuracy * 100, ens_nll)
            )

        swag_predictions /= bma_num_models

        swag_accuracy = np.mean(np.argmax(swag_predictions, axis=1) == targets)
        swag_nll = -np.mean(
            np.log(swag_predictions[np.arange(swag_predictions.shape[0]), targets] + eps)
        )

    print(f"bma Accuracy using {bma_num_models} model : {swag_accuracy * 100:.2f}% / NLL : {swag_nll:.4f}")
    return {"predictions" : swag_predictions,
            "targets" : targets,
            "bma_accuracy" : swag_accuracy,
            "nll" : swag_nll
    }


def bma_sabtl(te_loader, sabtl_model, bma_num_models,
            num_classes, criterion, device,
            bma_save_path=None, eps=1e-8, num_bins=50,
            ):
    '''
    run bayesian model averaging in test step
    '''
    sabtl_predictions = np.zeros((len(te_loader.dataset), num_classes))
    with torch.no_grad():
        for i in range(bma_num_models):
            
            if i == 0:
                params, _, _ = sabtl_model.sample(0)
            else:
                params, _, _  = sabtl_model.sample(1.0)
            
            # save sampled weight for bma
            if bma_save_path is not None:
                torch.save(params, f'{bma_save_path}/bma_model-{i}.pt')
            
            params = format_weights(params, sabtl_model)
            res = eval_sabtl(te_loader, sabtl_model, params, criterion, device, num_bins, eps)

            print(
                "SABTL Sample %d/%d. Accuracy: %.2f%% NLL: %.4f"
                % (i + 1, bma_num_models, res['accuracy'], res['nll'])
            )

            sabtl_predictions += res["predictions"]

            ens_accuracy = np.mean(np.argmax(sabtl_predictions, axis=1) == res["targets"])
            ens_nll = -np.mean(
                np.log(
                    sabtl_predictions[np.arange(sabtl_predictions.shape[0]), res["targets"]] / (i + 1)
                    + eps
                )
            )
            print(
                "Ensemble %d/%d. Accuracy: %.2f%% NLL: %.4f"
                % (i + 1, bma_num_models, ens_accuracy * 100, ens_nll)
            )

        sabtl_predictions /= bma_num_models

        sabtl_accuracy = np.mean(np.argmax(sabtl_predictions, axis=1) == res["targets"])
        sabtl_nll = -np.mean(
            np.log(sabtl_predictions[np.arange(sabtl_predictions.shape[0]), res["targets"]] + eps)
        )

    print(f"bma Accuracy using {bma_num_models} model : {sabtl_accuracy * 100:.2f}% / NLL : {sabtl_nll:.4f}")
    return {"predictions" : sabtl_predictions,
            "targets" : res["targets"],
            "bma_accuracy" : sabtl_accuracy,
            "nll" : sabtl_nll
    }



def calibration_curve(predictions, targets, num_bins):
    confidences = np.max(predictions, 1)
    step = (confidences.shape[0] + num_bins - 1) // num_bins
    bins = np.sort(confidences)[::step]
    if confidences.shape[0] % step != 1:
        bins = np.concatenate((bins, [np.max(confidences)]))
    # bins = np.linspace(0.1, 1.0, 30)
    predictions = np.argmax(predictions, 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    accuracies = predictions == targets

    xs = []
    ys = []
    zs = []

    # ece = Variable(torch.zeros(1)).type_as(confidences)
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = (confidences > bin_lower) * (confidences < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            xs.append(avg_confidence_in_bin)
            ys.append(accuracy_in_bin)
            zs.append(prop_in_bin)
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    out = {"confidence": xs, "accuracy": ys, "p": zs, "ece": ece}
    return out



def save_reliability_diagram(method, optim, save_path, unc, bma=False):
    plt.clf()
    plt.plot(unc['confidence'], unc['confidence'] - unc['accuracy'], 'r', label=f'{method}-{optim}')
    plt.xlabel("confidence")
    plt.ylabel("confidence - accuracy")
    plt.axhline(y=0, color='black')
    plt.title('Reliability Diagram')
    plt.legend()
    if bma:
        plt.savefig(f'{save_path}/unc_result/{method}_{optim}_bma_reliability_diagram.png')    
        
    else:
        plt.savefig(f'{save_path}/unc_result/{method}_{optim}_reliability_diagram.png')    