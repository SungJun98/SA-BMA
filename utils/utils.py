import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sam import sam, sam_utils
from utils.vi import vi_utils

from utils.models import resnet_noBN
import torchvision.models as torch_models
import timm

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
    ### Few-shot part
    if args.dat_per_cls >= 0:
        save_path_ = f"{args.save_path}/seed_{args.seed}/{args.dataset}/{args.dat_per_cls}shot"
    else:
        save_path_ = f"{args.save_path}/seed_{args.seed}/{args.dataset}/"
        
    ### scheduler part   
    if args.scheduler == "cos_anneal":
        save_path_ = f"{save_path_}/{args.model}/{args.method}-{args.optim}/{args.scheduler}_{args.t_max}"
    elif args.scheduler == "swag_lr":
        save_path_ = f"{save_path_}/{args.model}/{args.method}-{args.optim}/{args.scheduler}_{args.swa_lr}"
    elif args.scheduler == "cos_decay":
        # save_path_ = f"{args.save_path}/{args.dataset}/{args.model}/{args.method}-{args.optim}/{args.scheduler}({args.first_cycle_steps}/{args.cycle_mult}/{args.min_lr}/{args.warmup_steps}/{args.decay_ratio})"
        save_path_ = f"{save_path_}/{args.model}/{args.method}-{args.optim}/{args.scheduler}_{args.lr_min}/{args.warmup_t}_{args.warmup_lr_init}"
    else:    
        save_path_ = f"{save_path_}/{args.model}/{args.method}-{args.optim}/{args.scheduler}"
    
    ## learning hyperparameter part
    if args.method in ["swag", "last_swag"]:
        save_path_ = f"{save_path_}/{args.lr_init}_{args.wd}_{args.max_num_models}_{args.swa_start}_{args.swa_c_epochs}"
    elif args.method in ["vi"]:
        save_path_ = f"{save_path_}/{args.lr_init}_{args.wd}_{args.vi_prior_sigma}_{args.vi_posterior_rho_init}_{args.vi_moped_delta}"
    else:
        save_path_ = f"{save_path_}/{args.lr_init}_{args.wd}_{args.momentum}"
        
    if args.optim not in ["sgd", "adam"]:
        save_path_ = f"{save_path_}_{args.rho}"
        
    if args.optim in ["bsam"]:
        save_path_ = f"{save_path_}_{args.eta}"
    
    return save_path_
    


def set_wandb_runname(args):
    '''
    Set wandb run name following the method / model / dataset / optimizer / hyperparameters
    '''
    ### Few-shot part
    if args.dat_per_cls >= 0:
        run_name_ = f"seed{args.seed}_{args.method}-{args.optim}_{args.model}_{args.dataset}_{args.dat_per_cls}shot"
    else:
        run_name_ = f"seed{args.seed}_{args.method}-{args.optim}_{args.model}_{args.dataset}"
        
    ### scheduler part
    if args.scheduler == "cos_anneal":
        run_name_ = f"{run_name_}_{args.scheduler}({args.t_max})"
    elif args.scheduler == "swag_lr":
        run_name_ = f"{run_name_}_{args.scheduler}({args.swa_lr})"
    elif args.scheduler == "cos_decay":
        run_name_ = f"{run_name_}_{args.scheduler}({args.lr_min}_{args.warmup_t}_{args.warmup_lr_init})"
    else:
        run_name_ = f"{run_name_}_{args.scheduler}"
    
    ## learning hyperparameter part
    if args.method in ["swag", "last_swag"]:
        run_name_ = f"{run_name_}_{args.lr_init}_{args.wd}_{args.max_num_models}_{args.swa_start}_{args.swa_c_epochs}"
    elif args.method in ["vi"]:
        run_name_ = f"{run_name_}_{args.lr_init}_{args.wd}_{args.vi_prior_sigma}_{args.vi_posterior_rho_init}_{args.vi_moped_delta}"
    else:
        run_name_ = f"{run_name_}/{args.lr_init}_{args.wd}_{args.momentum}"
        
    if args.optim not in ["sgd", "adam"]:
        run_name_ = f"{run_name_}_{args.rho}"
    
    if args.optim in ["bsam"]:
        run_name_ = f"{run_name_}_{args.eta}"
    
    return run_name_



def get_dataset(dataset='cifar10',
                data_path='/data2/lsj9862/data/cifar10',
                dat_per_cls=-1,
                use_validation=True, 
                batch_size=256,
                num_workers=4,
                seed=0,
                aug=True,
                ):
    
    import utils.data.data as data
    
    ## Define Transform
    transform_train, transform_test = data.create_transform_v2(aug=aug)
    
    ## Load Data
    tr_data, val_data, te_data, num_classes = data.create_dataset(data_name=dataset, data_path=data_path,
                                        use_validation=use_validation,
                                        dat_per_cls=dat_per_cls, seed=seed,
                                        transform_train=transform_train, transform_test=transform_test,
                                        )
    
    ## Create loader
    tr_loader, val_loader, te_loader = data.create_loader(tr_data=tr_data, val_data=val_data, te_data=te_data,
                                            use_validation=use_validation,
                                            batch_size=batch_size, num_workers=num_workers, dat_per_cls=dat_per_cls,
                                            )
        

    return tr_loader, val_loader, te_loader, num_classes



def get_backbone(model_name, num_classes, device, pre_trained=True):
    '''
    Define Backbone Model
    '''
    ## ResNet
    if model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        model_cfg = getattr(torch_models, model_name)
        model = model_cfg(pretrained=pre_trained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    ## ResNet18-noBN
    elif model_name == "resnet18-noBN":
        model = resnet_noBN.resnet18(num_classes=num_classes)
    
    ## ViT-B/16-ImageNet21K
    if model_name == "vitb16-i21k":
        model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
        model.head = torch.nn.Linear(768, num_classes)
    
    model.to(device)
    print(f"Preparing model {model_name}")
    return model


def get_optimizer(args, model):
    '''
    Define optimizer
    '''
    if args.linear_probe:
        if args.model == 'vitb16-i21k':
            optim_param = model.head.parameters()
        elif args.model in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            optim_param = model.fc.parameters()
    else:
        optim_param = model.parameters()

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(optim_param,
                            lr=args.lr_init, weight_decay=args.wd,
                            momentum=args.momentum)
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(optim_param,
                            lr=args.lr_init, weight_decay=args.wd)
    elif args.optim == "sam":
        base_optimizer = torch.optim.SGD
        optimizer = sam.SAM(optim_param, base_optimizer, rho=args.rho, lr=args.lr_init, momentum=args.momentum,
                        weight_decay=args.wd)
        
    return optimizer



def get_scheduler(args, optimizer):
    '''
    Define Scheduler
    '''
    if args.scheduler == "step_lr":
        from timm.scheduler.step_lr import StepLRScheduler
        if args.optim in ['sgd', "adam"]:
            scheduler_ = StepLRScheduler(optimizer, decay_rate=0.2, )
        elif args.optim in ['sam', 'bsam']:
            scheduler_ = StepLRScheduler(optimizer.base_optimizer, decay_rate=0.2, )
            
    elif args.scheduler == "cos_decay":
        from timm.scheduler.cosine_lr import CosineLRScheduler
        if args.optim in ["sgd", "adam"]:
            scheduler_ = CosineLRScheduler(optimizer = optimizer,
                                        t_initial= args.epochs,
                                        lr_min=args.lr_min,
                                        cycle_mul=1,
                                        cycle_decay=1,
                                        cycle_limit=1,
                                        warmup_t=args.warmup_t,
                                        warmup_lr_init=args.warmup_lr_init,
                                            )
        elif args.optim in ["sam", "bsam"]:
            scheduler_ = CosineLRScheduler(optimizer = optimizer.base_optimizer,
                                        t_initial= args.epochs,
                                        lr_min=args.lr_min,
                                        cycle_mul=1,
                                        cycle_decay=1,
                                        cycle_limit=1,
                                        warmup_t=args.warmup_t,
                                        warmup_lr_init=args.warmup_lr_init,
                                            )
    
    return scheduler_



def get_scaler(args):
    '''
    Define Scaler for AMP
    '''
    if not args.no_amp:
        if args.optim in ["sgd", "adam"]:
            scaler = torch.cuda.amp.GradScaler()
            first_step_scaler = None
            second_step_scaler = None

        elif args.optim in ["sam", "bsam"]:
            scaler = None
            first_step_scaler = torch.cuda.amp.GradScaler(2 ** 8)
            second_step_scaler = torch.cuda.amp.GradScaler(2 ** 8)

        print(f"Set AMP Scaler for {args.optim}")

    else:
        scaler = None
        first_step_scaler = None
        second_step_scaler = None
    
    return scaler, first_step_scaler, second_step_scaler



def freeze_fe(model):
    '''
    Freezing Feature Extractor
    '''
    for name, _ in model.named_modules():
        last_layer_name = name
    
    for name, param in model.named_parameters():
        if name.split('.')[0] in last_layer_name:
            continue
        param.requires_grad = False
    
    print("Freeze Feature Extractor for Linear Probing")



def save_checkpoint(file_path, epoch, **kwargs):
    '''
    Save Model Checkpoint
    '''
    state = {"epoch": epoch}
    state.update(kwargs)
    torch.save(state, file_path)




def save_best_dnn_model(args, best_epoch, model, optimizer, scaler, first_step_scaler, second_step_scaler):
    if args.optim in ["sgd", "adam"]:
        if not args.no_amp:
            save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                            epoch = best_epoch,
                            state_dict = model.state_dict(),
                            optimizer = optimizer.state_dict(),
                            # scheduler = scheduler.state_dict(),
                            scaler = scaler.state_dict()
                            )
        else:
            save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                            epoch = best_epoch,
                            state_dict = model.state_dict(),
                            optimizer = optimizer.state_dict(),
                            # scheduler = scheduler.state_dict(),
                            )
    elif args.optim == "sam":
        if not args.no_amp:
            save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                            epoch = best_epoch,
                            state_dict = model.state_dict(),
                            optimizer = optimizer.state_dict(),
                            # scheduler = scheduler.state_dict(),
                            first_step_scaler = first_step_scaler.state_dict(),
                            second_step_scaler = second_step_scaler.state_dict()
                            )
        else:
            save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                            epoch = best_epoch,
                            state_dict = model.state_dict(),
                            optimizer = optimizer.state_dict(),
                            # scheduler = scheduler.state_dict(),
                            )


def save_best_swag_model(args, best_epoch, model, swag_model, optimizer, scaler, first_step_scaler, second_step_scaler):
    if args.optim in ["sgd", "adam"]:
        if not args.no_amp:
            save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                epoch = best_epoch,
                                state_dict = swag_model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                # scheduler = scheduler.state_dict(),
                                scaler = scaler.state_dict()
                                )
        else:
            save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                epoch = best_epoch,
                                state_dict = swag_model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                # scheduler = scheduler.state_dict(),
                                )
    elif args.optim in ["sam", "bsam"]:
        if not args.no_amp:
            save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                epoch = best_epoch,
                                state_dict = swag_model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                # scheduler = scheduler.state_dict(),
                                first_step_scaler = first_step_scaler.state_dict(),
                                second_step_scaler = second_step_scaler.state_dict()
                                )
        else:
            save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
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



def save_best_vi_model(args, best_epoch, model, optimizer, scaler, first_step_scaler, second_step_scaler):
    save_best_dnn_model(args, best_epoch, model, optimizer, scaler, first_step_scaler, second_step_scaler)
    
    mean = vi_utils.get_vi_mean_vector(model)
    torch.save(mean,f'{args.save_path}/{args.method}-{args.optim}_best_val_mean.pt')
    
    variance = vi_utils.get_vi_variance_vector(model)
    torch.save(variance, f'{args.save_path}/{args.method}-{args.optim}_best_val_variance.pt')
    
    return mean, variance



def save_best_sabtl_model(args, best_epoch, sabtl_model, optimizer, scaler, first_step_scaler, second_step_scaler):
    if args.optim == "sgd":
        if not args.no_amp:
            save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                            epoch = best_epoch,
                            state_dict =sabtl_model.state_dict(),
                            optimizer = optimizer.state_dict(),
                            # scheduler = scheduler.state_dict(),
                            scaler = scaler.state_dict(),
                            )
        else:
            save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                            epoch = best_epoch,
                            state_dict =sabtl_model.state_dict(),
                            optimizer = optimizer.state_dict(),
                            # scheduler = scheduler.state_dict(),
                            )
    elif args.optim in ["sam", "bsam"]:
        if not args.no_amp:
            save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
                                epoch = best_epoch,
                                state_dict = sabtl_model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                # scheduler = scheduler.state_dict(),
                                first_step_scaler = first_step_scaler.state_dict(),
                                second_step_scaler = second_step_scaler.state_dict()
                                )
        else:
            save_checkpoint(file_path = f"{args.save_path}/{args.method}-{args.optim}_best_val.pt",
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
    

"""
##################################################################################################
#
# 아래 부분 수정
#
##################################################################################################
"""

# parameter list to state_dict(ordered Dict)
def list_to_state_dict(model, sample_list, last=False, last_layer_name="fc"):
    '''
    Change sample list to state dict
    '''
    ordDict = collections.OrderedDict()
    if last:
        ordDict[f"{last_layer_name}.weight"] = sample_list[0]
        ordDict[f"{last_layer_name}.bias"] = sample_list[1]   
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
    if sabtl_model.last_layer:
        model_shape = sabtl_model.last_layer_shape
    else:
        model_shape = sabtl_model.full_model_shape
    sample = unflatten_like_size(sample, model_shape)
    
    state_dict = sabtl_model.backbone.state_dict().copy()
    if sabtl_model.last_layer:
        for name,  in sabtl_model.backbone.named_modules():
            if sabtl_model.last_layer_name in name:
                state_dict[f"{sabtl_model.last_layer_name}.weight"] = sample[0]
                state_dict[f"{sabtl_model.last_layer_name}.bias"] = sample[1]
    else:
        for idx, (name, _) in enumerate(sabtl_model.backbone.named_parameters()):
            state_dict[name] = sample[idx]
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

        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(X)
                loss = criterion(pred, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)  # optimizer.step()
                scaler.update()
                optimizer.zero_grad()
        else:
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
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

        if first_step_scaler is not None:
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
        
        else:
            ## first forward & backward
            pred = model(X)
            loss = criterion(pred, y)        
            loss.backward()
            optimizer.first_step(zero_grad=True, amp=False)
            
            ## second forward-backward pass
            pred = model( X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.second_step(zero_grad=True, amp=False)   
                       
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            loss_sum += loss.data.item() * X.size(0)
            num_objects_current += X.size(0)
                      
    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": correct / num_objects_current * 100.0,
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
    unc = calibration_curve(preds, targets, num_bins)
        
    return {
        "loss" : loss_sum / num_objects_total,
        "accuracy" : accuracy * 100.0,
        "nll" : nll,
        "ece" : unc['ece'],
        "unc" : unc
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
    
    os.makedirs(f'{save_path}/unc_result', exist_ok=True)
    if bma:
        plt.savefig(f'{save_path}/unc_result/{method}_{optim}_bma_reliability_diagram.png')           
    else:
        plt.savefig(f'{save_path}/unc_result/{method}_{optim}_reliability_diagram.png')    