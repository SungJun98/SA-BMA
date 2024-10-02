import numpy as np
import pickle, wandb
import os

import matplotlib.pyplot as plt
import collections, tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sam import sam, sam_utils
from utils.swag import swag_utils
from utils.vi import vi_utils
from utils.la import la_utils
from utils import temperature_scaling as ts
from utils.bsam import bsam, bsam_utils

from utils.models import resnet_noBN, mlp, vit_light, simple_vit
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
        
    method = args.method

    ### pre-trained / linear_probe / scratch
    if args.pre_trained:
        am = "pretrained"
    else:
        am = "scratch"        

    ### scheduler part  
    if args.scheduler == "swag_lr":
        save_path_ = f"{save_path_}/{am}_{args.model}/{method}-{args.optim}/{args.scheduler}_{args.swa_lr}"
    elif args.scheduler == "cos_decay":
        save_path_ = f"{save_path_}/{am}_{args.model}/{method}-{args.optim}/{args.scheduler}_{args.lr_min}/{args.warmup_t}_{args.warmup_lr_init}"
    else:    
        save_path_ = f"{save_path_}/{am}_{args.model}/{method}-{args.optim}/{args.scheduler}"
    
    ## learning hyperparameter part
    if args.method in ["swag", "ll_swag"]:
        save_path_ = f"{save_path_}/{args.lr_init}_{args.wd}_{args.max_num_models}_{args.swa_start}_{args.swa_c_epochs}"
    elif args.method in ["vi", "ll_vi"]:
        save_path_ = f"{save_path_}/{args.lr_init}_{args.wd}_{args.vi_prior_sigma}_{args.vi_posterior_rho_init}_{args.vi_moped_delta}_{args.kl_beta}"
    elif args.method in ['mcmc', 'emcmc']:
        save_path_ = f"{save_path_}/{args.lr_init}_{args.wd}_{args.n_cycle}_{args.temp}_{args.eta}"
    elif args.method in ["sabma"]:
        save_path_ = f"{save_path_}/{args.lr_init}_{args.wd}_{args.momentum}_{args.low_rank}"
    else:
        save_path_ = f"{save_path_}/{args.lr_init}_{args.wd}_{args.momentum}"
        
    if args.optim in ["sam", "fsam", "bsam"]:
        save_path_ = f"{save_path_}_{args.rho}"
        if args.optim in ["fsam"]:
            save_path_ = f"{save_path_}_{args.eta}"
        elif args.optim in ["bsam"]:
            save_path_ = f"{save_path_}_{args.beta2}_{args.damping}_{args.noise_scale}_{args.s_init}"
    if args.optim in ["sabma"]:
        save_path_ = f"{save_path_}_{args.kl_eta}_{args.alpha}"
    
    return save_path_
    


def set_wandb_runname(args):
    '''
    Set wandb run name following the method / model / dataset / optimizer / hyperparameters
    '''

    method = args.method
    if args.method == 'sabma':
        if args.tr_layer == 'll':
            method = 'll_' + method
            
        if args.diag_only:
            method = 'diag_' + method
        

    ### pre-trained / linear_probe / scratch
    if args.pre_trained:
        am = "pretrained"
    else:
        am = "scratch" 
    
    ### Few-shot part
    if args.dat_per_cls >= 0:
        run_name_ = f"seed{args.seed}_{method}-{args.optim}_{am}-{args.model}_{args.dataset}_{args.dat_per_cls}shot"
    else:
        run_name_ = f"seed{args.seed}_{method}-{args.optim}_{am}-{args.model}_{args.dataset}"

    ### scheduler part
    if args.scheduler == "swag_lr":
        run_name_ = f"{run_name_}_{args.scheduler}({args.swa_lr})"
    elif args.scheduler == "cos_decay":
        run_name_ = f"{run_name_}_{args.scheduler}({args.lr_min}_{args.warmup_t}_{args.warmup_lr_init})"
    else:
        run_name_ = f"{run_name_}_{args.scheduler}"
    
    ## learning hyperparameter part
    if args.method in ["swag", "ll_swag"]:
        run_name_ = f"{run_name_}_{args.lr_init}_{args.wd}_{args.max_num_models}_{args.swa_start}_{args.swa_c_epochs}"
    elif args.method in ["vi", "ll_vi"]:
        run_name_ = f"{run_name_}_{args.lr_init}_{args.wd}_{args.vi_prior_sigma}_{args.vi_posterior_rho_init}_{args.vi_moped_delta}"
    elif args.method in ['mcmc', 'emcmc']:    
        run_name_ = f"{run_name_}_{args.lr_init}_{args.wd}_{args.n_cycle}_{args.temp}_{args.eta}"
    elif args.method in ["sabma"]:
        run_name_ = f"{run_name_}/{args.lr_init}_{args.wd}_{args.momentum}_{args.low_rank}"
    else:
        run_name_ = f"{run_name_}/{args.lr_init}_{args.wd}_{args.momentum}"
        
    if args.optim in ["sam", "fsam", "bsam"]:
        run_name_ = f"{run_name_}_{args.rho}"
        if args.optim in ["fsam"]:
            run_name_ = f"{run_name_}_{args.eta}"
        elif args.optim in ["bsam"]:
            run_name_ = f"{run_name_}_{args.beta2}_{args.damping}_{args.noise_scale}_{args.s_init}"
    
    if args.optim in ["sabma"]:
        run_name_ = f"{run_name_}_{args.kl_eta}_{args.alpha}"
    
    return run_name_


def get_dataset(dataset='cifar10',
                data_path=None,
                dat_per_cls=-1,
                use_validation=True, 
                batch_size=256,
                num_workers=4,
                seed=0,
                aug=True,
                model_name='resnet18'
                ):
    
    import utils.data.data as data
    
    ## Define Transform
    transform_train, transform_test = data.create_transform_v2(data_name=dataset, aug=aug, model_name=model_name)
        
    ## Load Data
    tr_data, val_data, te_data, num_classes = data.create_dataset(data_name=dataset, data_path=data_path,
                                        use_validation=use_validation,
                                        dat_per_cls=dat_per_cls, seed=seed,
                                        transform_train=transform_train, transform_test=transform_test,
                                        )
    
    ## Create loader
    tr_loader, val_loader, te_loader = data.create_loader(data_name=dataset,
                                            tr_data=tr_data, val_data=val_data, te_data=te_data,
                                            use_validation=use_validation,
                                            batch_size=batch_size, num_workers=num_workers, dat_per_cls=dat_per_cls,
                                            )

    return tr_loader, val_loader, te_loader, num_classes



def get_dataset_dassl(args):
    from my_dassl.data import DataManager
    from my_dassl.config import get_cfg_default
    
    import my_dassl.datasets.oxford_pets
    import my_dassl.datasets.oxford_flowers
    import my_dassl.datasets.fgvc_aircraft
    import my_dassl.datasets.dtd
    import my_dassl.datasets.eurosat
    import my_dassl.datasets.stanford_cars
    import my_dassl.datasets.food101
    import my_dassl.datasets.sun397
    import my_dassl.datasets.caltech101
    import my_dassl.datasets.ucf101
    import my_dassl.datasets.imagenet
    import my_dassl.datasets.svhn
    import my_dassl.datasets.resisc45
    import my_dassl.datasets.clevr

    import my_dassl.datasets.locmnist
    
    cfg = get_cfg_default()
    
    dataset_config_file = f'./my_dassl/datasets/config/{args.dataset}.yaml'
    cfg.SEED = args.seed
    cfg.merge_from_file(dataset_config_file)
    cfg.DATASET.ROOT = args.data_path
    cfg.DATASET.NUM_SHOTS = args.dat_per_cls
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    
    dm = DataManager(cfg)
    tr_loader = dm.train_loader_x
    val_loader = dm.val_loader
    te_loader = dm.test_loader
    num_classes = dm.num_classes
    
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
    
    ## ViT-B/16 pre-trained on ImageNet21K
    elif model_name == "vitb16-i21k":
        model = timm.create_model('vit_base_patch16_224_in21k', pretrained=pre_trained)
        model.head = torch.nn.Linear(768, num_classes)
    
    ## ViT-B/16 pre-trained on ImageNet11K
    elif model_name == 'vitb16-i1k':
        if pre_trained == True:
            model = torch_models.vit_b_16(weights='IMAGENET1K_V1')
            model.heads.head = torch.nn.Linear(768, num_classes)
        else:
            model = vit_light.ViT_light(num_classes=num_classes)
            print("[Warning] You load light version of ViT-B/16 for scratch training")
    
    ## mlp
    elif model_name == 'mlp':
        model = mlp.MLP(input_size=784, hidden_size=32, output_size=10) ## for MNIST
    
    else:
        raise NotImplementedError("No code for the backbone")
    
    model.to(device)
    if pre_trained:
        print(f"Preparing Pre-trained model {model_name}")
    else:
        print(f"Preparing model {model_name}")
    return model


def get_optimizer(args, model, num_classes=10):
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
        
    elif args.optim == "fsam":
        base_optimizer = torch.optim.SGD
        optimizer = sam.FSAM(optim_param, base_optimizer, rho=args.rho, eta=args.eta, lr=args.lr_init, momentum=args.momentum,
                        weight_decay=args.wd)
        
    elif args.optim == "bsam":
        if args.dat_per_cls < 0:
            args.dat_per_cls = 5000 if args.dataset == 'cifar10' else 500
        optimizer = bsam.bSAM(optim_param, Ndata=num_classes * args.dat_per_cls, lr=args.lr_init, 
                            betas=(args.momentum, args.beta2), weight_decay=args.wd, rho=args.rho,
                            noise_scale=args.noise_scale, s_init=args.s_init, damping=args.damping)
        
    return optimizer



def get_scheduler(args, optimizer):
    '''
    Define Scheduler
    '''
    if args.scheduler == "step_lr":
        from timm.scheduler.step_lr import StepLRScheduler
        if args.optim in ['sgd', "adam"]:
            scheduler_ = StepLRScheduler(optimizer, decay_rate=0.2, )
        elif args.optim in ['sam', 'fsam', 'sabma']:
            scheduler_ = StepLRScheduler(optimizer.base_optimizer, decay_rate=0.2, )
            
    elif args.scheduler == "cos_decay":
        from timm.scheduler.cosine_lr import CosineLRScheduler
        if args.optim in ["sgd", "adam", "bsam"]:
            scheduler_ = CosineLRScheduler(optimizer = optimizer,
                                        t_initial= args.epochs,
                                        lr_min=args.lr_min,
                                        cycle_mul=1,
                                        cycle_decay=1,
                                        cycle_limit=1,
                                        warmup_t=args.warmup_t,
                                        warmup_lr_init=args.warmup_lr_init,
                                            )
        elif args.optim in ["sam", "fsam", "sabma"]:
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

        elif args.optim in ["sam", "fsam", "bsam", "sabma"]:
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



def unfreeze_norm_layer(model):
    import torch.nn as nn
    for module in model.modules():
        if (isinstance(module, nn.LayerNorm)) or (isinstance(module, nn.BatchNorm2d)):
            for param in module.parameters():
                param.requires_grad = True


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
    elif args.optim in ["sam", "fsam", "bsam"]:
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

    

# parameter list to state_dict(ordered Dict)
def list_to_state_dict(model, sample_list, tr_layer="last_layer", last_layer_name="fc"):
    '''
    Change sample list to state dict
    '''
    ordDict = collections.OrderedDict()
    if tr_layer == "last_layer":
        ordDict[f"{last_layer_name}.weight"] = sample_list[0]
        ordDict[f"{last_layer_name}.bias"] = sample_list[1]   
    elif tr_layer == "last_block":
        raise NotImplementedError("Need code for last block SA-BMA")
    else:        
        for sample, (name, param) in zip(sample_list, model.named_parameters()):
            ordDict[name] = sample
    return ordDict


def unflatten_like_size(vector, likeTensorSize):
    """
    Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    Input
     - vector : flattened parameters
     - likeTensorSize : list of torch.Size
    """
    outList = []
    i = 0
    for layer_size in likeTensorSize:
        n = layer_size.numel()
        outList.append(vector[i : i + n].view(layer_size))
        i += n

    return outList


    
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
    for batch_idx, batch in enumerate(dataloader):
        try:
            X = batch["img"].to(device)
            y = batch["label"].to(device)
        except:
            X, y = batch[0].to(device), batch[1].to(device)

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
    for batch_idx, batch in enumerate(dataloader):
        try:
            X = batch["img"].to(device)
            y = batch["label"].to(device)
        except:
            X, y = batch[0].to(device), batch[1].to(device)
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
            
        
        else:
            ## first forward & backward
            pred = model(X)
            loss = criterion(pred, y)        
            loss.backward()
            optimizer.first_step(zero_grad=True, amp=False)
            
            ## second forward-backward pass
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.second_step(zero_grad=True, amp=False)   
        
        # Calculate loss and accuracy              
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss_sum += loss.data.item() * X.size(0)
        num_objects_current += X.size(0)
                      
    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": correct / num_objects_current * 100.0,
    }



# train FSAM
def train_fsam(dataloader, model, criterion, optimizer, device, first_step_scaler, second_step_scaler):
    # https://github.com/davda54/sam/issues/7
    loss_sum = 0.0
    correct = 0.0

    num_objects_current = 0
    num_batches = len(dataloader)

    model.train()
    for batch_idx, batch in enumerate(dataloader):
        try:
            X = batch["img"].to(device)
            y = batch["label"].to(device)
        except:
            X, y = batch[0].to(device), batch[1].to(device)
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
        
        else:
            ## first forward & backward
            pred = model(X)
            loss = criterion(pred, y)        
            loss.backward()
            optimizer.first_step(zero_grad=True, amp=False)
            
            ## second forward-backward pass
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.second_step(zero_grad=True, amp=False)   
            
        # Calculate loss and accuracy               
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss_sum += loss.data.item() * X.size(0)
        num_objects_current += X.size(0)
                      
    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": correct / num_objects_current * 100.0,
    }


# Train bSAM
def train_bsam(dataloader, model, criterion, optimizer, device):
    loss_sum = 0.0
    correct = 0.0

    num_objects_current = 0

    model.train()
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        
        ## noisy sample : p + e
        optimizer.add_noise(zero_grad=True)
        
        ## perturb parameter : p + eps
        pred = model(X)
        loss = criterion(pred, y)        
        loss.backward() # gradient of "p + e"
        optimizer.first_step(zero_grad=True)

        ## actual sharpness-aware update
        pred = model(X)
        loss = criterion(pred, y)        
        loss.backward() # gradient of "p + eps"
        optimizer.second_step(zero_grad=True)

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss_sum += loss.data.item() * X.size(0)
        num_objects_current += X.size(0)

    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": correct / num_objects_current * 100.0,
    }
    
    

# Test
def eval(loader, model, criterion, device, num_bins=15, eps=1e-8):
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
        for batch_idx, batch in enumerate(loader):
            try:
                input = batch["img"].to(device)
                target = batch["label"].to(device)
            except:
                input, target = batch[0].to(device), batch[1].to(device)
        
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
        "predictions" : preds,
        "targets" : targets,
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
        


def load_best_model(args, model, swag_model, num_classes):
    print("Load Best Validation Model (Lowest Loss)")
    state_dict_path = f'{args.save_path}/{args.method}-{args.optim}_best_val.pt'
    checkpoint = torch.load(state_dict_path)
    if not args.ignore_wandb:
        wandb.run.summary['Best epoch'] = checkpoint["epoch"]
    mean = None; variance = None
    if args.method in ["swag", "ll_swag"]:
        swag_model.load_state_dict(checkpoint["state_dict"])
        model = swag_model
        
    elif args.method in ["vi", "ll_vi"] or args.optim == 'bsam':
        model = get_backbone(args.model, num_classes, args.device, args.pre_trained)
        if args.method == "ll_vi":
            vi_utils.make_ll_vi(args, model)
        vi_utils.load_vi(model, checkpoint)
        # mean = vi_utils.get_vi_mean_vector(model)
        # variance = vi_utils.get_vi_variance_vector(model)
        mean = torch.load(f'{args.save_path}/{args.method}-{args.optim}_best_val_mean.pt')
        variance = torch.load(f'{args.save_path}/{args.method}-{args.optim}_best_val_variance.pt')
        
    else:
        model.load_state_dict(checkpoint["state_dict"])
        
    model.to(args.device)        
    
    return model, mean, variance, checkpoint["epoch"]
 

def no_ts_map_estimation(args, te_loader, num_classes, model, mean, variance, criterion):
    if args.method in ["swag", "ll_swag"]:
        model.sample(0)
        res = eval(te_loader, model, criterion, args.device)
    elif args.method in ["vi", "ll_vi"]:
        res = vi_utils.bma_vi(None, te_loader, mean, variance, model, args.method, criterion, num_classes, temperature=None, bma_num_models=1,  bma_save_path=None, num_bins=args.num_bins, eps=args.eps)  
    elif args.method in ["la", "ll_la"]:
        res = la_utils.eval_la(te_loader, model, criterion, args.device)
    else:
        res = eval(te_loader, model, criterion, args.device)
    
    return res


def ts_map_estimation(args, val_loader, te_loader, num_classes, model, mean, variance, criterion, save=True):
    # Temperature Scaled Results
    scaled_model = None
    if args.method in ["dnn", "swag", "ll_swag"]:
        scaled_model = ts.ModelWithTemperature(model)
        scaled_model.set_temperature(val_loader)
        temperature = scaled_model.temperature
        if save:
            torch.save(scaled_model, f"{args.save_path}/{args.method}-{args.optim}_best_val_scaled_model.pt")
        if not args.ignore_wandb:
            wandb.run.summary['temperature'] = temperature.item()
        res = eval(te_loader, scaled_model, criterion, args.device)   
        
    elif args.method in ["vi", "ll_vi"]:
        res = vi_utils.bma_vi(val_loader, te_loader, mean, variance, model, args.method, criterion, num_classes, temperature='local', bma_num_models=1,  bma_save_path=None, num_bins=args.num_bins, eps=args.eps)
        temperature = res["temperature"]
    else:
        pass
        print("No Code for temperature scaling on this method")

    if save:
        save_reliability_diagram(args.method, args.optim, args.save_path, res['unc'], False)
    
    return res, temperature



def bma(args, tr_loader, val_loader, te_loader, num_classes, model, mean, variance, criterion, bma_save_path, temperature=None):
    if args.no_save_bma:
        bma_save_path = None
        
    if args.method in ["swag", "ll_swag"]:
        bma_res = swag_utils.bma_swag(tr_loader, te_loader, model, num_classes, criterion, args.bma_num_models, bma_save_path, args.eps, args.batch_norm, num_bins=args.num_bins)       
    elif args.method in ["vi", "ll_vi"]:
        bma_res = vi_utils.bma_vi(val_loader, te_loader, mean, variance, model, args.method, criterion, num_classes, None, args.bma_num_models, bma_save_path, args.num_bins, args.eps)
    elif args.optim == 'bsam':
        bma_res = bsam_utils.bma_bsam(args, val_loader, te_loader, mean, variance, model, args.method, criterion, num_classes, None, args.bma_num_models,  bma_save_path, args.num_bins, args.eps)
    else:
        raise NotImplementedError("Add code for Bayesian Model Averaging with Temperature scaling for this method")
    bma_logits = bma_res["logits"]; bma_targets = bma_res["targets"]

    bma_accuracy = bma_res['accuracy']
    bma_nll = bma_res['nll']
    bma_ece = bma_res['ece']
    
    
    print(f"3) Uncalibrated BMA Results:")
    table = [["Num BMA models", "Test Accuracy", "Test NLL", "Test Ece"],
            [args.bma_num_models, format(bma_accuracy, '.4f'), format(bma_nll, '.4f'), format(bma_ece, '.4f')]]
    print(tabulate.tabulate(table, tablefmt="simple"))
    
    ## Adjust temperature scaling on bma logits
    if temperature is not None:
        bma_logits_ts = torch.tensor(bma_logits) / temperature.cpu()
        bma_predictions_ts = F.softmax(bma_logits_ts, dim=1).detach().numpy()
        
        bma_accuracy_ts = np.mean(np.argmax(bma_predictions_ts, axis=1) == bma_targets) * 100
        bma_nll_ts = -np.mean(np.log(bma_predictions_ts[np.arange(bma_predictions_ts.shape[0]), bma_targets] + args.eps))
        bma_unc_ts = calibration_curve(bma_predictions_ts, bma_targets, args.num_bins)
        bma_ece_ts = bma_unc_ts['ece']
    
    
        print(f"4) Calibrated BMA Results:")
        table = [["Num BMA models", "Test Accuracy", "Test NLL", "Test Ece", "Temperature"],
                [args.bma_num_models, format(bma_accuracy_ts, '.4f'), format(bma_nll_ts, '.4f'), format(bma_ece_ts, '.4f'), format(temperature.item(), '.4f')]]
        print(tabulate.tabulate(table, tablefmt="simple"))
    else:
        bma_accuracy_ts = 0.0
        bma_nll_ts = 0.0
        bma_ece_ts = 0.0
        temperature = torch.tensor(1.0)

    if not args.ignore_wandb:
        wandb.run.summary['bma accuracy'] = bma_accuracy
        wandb.run.summary['bma nll'] = bma_nll
        wandb.run.summary['bma ece'] = bma_ece

        if temperature is not None:
            wandb.run.summary['bma accuracy w/ ts'] = bma_accuracy_ts
            wandb.run.summary['bma nll w/ ts'] = bma_nll_ts
            wandb.run.summary['bma ece w/ ts'] = bma_ece_ts
            wandb.run.summary['bma temperature'] = temperature.item()
        
    # if not bma_save_path is not None:    
    #     save_reliability_diagram(args.method, args.optim, args.save_path, bma_res['unc'], True)
    
    return bma_res, bma_accuracy, bma_nll, bma_ece, bma_accuracy_ts, bma_nll_ts, bma_ece_ts, temperature.item()