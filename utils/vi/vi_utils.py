import numpy as np
import torch
from ..swag.swag_utils import predict, flatten
import utils.sam.sam_utils as sam_utils
import utils.utils as utils
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
from utils import temperature_scaling as ts
import torchvision

def get_vi_mean_vector(model):
    """
    Get Mean parameters in Variational Inference model
    """
    mean_list = []
    for name, param in model.named_parameters():
        if "rho" not in name:
            mean_list.append(param.cpu())
    return flatten(mean_list)
            
            
def get_vi_variance_vector(model):
    """
    Get (Diagonal) Variance Parameters in Variatioanl Inference model
    """
    var_list = []
    for name, param in model.named_parameters():
        if "rho" in name:            
            var_list.append(torch.log(1+torch.exp(param.cpu())))  # rho to variance
        elif ("mu" not in name) and ("rho" not in name):
            var_list.append(torch.zeros_like(param.cpu()))
    return flatten(var_list)


def make_ll_vi(args, model):
    from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
    if 'resnet' in args.model:
        bayesian_last_layer = torch.nn.Sequential(list(model.children())[-1])
    else:
        bayesian_last_layer = model.heads.head
    const_bnn_prior_parameters = {
        "prior_mu": args.vi_prior_mu,
        "prior_sigma": args.vi_prior_sigma,
        "posterior_mu_init": args.vi_posterior_mu_init,
        "posterior_rho_init": args.vi_posterior_rho_init,
        "type": args.vi_type,
        "moped_enable": True,
        "moped_delta": args.vi_moped_delta,
    }
    dnn_to_bnn(bayesian_last_layer, const_bnn_prior_parameters)
    if 'resnet' in args.model:
        model.fc = bayesian_last_layer.to(args.device)
    elif 'vitb16-i1k' == args.model:
        model.heads.head = bayesian_last_layer.to(args.device)
    else:
        raise NotImplementedError()


def dnn_to_bnn_vit16i1k(model, const_bnn_prior_parameters):
    from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, bnn_linear_layer, bnn_conv_layer
    
    ## modified dnn_to_bnn() [https://github.com/IntelLabs/bayesian-torch/blob/main/bayesian_torch/models/dnn_to_bnn.py#L127]
    for name, value in list(model._modules.items()):
        if model._modules[name]._modules:
            dnn_to_bnn_vit16i1k(model._modules[name], const_bnn_prior_parameters)
        elif "Conv" in model._modules[name].__class__.__name__:
            setattr(
                model,
                name,
                bnn_conv_layer(
                    const_bnn_prior_parameters,
                    model._modules[name]))
        elif "Linear" in model._modules[name].__class__.__name__:
            if "NonDynamicallyQuantizableLinear" in model._modules[name].__class__.__name__:
                pass
            else:
                setattr(
                    model,
                    name,
                    bnn_linear_layer(
                        const_bnn_prior_parameters,
                        model._modules[name]))
        else:
            pass



def load_vi(model, checkpoint):   
    import collections
    ## load only bn (non-dnn) params
    st_dict = collections.OrderedDict()
    for name in checkpoint["state_dict"].copy():
        if not ("mean" in name) or not ("rho" in name):
            st_dict[name] = checkpoint["state_dict"][name]
    model.load_state_dict(st_dict, strict=False)   


def save_best_vi_model(args, best_epoch, model, optimizer, scaler, first_step_scaler, second_step_scaler):
    utils.save_best_dnn_model(args, best_epoch, model, optimizer, scaler, first_step_scaler, second_step_scaler)
    
    mean = get_vi_mean_vector(model)
    torch.save(mean,f'{args.save_path}/{args.method}-{args.optim}_best_val_mean.pt')
    
    variance = get_vi_variance_vector(model)
    torch.save(variance, f'{args.save_path}/{args.method}-{args.optim}_best_val_variance.pt')
    
    return mean, variance


# train variational inference
def train_vi_sgd(dataloader, model, criterion, optimizer, device, scaler, batch_size, kl_beta=1.0):
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
                kl = get_kl_loss(model)
                loss = criterion(pred, y)
                loss += kl_beta * kl / batch_size
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)  # optimizer.step()
                scaler.update()
                optimizer.zero_grad()
        else:
            pred = model(X)
            kl = get_kl_loss(model)
            loss = criterion(pred, y)
            loss += kl_beta * kl/batch_size
            
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


def train_vi_sam(dataloader, model, criterion, optimizer, device, first_step_scaler, second_step_scaler, batch_size, kl_beta=1.0):
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
            with torch.cuda.amp.autocast():
                pred = model(X)
                kl = get_kl_loss(model)
                loss = criterion(pred, y)
                loss += kl_beta * kl / batch_size
                
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
                
            with torch.cuda.amp.autocast():
                pred = model(X)
                kl = get_kl_loss(model)
                loss = criterion(pred, y)
                loss += kl_beta * kl / batch_size
            second_step_scaler.scale(loss).backward()
            if sam_first_step_applied:
                optimizer.second_step()
            second_step_scaler.step(optimizer)
            second_step_scaler.update()
        else:
            pred = model(X)
            kl = get_kl_loss(model)
            loss = criterion(pred, y)
            loss += kl_beta * kl/batch_size
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.first_step(zero_grad=True, amp=False)
            
            pred = model(X)
            kl = get_kl_loss(model)
            loss = criterion(pred, y)
            loss += kl_beta * kl/batch_size
            loss.backward()
            optimizer.second_step(zero_grad=True, amp=False)
            
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss_sum += loss.data.item() * X.size(0)
        num_objects_current += X.size(0)
    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": correct / num_objects_current * 100.0,
    }



def eval_vi(val_loader, model, num_classes, criterion, val_mc_num, num_bins=15, eps=1e-8):    
    mc_predictions = np.zeros((len(val_loader.dataset), num_classes))
    model.eval()
    with torch.no_grad():
        if val_mc_num == 1:
            res = predict(val_loader, model)
            predictions = res["predictions"]; targets = res["targets"]
            loss = criterion(torch.tensor(predictions), torch.tensor(targets)).item()
            accuracy = np.mean(np.argmax(predictions, axis=1) == targets)
            nll = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), targets] + eps))
            ece = utils.calibration_curve(predictions, targets, num_bins)['ece']
            
        else:
            for i in range(val_mc_num):
                res = predict(val_loader, model)
                mc_predictions += res["predictions"]
            mc_predictions /= val_mc_num

            loss = criterion(torch.tensor(mc_predictions), torch.tensor(res['targets'])).item()
            accuracy = np.mean(np.argmax(mc_predictions, axis=1) == res["targets"])
            nll = -np.mean(np.log(mc_predictions[np.arange(mc_predictions.shape[0]), res["targets"]] + eps))
            ece = utils.calibration_curve(mc_predictions, res["targets"], num_bins)['ece']
            
            predictions = mc_predictions
            targets = res["targets"]
            
    return {
        "predictions" : predictions,
        "targets" : targets,
        "loss" : loss, # loss_sum / num_objects_total,
        "accuracy" : accuracy * 100.0,
        "nll" : nll,
        "ece" : ece,
    }
    
    
    
def bma_vi(val_loader, te_loader, mean, variance, model, method, criterion, num_classes, temperature=None, bma_num_models=30,  bma_save_path=None, num_bins=15, eps=1e-8):
    '''
    run bayesian model averaging
    '''
    ## Check whether it's last layer or not
    model_shape = list()
    for p in model.parameters():
        model_shape.append(p.shape)
        
    if "ll_" in method:
        tr_layer = "last_layer"
        model_shape = model_shape[-2:]
        for name, _ in model.named_modules():
            tr_layer_name = name
    else:
        tr_layer = "full_layer"
        tr_layer_name = None
    
        
    bma_logits = np.zeros((len(te_loader.dataset), num_classes))
    bma_predictions = np.zeros((len(te_loader.dataset), num_classes))
    with torch.no_grad():
        for i in range(bma_num_models):
            if i == 0:
               sample = mean
            else:
                sample = mean + variance * torch.randn_like(variance, requires_grad=False)
            sample = utils.unflatten_like_size(sample, model_shape)
            sample = utils.list_to_state_dict(model, sample, tr_layer, tr_layer_name)
            model.load_state_dict(sample, strict=False)
            
            if temperature == 'local':
                scaled_model = ts.ModelWithTemperature(model)
                scaled_model.set_temperature(val_loader)
                temperature_ = scaled_model.temperature
            else:
                temperature_ = temperature
            
            if not bma_num_models == 1:
                # save sampled weight for bma
                if bma_save_path is not None:
                    torch.save(model, f'{bma_save_path}/bma_model-{i}.pt')
 
            res = predict(te_loader, model)
            logits = res["logits"]; predictions = res["predictions"];targets = res["targets"]

            
            accuracy = np.mean(np.argmax(predictions, axis=1) == targets)
            nll = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), targets] + eps))
            if not bma_num_models == 1:
                print(
                    "Sample %d/%d. Accuracy: %.2f%% NLL: %.4f"
                    % (i + 1, bma_num_models, accuracy * 100, nll)
                )

            bma_logits += logits
            bma_predictions += predictions

            ens_accuracy = np.mean(np.argmax(bma_predictions, axis=1) == targets)
            ens_nll = -np.mean(
                np.log(
                    bma_predictions[np.arange(bma_predictions.shape[0]), targets] / (i + 1)
                    + eps
                )
            )
            if not bma_num_models == 1:
                print(
                    "Ensemble %d/%d. Accuracy: %.2f%% NLL: %.4f"
                    % (i + 1, bma_num_models, ens_accuracy * 100, ens_nll)
                )

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
        "loss" : bma_loss,
        "accuracy" : bma_accuracy * 100,
        "nll" : bma_nll,
        "unc" : bma_unc,
        "ece" : bma_unc['ece'],
        "temperature" : temperature_,
        }