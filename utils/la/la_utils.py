import numpy as np
import torch
from ..swag.swag_utils import predict, flatten
import utils.utils as utils


def get_la_mean_vector(model, last=False):
    mean_list = []
    if last:
        for name, _ in model.named_modules():
            last_layer_name = name
        
        for name, param in model.named_parameters():
            if name.split('.')[0] == last_layer_name:
                mean_list.append(param)
    else:
        for param in model.parameters():
            mean_list.append(param.cpu())
        
    return flatten(mean_list)


def get_la_covariance_matrix(la, last=False, low_rank=20):
    if last:
        print("You should correct this")
        cov_sqrt = torch.inverse(la.posterior_precision.to_matrix())
        cov_sqrt = torch.svd_lowrank(cov_sqrt, q=low_rank)
        cov_sqrt = cov_sqrt[0] * torch.sqrt(cov_sqrt[1])
        
    else:
        l = torch.diag(la.posterior_precision[0][1])
        l = torch.inverse(torch.sqrt(l))
        cov_sqrt = torch.pinverse(la.posterior_precision[0][0]).t()
        cov_sqrt = cov_sqrt.matmul(l).t()
        
        p_0_inv = 1 / la.posterior_precision[1]
        p_0_inv = torch.svd_lowrank(p_0_inv, q=cov_sqrt.size(0))
        p_0_inv = p_0_inv[0].matmul(torch.sqrt(p_0_inv[1]))
        
        cov_sqrt += p_0_inv
        
    return cov_sqrt



def save_la_model(args, model, la):
    mean = get_la_mean_vector(model, last=args.linear_probe)
    torch.save(mean,f'{args.save_path}/{args.method}-{args.optim}_best_val_mean.pt')

    cov_sqrt = get_la_covariance_matrix(la, last=args.linear_probe, low_rank=args.low_rank)
    torch.save(cov_sqrt, f'{args.save_path}/{args.method}-{args.optim}_best_val_covmat.pt')  

    return mean, cov_sqrt



def sample_la(mean, cov_sqrt, scale=1.0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    eps = cov_sqrt.new_empty((cov_sqrt.size(0),), requires_grad=False).normal_()
    cov_sample = cov_sqrt.t().matmul(eps).cpu()
    cov_sample /= (cov_sqrt.size(0) - 1) ** 0.5
    
    sample = mean + scale**0.5 * cov_sample
    
    return sample
    
    
    

def bma_la(te_loader, mean, cov_sqrt, model, method, bma_num_models, num_classes, bma_save_path=None, eps=1e-8):
    model_shape = list()
    for p in model.parameters():
        model_shape.append(p.shape)
    
    if "last" in method:
        last = True
        model_shape = model_shape[-2:]
    else:
        last = False
        
    for name, _ in model.named_modules():
        last_layer_name = name
    
    bma_predictions = np.zeros((len(te_loader.dataset), num_classes))
    with torch.no_grad():
        for i in range(bma_num_models):
            if i == 0:
               sample = sample_la(mean, cov_sqrt, scale=0.0)
            else:
                sample = sample_la(mean, cov_sqrt)

            sample = utils.unflatten_like_size(sample, model_shape)
            sample = utils.list_to_state_dict(model, sample, last, last_layer_name)
            model.load_state_dict(sample, strict=False)
            
            # save sampled weight for bma
            if bma_save_path is not None:
                torch.save(sample, f'{bma_save_path}/bma_model-{i}.pt')
 
            res = predict(te_loader, model, verbose=False)
            predictions = res["predictions"];targets = res["targets"]

            accuracy = np.mean(np.argmax(predictions, axis=1) == targets)
            nll = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), targets] + eps))
            print(
                "Sample %d/%d. Accuracy: %.2f%% NLL: %.4f"
                % (i + 1, bma_num_models, accuracy * 100, nll)
            )

            bma_predictions += predictions

            ens_accuracy = np.mean(np.argmax(bma_predictions, axis=1) == targets)
            ens_nll = -np.mean(
                np.log(
                    bma_predictions[np.arange(bma_predictions.shape[0]), targets] / (i + 1)
                    + eps
                )
            )
            print(
                "Ensemble %d/%d. Accuracy: %.2f%% NLL: %.4f"
                % (i + 1, bma_num_models, ens_accuracy * 100, ens_nll)
            )

        bma_predictions /= bma_num_models

        bma_accuracy = np.mean(np.argmax(bma_predictions, axis=1) == targets)
        bma_nll = -np.mean(
            np.log(bma_predictions[np.arange(bma_predictions.shape[0]), targets] + eps)
        )
    
    print(f"bma Accuracy using {bma_num_models} model : {bma_accuracy * 100:.2f}% / NLL : {bma_nll:.4f}")
    return {"predictions" : bma_predictions,
            "targets" : targets,
            "bma_accuracy" : bma_accuracy,
            "nll" : bma_nll
    }