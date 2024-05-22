import numpy as np
import torch
from ..swag.swag_utils import predict, flatten
import utils.utils as utils


def get_la_mean_vector(model):
    mean_list = []
    for param in model.parameters():
        mean_list.append(param)
    return flatten(mean_list)


def get_la_variance_vector(la):
    return la.posterior_variance

# def get_la_covariance_matrix(la, last=False, low_rank=20):
#     if last:
#         print("You should correct this")
#         cov_sqrt = torch.inverse(la.posterior_precision.to_matrix())
#         cov_sqrt = torch.svd_lowrank(cov_sqrt, q=low_rank)
#         cov_sqrt = cov_sqrt[0] * torch.sqrt(cov_sqrt[1])
        
#     else:
#         l = torch.diag(la.posterior_precision[0][1])
#         l = torch.inverse(torch.sqrt(l))
#         cov_sqrt = torch.pinverse(la.posterior_precision[0][0]).t()
#         cov_sqrt = cov_sqrt.matmul(l).t()
        
#         p_0_inv = 1 / la.posterior_precision[1]
#         p_0_inv = torch.svd_lowrank(p_0_inv, q=cov_sqrt.size(0))
#         p_0_inv = p_0_inv[0].matmul(torch.sqrt(p_0_inv[1]))
        
#         cov_sqrt += p_0_inv
        
#     return cov_sqrt



def save_la_model(args, model, la):
    mean = get_la_mean_vector(model)
    torch.save(mean,f'{args.save_path}/{args.method}-{args.optim}_best_val_mean.pt')

    variance = get_la_variance_vector(la)
    torch.save(variance, f'{args.save_path}/{args.method}-{args.optim}_best_val_variance.pt')  

    return mean, variance


def sample_la(mean, var, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    rand_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)
    sample = mean + rand_sample
    return sample
    

def bma_la(te_loader, mean, var, model, la, tr_layer="last_layer", bma_num_models=30, num_classes=10, bma_save_path=None, eps=1e-8):
    model_shape = list()
    for p in model.parameters():
        model_shape.append(p.shape)
    
    if tr_layer == "last_layer":
        model_shape = model_shape[-2:]

    elif tr_layer == "last_block":
        raise NotImplementedError("Need code for last block LA")
    else:
        pass
    
    ## in case of last layer LA (Need to fix for last block LA)
    for name, _ in model.named_modules():
        tr_layer_name = name
    
    
    bma_predictions = np.zeros((len(te_loader.dataset), num_classes))
    with torch.no_grad():
        for i in range(bma_num_models):
            if i == 0:
               sample = sample_la(mean, var, scale=0.0)
            else:
                # sample = sample_la(mean, var)
                sample = la.sample(n_samples=1)

            sample = utils.unflatten_like_size(sample, model_shape)
            sample = utils.list_to_state_dict(model, sample, tr_layer, tr_layer_name)
            model.load_state_dict(sample, strict=False)

            # save sampled weight for bma
            if bma_save_path is not None:
                torch.save(sample, f'{bma_save_path}/bma_model-{i}.pt')
 
            res = predict(te_loader, model)
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