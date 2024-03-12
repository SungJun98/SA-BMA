import numpy as np
import torch
from ..swag.swag_utils import predict, flatten
import utils.sam.sam_utils as sam_utils
import utils.utils as utils
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
from utils import temperature_scaling as ts

def bma_bsam(val_loader, te_loader, mean, variance, model, method, criterion, num_classes, temperature=None, bma_num_models=32,  bma_save_path=None, num_bins=15, eps=1e-8):
    '''
    run bayesian model averaging in test step
    '''
    ## Check whether it's last layer or not
    model_shape = list()
    for p in model.parameters():
        model_shape.append(p.shape)
        
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
                    if temperature == 'local':
                        torch.save(scaled_model, f'{bma_save_path}/bma_model-{i}.pt')
                    else:
                        torch.save(model, f'{bma_save_path}/bma_model-{i}.pt')
 
            res = predict(te_loader, model, temperature_)
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
            "temperature" : temperature_
    }    
