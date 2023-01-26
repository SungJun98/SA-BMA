import os, sys
import time, copy

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import pickle

import utils, data #, losses

from baselines.sam.sam import SAM, FSAM
from baselines.swag import swag, swag_utils

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tr_loader, val_loader, te_loader, num_classes = data.get_cifar10('./data/cifar10/', 256,4,use_validation = True)


from models import wide_resnet
model_cfg = getattr(wide_resnet, "WideResNet40x10")
model = model_cfg.base(num_classes=num_classes).to(device)


bma_model_path  = ''
bma_models = os.listdir(bma_model_path)



swag_predictions = np.zeros((len(te_loader.dataset), num_classes))
with torch.no_grad():
    for i, bma_model in enumerate(range(len(bma_models))):
        model.load_state_dict(torch.load('best_model path'))
        model_state_dict = model.state_dict()

        # get sampled model
        bma_sample = torch.load(f"{bma_model_path}/{bma_model}")
        bma_state_dict = utils.list_to_state_dict(model, bma_sample)

        model_state_dict.update(bma_state_dict)
        model.load_state_dict(model_state_dict)

        res = swag_utils.predict(te_loader, model, verbose=False)

        predictions = res["predictions"]
        targets = res["targets"]

        accuracy = np.mean(np.argmax(predictions, axis=1) == targets)
        nll = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), targets] + 1e-8))
        print(
            "SWAG Sample %d/%d. Accuracy: %.2f%% NLL: %.4f"
            % (i + 1, len(bma_models), accuracy * 100, nll)
        )

        swag_predictions += predictions

        ens_accuracy = np.mean(np.argmax(swag_predictions, axis=1) == targets)
        ens_nll = -np.mean(
            np.log(
                swag_predictions[np.arange(swag_predictions.shape[0]), targets] / (i + 1)
                + 1e-8
            )
        )
        print(
            "Ensemble %d/%d. Accuracy: %.2f%% NLL: %.4f"
            % (i + 1, len(bma_models), ens_accuracy * 100, ens_nll)
        )

    swag_predictions /= len(bma_models)

    swag_accuracy = np.mean(np.argmax(swag_predictions, axis=1) == targets)
    swag_nll = -np.mean(
        np.log(swag_predictions[np.arange(swag_predictions.shape[0]), targets] + 1e-8)
    )

print(f"bma Accuracy using {len(bma_models)} models : {swag_accuracy * 100:.2f}% / NLL : {swag_nll:.4f}")