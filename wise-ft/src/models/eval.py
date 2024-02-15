import os
import json

import torch
import numpy as np

from src.models import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
import src.datasets as datasets

import utils.utils_img as utils_img

def eval_single_dataset(image_classifier, dataset, args):
    if args.freeze_encoder:
        model = image_classifier.classification_head
        input_key = 'features'
        image_enc = image_classifier.image_encoder
    else:
        model = image_classifier
        input_key = 'images'
        image_enc = None

    model.eval()
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc)
    batched_data = enumerate(dataloader)
    device = args.device

    if hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in batched_data:
            data = maybe_dictionarize(data)
            x = data[input_key].to(device)
            y = data['labels'].to(device)

            if 'image_paths' in data:
                image_paths = data['image_paths']
            
            logits = utils.get_logits(x, model)
            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y, image_paths, args)
                correct += acc1
                n += num_total
            else:
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

            if hasattr(dataset, 'post_loop_metrics'):
                all_labels.append(y.cpu().clone().detach())
                all_preds.append(logits.cpu().clone().detach())
                metadata = data['metadata'] if 'metadata' in data else image_paths
                all_metadata.extend(metadata)

        top1 = correct / n

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            metrics = dataset.post_loop_metrics(all_labels, all_preds, all_metadata, args)
            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1
    
    return metrics



def bma_single_dataset(image_classifier, swag_model, dataset, args):  
    if args.freeze_encoder:
        model = swag_model
        input_key = 'features'
        image_enc = image_classifier.image_encoder
        
    else:
        model = image_classifier
        input_key = 'images'
        image_enc = None

    model.eval()
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc)
    device = args.device
    metrics = {}
    
    num_classes = 1000
    
    
    with torch.no_grad():
        # bma_predictions = np.zeros((len(dataset.test_loader.dataset), len(dataset.test_loader.dataset.classes)))
        try:
            bma_predictions = torch.zeros((len(dataset.test_loader.dataset), len(dataset.test_loader.dataset.classes)))
        except:
            bma_predictions = torch.zeros((len(dataset.test_loader.dataset), 1000))
        for idx in range(args.bma_num_models):
            model.sample(1.0, cov=True)
            try:
                utils_img.bn_update(dataset.train_loader, model, input_key)
            except:
                pass
            
            inputs_list = list(); targets_list = list()
            for i, data in enumerate(dataloader):
                data = maybe_dictionarize(data)
                x = data[input_key].to(device)
                y = data['labels'].to(device)
                if hasattr(dataset, 'project_labels'):
                    y = dataset.project_labels(y, device) 
                
                inputs_list.append(x)
                targets_list.append(y)
            inputs = torch.concat(inputs_list)
            logits = utils.get_logits(inputs, model)
            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)   
            
            ## cumulate preds and y
            predictions = torch.nn.functional.softmax(logits, dim=1).cpu() #.cpu().numpy())
            bma_predictions += predictions
            targets = torch.concat(targets_list).cpu()
            
            # sample_acc = np.mean(np.argmax(predictions, axis=1) == targets)*100
            sample_acc = (torch.argmax(predictions, axis=1) == targets).sum().item() / targets.size(0) * 100
            print(f"Sample {idx+1}/{args.bma_num_models}. Accuracy: {sample_acc:.2f}")
            # ens_acc = np.mean(np.argmax(bma_predictions, axis=1) == targets)*100
            ens_acc = (torch.argmax(bma_predictions, axis=1) == targets).sum().item() / targets.size(0) *100
            print(f"Ensemble {idx+1}/{args.bma_num_models}. Accuracy: {ens_acc:.2f}")
            
        bma_predictions /= args.bma_num_models
            
    # bma_accuracy = np.mean(np.argmax(bma_predictions, axis=1) == targets)
    bma_accuracy = (torch.argmax(bma_predictions, axis=1) == targets).sum().item() / targets.size(0) *100
    metrics['bma_top1'] = bma_accuracy
    
    return metrics
 
    



def evaluate(image_classifier, args, swag_model):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            image_classifier.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size
        )

        results = eval_single_dataset(image_classifier, dataset, args)

        if (args.method in ["swag", "sabma"]) and ((args.current_epoch + 1) == args.epochs):
            bma_results = bma_single_dataset(image_classifier, swag_model, dataset, args)
        else:
            bma_results = None
 
        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
            if bma_results is not None:
                print(f"{dataset_name} Top-1 BMA accuracy : {bma_results['bma_top1']:.4f}")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ':' + key] = val
        if bma_results is not None:
            for key, val in bma_results.items():
                info[dataset_name + ":" + key] = val

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(info) + '\n')
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info