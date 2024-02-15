import os, copy, time, tqdm, wandb

import torch

import clip.clip as clip

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing

import src.datasets as datasets

import utils.utils_img as utils
# from utils.swag import swag
# from utils.sam import sam, sam_utils
# from utils.sabma import sabma, sabma_utils
# from utils import temperature_scaling as ts

def train_sabma(args):
    pass


def finetune(args):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    assert args.train_dataset is not None, "Please provide a training dataset."
    
    
    image_classifier = ImageClassifier.load(args.load)

    if args.freeze_encoder:
        print('Fine-tuning a linear classifier')
        model = image_classifier.classification_head
        input_key = 'features'
        preprocess_fn = image_classifier.val_preprocess
        image_enc = image_classifier.image_encoder
        print_every = 1000
    else:
        print('Fine-tuning end-to-end')
        model = image_classifier
        input_key = 'images'
        preprocess_fn = image_classifier.train_preprocess
        image_enc = None
        image_classifier.process_images = True
        print_every = 100
    
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    num_batches = len(dataset.train_loader)


    swag_model = None
    if args.method == "swag":
        swag_model = utils.SWAG(copy.deepcopy(model),
                            no_cov_mat=False,
                            max_num_models=args.max_num_models,
                            last_layer=False).cuda()

    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss().cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(params,
                            lr=args.lr, weight_decay=args.wd,
                            momentum=0.9)
    elif args.optim == 'sam':
        base_optimizer = torch.optim.SGD
        optimizer = utils.SAM(params, base_optimizer, rho=args.rho, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.optim == 'sabma':
        base_optimizer = torch.optim.SGD
        optimizer = utils.SABMA_optim(model.bnn_param.values(), base_optimizer, rho=args.rho, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.optim == 'bsam':
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    
    for epoch in range(args.epochs):
        model.train()
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=image_enc)
        
        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()
            data_time = time.time() - start_time
            
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            if args.optim == 'sam':
                optimizer.first_step(zero_grad=True, amp=False)

                logits = model(inputs)
                loss = loss_fn(logits, labels)    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.second_step(zero_grad=True, amp=False)
            
            else:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
            
            batch_time = time.time() - start_time
            
            ## swag
            if (args.method == 'swag') and ((epoch + 1) > args.swa_start) and ((epoch + 1 - args.swa_start) % args.swa_c_epochs == 0):
                swag_model.collect_model(model)
                swag_model.sample(0.0)
                utils.bn_update(data_loader, swag_model, input_key)
            
            if i % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
            
            wandb.log({"Train Loss" : loss.item(),
                       "running_lr" : optimizer.param_groups[0]['lr']})    
        
        if args.freeze_encoder:
            image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
        else:
            image_classifier = model.module
        
        # Saving model
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint.pt')
            print('Saving model to', model_path)
            image_classifier.save(model_path)
            # optim_path = os.path.join(args.save, f'optim_{epoch+1}.pt')
            # torch.save(optimizer.state_dict(), optim_path)    
            
        # Evaluate
        args.current_epoch = epoch
        
        if ((args.current_epoch + 1) == args.epochs) and (args.method == "swag"):
            mean = swag_model.get_mean_vector()
            torch.save(mean, os.path.join(args.save, f'swag_mean.pt'))
            
            variance = swag_model.get_variance_vector()
            torch.save(variance, os.path.join(args.save, f'swag_variance.pt'))
            
            cov_mat = swag_model.get_covariance_matrix()
            torch.save(cov_mat, os.path.join(args.save, f'swag_covmat.pt'))
        
        
        eval_results = evaluate(image_classifier, args, swag_model)
        wandb.log(eval_results)
            
            
            
    if args.save is not None:
        return model_path    



if __name__ == '__main__':
    args = parse_arguments()
    utils.set_seed(args.seed)
    finetune(args)
