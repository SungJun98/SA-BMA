import numpy as np
import os.path as osp
import datetime, time, copy
from tqdm import tqdm
from collections import OrderedDict

import wandb
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.nn.functional as F

from my_dassl.modeling import build_backbone
from my_dassl.optim import build_optimizer, build_lr_scheduler
from my_dassl.engine import TRAINER_REGISTRY, TrainerX
from my_dassl.utils import (MetricMeter, AverageMeter, set_random_seed, count_num_param, save_checkpoint)
from my_dassl.metrics import compute_accuracy

import utils_swag as swag
import utils_sabma as sabma
import utils_vi as vi

class ResNet50(nn.Module):
    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )        

        self._fdim = self.backbone.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(self._fdim, num_classes)
        
    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        y = self.classifier(f)
        if return_feature:
            return y, f
        else:
            return y



class ConvNextV2(nn.Module):
    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )        
        ## convnext v2 -----------------------------
        if self.backbone.out_features is None:
            fdim = self.backbone.head.in_features
        else:
            fdim = self.backbone.out_features
        self.backbone.head = nn.Linear(fdim, num_classes)
        ## -----------------------------------------
        
        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x):
        y = self.backbone(x)
        return y




@TRAINER_REGISTRY.register()
class CONVNET(TrainerX):    
    def __init__(self, cfg):
        super().__init__(cfg)


    def build_model(self):
        cfg = self.cfg

        print("Building model")
        if cfg.MODEL.BACKBONE.NAME == 'convnextv2_tiny':
            self.model = ConvNextV2(cfg, cfg.MODEL, self.num_classes)
        elif cfg.MODEL.BACKBONE.NAME == 'resnet50':
            self.model = ResNet50(cfg, cfg.MODEL, self.num_classes)
        else:
            raise "Please add backbone"
        self.model.to(self.device)
        self.swag_model = None
        if self.cfg.METHOD == "swag":
            self.swag_model = swag.SWAG(copy.deepcopy(self.model),
                                no_cov_mat=self.cfg.SWAG.DIAG_ONLY,
                                max_num_models=self.cfg.SWAG.MAX_NUM_MODELS,
                                last_layer=False).to(self.device)
            print("Successfully set the swag model")
            
        elif self.cfg.METHOD == 'vi':
            from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
            const_bnn_prior_parameters = {
                "prior_mu": 0.0,
                "prior_sigma": 1.0,
                "posterior_mu_init": 0.0,
                "posterior_rho_init": -3.0,
                "type": 'Reparameterization',
                "moped_enable": True,
                "moped_delta": cfg.VI.MOPED_DELTA,
            }
            dnn_to_bnn(self.model, const_bnn_prior_parameters)
            self.model.to(self.device)
            print(f"Preparing Model for VI")


        elif self.cfg.METHOD == 'sabma':
            w_mean = torch.load(osp.join(self.cfg.SABMA.PRIOR_PATH, f'{self.cfg.MODEL.BACKBONE.NAME}_mean.pt'))
            w_var = torch.load(osp.join(self.cfg.SABMA.PRIOR_PATH, f'{self.cfg.MODEL.BACKBONE.NAME}_variance.pt'))
            if not self.cfg.SABMA.DIAG_ONLY:
                w_cov_sqrt = torch.load(osp.join(self.cfg.SABMA.PRIOR_PATH, f'{self.cfg.MODEL.BACKBONE.NAME}_covmat.pt'))
            else:
                w_cov_sqrt = None
                
            self.sabma_model = sabma.SABMA(copy.deepcopy(self.model),
                                cfg=cfg,
                                w_mean = w_mean,
                                w_var = w_var,
                                w_cov_sqrt = w_cov_sqrt,
                                ).to(self.device)

        print(f"# params: {count_num_param(self.model):,}")
        if self.cfg.METHOD == 'sabma':
            base_optimizer = torch.optim.SGD
            self.optim = sabma.SABMA_optim(self.sabma_model.bnn_param.values(), base_optimizer,
                                    rho=self.cfg.OPTIM.RHO, lr=self.cfg.OPTIM.LR, momentum=self.cfg.OPTIM.MOMENTUM,
                                    weight_decay=self.cfg.OPTIM.WEIGHT_DECAY)
        else:
            self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)
            
        self.register_model(name="model", model=self.model, optim=self.optim, sched=self.sched)

    def train(self):
        """Generic training loops."""
        self.before_train()
        set_random_seed(self.cfg.SEED)
        print(f"Start training with {self.cfg.METHOD}-{self.cfg.OPTIM.NAME}")
        self.best_epoch = 0; self.tolerance = 0
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
            
            # ### REMOVE ###############
            # if (self.epoch + 1) == 130 and self.cfg.METHOD == 'swag':
            #     self.test(method='swag')
            # ##########################
            
            if self.tolerance == self.cfg.TOLERANCE:
                break
            
        self.after_train() 


    def before_epoch(self):
        pass

    
    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)
        self.total_length = self.num_batches * self.max_epoch
        self.warmup_length = self.total_length * 0.1
        end = time.time()

        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            
            if self.cfg.METHOD in ['dnn', 'vi'] and self.cfg.OPTIM.NAME == 'sgd':
                loss_summary = self.forward_backward(batch)
            elif self.cfg.METHOD == 'dnn' and self.cfg.OPTIM.NAME == 'sam':
                loss_summary = self.forward_backward_sam(batch)
            elif self.cfg.METHOD == 'swag' and self.cfg.OPTIM.NAME == 'sgd':
                loss_summary = self.forward_backward_swag(batch)
            elif self.cfg.METHOD == 'sabma' and self.cfg.OPTIM.NAME == 'sabma':
                loss_summary = self.forward_backward_sabma(batch)
            
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)
            end = time.time()



    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label


    ## DNN-SGD / VI-SGD
    def forward_backward(self, batch):
        self.optim.zero_grad()
        ## Training Phase
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        if self.cfg.METHOD == 'vi':
            from bayesian_torch.models.dnn_to_bnn import get_kl_loss
            kl = get_kl_loss(self.model)
            loss += self.cfg.VI.KL_BETA * kl / len(label)
        
        loss.backward()
        self.optim.step()

        loss_summary = {
            "loss": loss.item(),
            "acc_train": compute_accuracy(output, label)[0].item(),
        }

        if self.cfg.use_wandb:
            wandb.log({'tr_loss (batch)': loss_summary["loss"],
                    'tr_acc (batch)' : loss_summary["acc_train"]})

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    
    ## DNN-SAM
    def forward_backward_sam(self, batch):
        self.optim.zero_grad()
        ## Training Phase
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        
        loss.backward()
        self.optim.first_step(zero_grad=True)
        
        # second step
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(params, 1.0)
        self.optim.second_step(zero_grad=True)

        loss_summary = {
            "loss": loss.item(),
            "acc_train": compute_accuracy(output, label)[0].item(),
        }

        if self.cfg.use_wandb:
            wandb.log({'tr_loss (batch)': loss_summary["loss"],
                    'tr_acc (batch)' : loss_summary["acc_train"]})

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        # torch.cuda.empty_cache()
        
        return loss_summary


    ## SWAG-SGD
    def forward_backward_swag(self, batch):
        self.optim.zero_grad()
        ## Training Phase
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        
        loss.backward()
        self.optim.step()
        
        if ((self.epoch + 1) >= self.cfg.SWAG.SWA_START) and ((self.epoch + 1 - self.cfg.SWAG.SWA_START) % self.cfg.SWAG.SWA_C_EPOCHS == 0):
            self.swag_model.collect_model(self.model)
            self.swag_model.sample(0.0, cov=(not self.cfg.SWAG.DIAG_ONLY))
            swag.bn_update(self.train_loader_x, self.swag_model) 

        loss_summary = {
            "loss": loss.item(),
            "acc_train": compute_accuracy(output, label)[0].item(),
        }
        if self.cfg.use_wandb:
            wandb.log({'tr_loss (batch)': loss_summary["loss"],
                    'tr_acc (batch)' : loss_summary["acc_train"]})
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
    
        return loss_summary

    ## SABMA
    def forward_backward_sabma(self, batch):
        self.optim.zero_grad()
        ## Training Phase
        image, label = self.parse_batch_train(batch)
        
        ### First forward-backward
        # Sample weight
        tr_params, z_1, z_2 = self.sabma_model.sample(z_scale = 1.0, sample_param='tr')
        frz_params, _, _ = self.sabma_model.sample(z_scale = 1.0, sample_param='frz')
    
        # compute log probability and gradient of log probability w.r.t. model parameters
        _, log_grad = self.sabma_model.log_grad(tr_params)

        # Change weight sample shape to input model
        params_ = sabma.format_weights(tr_params, frz_params, self.sabma_model)

        # first forward & backward
        output = self.sabma_model(params_, image)
        loss = F.cross_entropy(output, label)
        loss.backward()
        self.optim.first_step(log_grad, zero_grad=True)

        # second step
        tr_params = self.optim.second_sample(z_1, z_2, self.sabma_model)
        params_ = sabma.format_weights(tr_params, frz_params, self.sabma_model)
        output = self.sabma_model(params_, image)
        loss = F.cross_entropy(output, label)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.sabma_model.bnn_param.values(), 1.0)
        self.optim.second_step(zero_grad=True)
        
        loss_summary = {
            "loss": loss.item(),
            "acc_train": compute_accuracy(output, label)[0].item(),
        }

        if self.cfg.use_wandb:
            wandb.log({'tr_loss (batch)': loss_summary["loss"],
                    'tr_acc (batch)' : loss_summary["acc_train"]})
            

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        torch.cuda.empty_cache()
        return loss_summary



    def after_train(self):
        print("Finish training")
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
        self.close_writer()



    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )
        
        ## dnn validation and save best model
        if self.cfg.METHOD == 'dnn':
            curr_result = 0.0
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                curr_result = self.test(method='dnn', split="val")
                is_best = curr_result > self.best_result                
                if is_best:
                    self.best_result = curr_result
                    self.best_epoch = self.epoch
                    self.save_model(
                        self.epoch,
                        self.output_dir,
                        model_name="model-best.pth.tar"
                    )
                    self.tolerance = 0
                else:
                    self.tolerance += 1
                    
            else:
                if meet_checkpoint_freq or last_epoch:
                    self.save_model(self.epoch, self.output_dir)
        
        ## swag validation and save best model
        elif self.cfg.METHOD == 'swag':            
            curr_result = self.test(method='dnn', split="val")
            self.best_epoch = 0
            if (self.epoch + 1) >= self.cfg.SWAG.SWA_START:
                curr_result_swag = self.test(method='swag', split="val")
                is_best = curr_result_swag > self.best_result
                if is_best:
                    self.best_result = curr_result_swag
                    self.best_epoch = self.epoch
                    self.save_model(
                        self.epoch,
                        self.output_dir,
                        model_name='model-best.pth.tar'
                    )
                    self.tolerance = 0
                else:
                    self.tolerance +=1

                if self.cfg.use_wandb:
                    wandb.log({'swag_val_acc':curr_result_swag})
            
        ## vi validation and save best model
        elif self.cfg.METHOD == 'vi':
            curr_result = 0.0
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                curr_result = self.test(method='vi', split='val')
                is_best = curr_result > self.best_result                
                if is_best:
                    self.best_result = curr_result
                    self.best_epoch = self.epoch
                    self.save_model(
                        self.epoch,
                        self.output_dir,
                        model_name="model-best.pth.tar"
                    )
                    self.tolerance = 0
                else:
                    self.tolerance += 1
                    
            else:
                if meet_checkpoint_freq or last_epoch:
                    self.save_model(self.epoch, self.output_dir)
           
        ## sabma validation and save best model
        elif self.cfg.METHOD == 'sabma':
            curr_result = 0.0
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                curr_result = self.bma(split="val", bma_num_models=3)
                is_best = curr_result > self.best_result                
                if is_best:
                    self.best_result = curr_result
                    self.best_epoch = self.epoch
                    self.save_model(
                        self.epoch,
                        self.output_dir,
                        model_name="model-best.pth.tar"
                    )
                    self.tolerance = 0
                else:
                    self.tolerance += 1
                    
            else:
                if meet_checkpoint_freq or last_epoch:
                    self.save_model(self.epoch, self.output_dir)

        if self.cfg.use_wandb:
            wandb.log({'val_acc':curr_result})
            

    @torch.no_grad()
    def test(self, method, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            if method in ['dnn', 'vi']:
                output = self.model_inference(input)
            elif method == 'swag':    
                self.swag_model.sample(0.0, cov=(not self.cfg.SWAG.DIAG_ONLY))
                output = self.swag_model(input)
                
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        if split == "test":
            if self.cfg.use_wandb:
                wandb.run.summary['test acc'] = results['accuracy']
                wandb.run.summary['test error rate'] = results['error_rate']
                wandb.run.summary['test macro f1'] = results['macro_f1']
                wandb.run.summary['best epoch'] = self.best_epoch
                

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
        
        
    def bma(self, split="test", bma_num_models=30):
        # self.set_model_mode("eval")
        results = dict()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        with torch.no_grad():
            bma_predictions = torch.zeros(len(data_loader.dataset), self.num_classes)
            for idx in range(bma_num_models):
                # sample model
                if self.cfg.METHOD == "swag":
                    self.swag_model.sample(1.0, cov=(not self.cfg.SWAG.DIAG_ONLY))
                    swag.bn_update(self.train_loader_x, self.swag_model)    
                elif self.cfg.METHOD == "sabma":
                    # Sample weight
                    tr_params, z_1, z_2 = self.sabma_model.sample(z_scale = 1.0, sample_param='tr')
                    frz_params, _, _ = self.sabma_model.sample(z_scale = 1.0, sample_param='frz')
                    params_ = sabma.format_weights(tr_params, frz_params, self.sabma_model)
                
                # run bma
                pred_list = list(); label_list = list()
                for batch_idx, batch in enumerate(tqdm(data_loader)):
                    input, label = self.parse_batch_test(batch)
                    
                    # forward
                    if self.cfg.METHOD == "swag":
                        output = self.swag_model(input)
                    elif self.cfg.METHOD == "vi":
                        output = self.model(input)
                    elif self.cfg.METHOD == "sabma":
                        output = self.sabma_model(params_, input)
                    
                    pred = torch.nn.functional.softmax(output, dim=1).cpu()
                    pred_list.append(pred)
                    label_list.append(label)
                predictions = torch.concat(pred_list).cpu()
                bma_predictions += predictions
                labels = torch.concat(label_list).cpu()
                
                sample_acc = (torch.argmax(predictions, axis=1) == labels).sum().item() / labels.size(0) * 100
                if split == "test":
                    print(f"Sample {idx+1}/{bma_num_models}. Accuracy: {sample_acc:.2f}")
                
                ens_acc = (torch.argmax(bma_predictions, axis=1) == labels).sum().item() / labels.size(0) * 100
                if split == "test":
                    print(f"Ensemble {idx+1}/{bma_num_models}. Accuracy: {ens_acc:.2f}")
                
            bma_predictions /= bma_num_models
        bma_accuracy = (torch.argmax(bma_predictions, axis=1) == labels).sum().item() / labels.size(0) * 100
        
        results["bma_accuracy"] = bma_accuracy
        
        if split == "test":
            if self.cfg.use_wandb:
                wandb.run.summary['test bma acc'] = results['bma_accuracy']
                wandb.run.summary['best epoch'] = self.best_epoch
        else:
            print(f"Valid Acc : {bma_accuracy:.4f}")
            

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
            
            
            
    def save_model(self, epoch, directory, is_best=False, model_name=""):

        names = self.get_model_names()
        for name in names:
            model_dict = self._models[name].state_dict()
            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )
        
        
        if self.cfg.METHOD == 'swag' and (self.epoch+1 > self.cfg.SWAG.SWA_START):
            torch.save(self.swag_model.state_dict(), osp.join(self.cfg.OUTPUT_DIR, 'swag_model.pt'))
            mean = self.swag_model.get_mean_vector();torch.save(mean, osp.join(self.cfg.OUTPUT_DIR, f'swag_mean.pt'))
            variance = self.swag_model.get_variance_vector();torch.save(variance, osp.join(self.cfg.OUTPUT_DIR, f'swag_variance.pt'))
            if not self.cfg.SWAG.DIAG_ONLY:
                cov_mat = self.swag_model.get_covariance_matrix();torch.save(cov_mat, osp.join(self.cfg.OUTPUT_DIR, f'swag_covmat.pt'))
        
        elif self.cfg.METHOD == 'vi':
            torch.save(self.model.state_dict(), osp.join(self.cfg.OUTPUT_DIR, 'vi_model.pt'))
            mean = vi.get_vi_mean_vector(self.model);torch.save(mean, osp.join(self.cfg.OUTPUT_DIR, 'vi_mean.pt'))
            variance = vi.get_vi_variance_vector(self.model);torch.save(mean, osp.join(self.cfg.OUTPUT_DIR, 'vi_variance.pt'))
            
        elif self.cfg.METHOD == 'sabma':
            torch.save(self.sabma_model, osp.join(self.cfg.OUTPUT_DIR, 'sabma_model.pt'))
        
        
        
    def load_swag_model(self, checkpoint):
        import pdb;pdb.set_trace()
        checkpoint = torch.load(checkpoint)
        self.swag_model.load_state_dict(checkpoint)
        print("Successfully load swag model")
        
        
    def load_sabma_model(self, checkpoint):
        self.sabma_model = torch.load(checkpoint)
        # self.sabma_model.load_state_dict(checkpoint)
        print("Successfully load sabma model")
        
    # def load_vi_model(self, checkpoint):
    #     import collections
    #     ## load only bn (non-dnn) params
    #     st_dict = collections.OrderedDict()
    #     for name in checkpoint["state_dict"].copy():
    #         if not ("mean" in name) or not ("rho" in name):
    #             st_dict[name] = checkpoint["state_dict"][name]
    #     self.model.load_state_dict(st_dict, strict=False)   