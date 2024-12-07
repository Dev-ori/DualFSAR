import torch
import copy
import click
from apps.validation import validation
from apps.test import test
from utils.ddp import reduce_loss
from utils.logger import logging_result
from datasets.sample_datasets import SampleDataset
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from timm.models import create_model
from optimizer import create_optimizer
from optimizer.scheduler import CosineAnnealingWarmRestartsWitchWarmUp as CosineAnnealing
from loss import create_criterion
import models
from utils.save import save_model, auto_load_model, load_model
from utils.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from utils.metircs import MetricList
from torcheval.metrics import MeanSquaredError, R2Score
import torch.distributed as dist
import os
from utils.logger import Logger, logging_result_in_wandb
from timm.utils import ModelEma
from datasets import build_dataset
from utils.visualize import visualize

def train(local_rank, global_rank, args):
    args.output_dir = os.path.join('outputs', args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(print_freq=args.print_freq, epoch=args.epoch, global_rank=global_rank, save_dir=args.output_dir)

    is_wandb = args.wandb and global_rank == 0
    if is_wandb:
        wandb.init(
            dir=args.output_dir,
            project=args.project,
            name=args.run_name,
            config=args,
            resume=args.wandb_resume, 
            id = args.wandb_project_id
        )

    # model = ToyModel().to(local_rank)
    model = create_model(args.model, pretrained=False,)
    model = model.to(local_rank)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.log(logger.INFO, "Model = %s" % str(model))
    logger.log(logger.INFO, "num of parameters: {0}".format(n_parameters))
        
    if is_wandb:
        wandb.config.update({"num of parameters" : n_parameters})

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
    
    if args.loss_scaler:    
        loss_scaler = NativeScaler()
    else:
        loss_scaler = None
        
    # build optimizer
    optimizer = create_optimizer(model.parameters(), args)
    
    auto_load_model(args, model, model_ema, optimizer, loss_scaler, logger = logger)
    
    if args.T is None:
        args.T = args.epoch
        
    scheduler = CosineAnnealing(optimizer=optimizer, 
                                T_0=args.T,
                                T_mult=args.T_mult,
                                eta_min=args.min_lr,
                                warmup_start_value=args.warmup_lr,
                                warmup_epoch=args.warmup_epochs)
    
    # data distribution
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = DDP(model, device_ids=[local_rank])    
    
    # build dataset
    (dataset_train, dataloader_train), train_cfg = build_dataset(cfg=args.cfg_dataset, mode='train', global_rank=global_rank, world_size=args.world_size)
    (dataset_valid, dataloader_valid), valid_cfg = build_dataset(cfg=args.cfg_dataset, mode='val', global_rank=global_rank, world_size=args.world_size)
    if args.test:
        (dataset_test, dataloader_test), test_cfg = build_dataset(cfg=args.cfg_dataset, mode='test', global_rank=global_rank, world_size=args.world_size)
    
    # build Loss function
    criterion = create_criterion(args)
    
    # https://pytorch.org/torcheval/stable/torcheval.metrics.html
    # confusion matrix's key must be 'confusion_matrix'
    
    train_metrics = MetricList({
        'MSE' : MeanSquaredError(device=local_rank),
        'R2_Score' : R2Score(device=local_rank)
    })
    
    valid_metrics = MetricList({
        'MSE' : MeanSquaredError(device=local_rank),
        'R2_Score' : R2Score(device=local_rank)
    })
    
    test_metrics = MetricList({
        'MSE' : MeanSquaredError(device=local_rank),
        'R2_Score' : R2Score(device=local_rank)
    })

    if is_wandb:
        wandb.watch(model, criterion, log="all", log_freq=10)
    
    best = 1e9
    best_result = None
    best_eval = 'MSE'

    
    for epoch in range(args.start_epoch, args.epoch):
        logger.log(logger.INFO, f"Epoch {epoch} start")
        train_result, vis_imgs = train_one_epoch(model=ddp_model,
                                       epoch=epoch,
                                       dataloader= dataloader_train,
                                       criterion=criterion, 
                                       optimizer=optimizer,
                                       metrics=train_metrics,
                                       loss_scaler=loss_scaler, 
                                       scheduler=scheduler,
                                       model_ema = model_ema,
                                       clip_grad=args.clip_grad,
                                       logger=logger,  
                                       device=local_rank,
                                       save_vis_freq=args.save_vis_freq,
                                       visualize = visualize
                                       )
                                       

        if global_rank == 0:
            logger.log_every_epcoh(train_result, epoch, header=click.style('Train', fg='blue', bold=True))
            if args.wandb:
                logging_result_in_wandb(result = train_result, stage = 'train', vis_imgs=vis_imgs, epoch = epoch, commit = False)
                
        # logger.log(logger.INFO, "Validation")
        valid_result, vis_imgs = validation(model=model,
                                  epoch=epoch,
                                  dataloader=dataloader_valid, 
                                  criterion=criterion, 
                                  metrics=valid_metrics,
                                  logger=logger, 
                                  loss_scaler=loss_scaler,
                                  device=local_rank,
                                  save_vis_freq=args.save_vis_freq,
                                  visualize = visualize)

        if global_rank==0:
            logger.log_every_epcoh(valid_result, epoch, header=click.style('Valid', fg='yellow', bold=True))
            if args.wandb:
                logging_result_in_wandb(result=valid_result, stage = 'valid', vis_imgs=vis_imgs, epoch=epoch)
        
        # Save Checkpoint
        if args.output_dir and args.save_ckpt and global_rank==0:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch+1 == args.epoch:
                save_model(args, epoch, model, optimizer, loss_scaler, save_dir=args.output_dir, save_file_name=f'checkpoint-{epoch}', model_ema=model_ema)
                
        # Save Checkpoint of Best Socre
        if global_rank==0: 
            valid_eval = valid_result['Evaluation']
            if valid_eval[best_eval] < best:
                logger.log(logger.INFO, '🚀 ' + click.style('Best {0} : {1}'.format(best_eval, valid_eval[best_eval]), bg = 'bright_yellow', fg='cyan', bold = True))
                best = valid_eval[best_eval]
                if args.wandb:
                    wandb.run.summary["best_valid_evaluation_{0}".format(best_eval)] = best
                best_result = copy.deepcopy(valid_eval)
                save_model(args, epoch, model, optimizer, loss_scaler, save_dir=args.output_dir, save_file_name=f'best', model_ema=model_ema)
                logger.log(logger.INFO, "💾 Bset model is saved.\n")
    
    logger.log(logger.INFO, "👍 Best Valid Reulst")
    if best_result is not None:
        for meter, value in best_result.items():
            logger.log(logger.INFO, f"  {meter} : {value}")
            
    if is_wandb:
        wandb.save(os.path.join(args.output_dir, "best.pth"))
        logger.log(logger.INFO, "💾 Save best model in {0}...".format(click.style("wnadb", fg='blue', bold=True)))

    if args.test:
        best_checkpoint = torch.load(os.path.join(args.output_dir, 'best.pth'), map_location='cpu')
        load_model(checkpoint=best_checkpoint, model=model, model_keys=args.model_keys)
        
        test_result, vis_imgs = test(model=model, 
            dataloader=dataloader_test, 
            criterion=criterion, 
            metrics=test_metrics,
            logger=logger, 
            loss_scaler=loss_scaler,
            device=local_rank,
            visualize=visualize)
        
        if global_rank==0:
            logger.log_every_epcoh(test_result, epoch, header=click.style('Test', fg='red', bold=True))
            if args.wandb:
                logging_result_in_wandb(result=test_result, stage = 'test', vis_imgs = vis_imgs, epoch=epoch, commit=True)

def train_one_epoch(model, epoch, dataloader, criterion, optimizer, metrics, loss_scaler, scheduler, model_ema, clip_grad, logger, device, save_vis_freq, visualize=None):
    all_loss = list()
    all_grad_norm = list()
    all_loss_scale_value = list()
    
    iter_len = len(dataloader)
    model.train()
    
    for iter_step, (inputs, targets) in logger.log_every_iter(dataloader, click.style('Train', fg='blue', bold=True)):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
                
        if loss_scaler is None:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            all_loss.append(loss.item())
            loss.backward()

            optimizer.step()
            
        else :
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                all_loss.append(loss.item())
            
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            
            grad_norm = loss_scaler(loss, optimizer, clip_grad=clip_grad, 
                        parameters =model.parameters(), 
                        create_graph=is_second_order)
            
            loss_scale_value = loss_scaler.state_dict()["scale"]
            
            all_grad_norm.append(grad_norm)
            all_loss_scale_value.append(loss_scale_value)

        optimizer.zero_grad()
        loss = loss.detach()
        outputs = outputs.detach()
        targets = targets.detach()
        
        if model_ema is not None:
            model_ema.update(model)
            
        
        metrics.update(outputs, targets)
        if logger.is_print_freq(iter_step, iter_len):
            results = dict()
            results['Evaluation'] = metrics.compute()
            results.update({
                'Loss': 
                    {
                        'Loss':sum(all_loss)/len(all_loss), 
                        'Grad_Norm':sum(all_grad_norm)/len(all_grad_norm),
                        'Loss_Scale':sum(all_loss_scale_value)/len(all_loss_scale_value)
                    }
                })
            results['State'] = {}
            logger.update_meters(results)
    
    vis_imgs = None
    
    if dist.get_rank() ==0 and epoch%save_vis_freq ==0 and visualize is not None:
        vis_imgs = []

        inputs = inputs.detach().cpu()
        outputs = outputs.detach().cpu().squeeze(1)
        targets = targets.detach().cpu()

        for idx in range(inputs.shape[0]):
            if idx >= 5:
                break
            vis_img = visualize(img=inputs[idx], pred=outputs[idx], targets = targets[idx], logger=logger)
            if vis_img == None:
                logger.log(logger.WARNING, click.style("Visualization of reults are None", bg='red', bold=True))
                vis_imgs = None
                break
            vis_imgs.append(vis_img)

    dist.barrier()
    
    # Reduce loss
    all_loss = reduce_loss(device, all_loss)        

    final_results = dict()
    metrics.synced_metric()
    final_results['Evaluation'] = metrics.compute()
    final_results.update({
        'Loss' : {
            'Loss' : all_loss,
            'Grad_Norm':sum(all_grad_norm)/len(all_grad_norm),
            'Loss_Scale':sum(all_loss_scale_value)/len(all_loss_scale_value),
        }})
    if scheduler is not None:
        final_results['State'] = {
            'LR' : scheduler.get_last_lr()[0]
        }
        scheduler.step()
        
    metrics.reset()

    return final_results, vis_imgs

