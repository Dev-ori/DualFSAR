import torch
from utils.ddp import reduce_loss
import torch.distributed as dist
import click

@torch.no_grad()
def validation(model, epoch, dataloader, criterion, metrics, logger, loss_scaler=None, device='cpu', save_vis_freq=5, visualize=None):
    model.eval()
    all_loss = list()
    iter_len = len(dataloader)
    for iter_step, (inputs, targets) in logger.log_every_iter(dataloader, click.style('Valid', fg='yellow', bold=True)):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        if loss_scaler is None:
            outputs = model(inputs)
        else :
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        loss = loss.detach()
        outputs = outputs.detach()
        targets = targets.detach()
        
        all_loss.append(loss.item())
        
        metrics.update(outputs, targets)
            
        if logger.is_print_freq(iter_step, iter_len):
            results = dict()
            results['Evaluation'] = metrics.compute()
            results.update({
                'Loss': 
                    {
                        'Loss':sum(all_loss)/len(all_loss), 
                    }
                })
            results['State'] = {}
            logger.update_meters(results)
    vis_imgs = None
    if dist.get_rank() == 0 and epoch%save_vis_freq == 0 and visualize is not None:
        vis_imgs = []

        inputs = inputs.detach().cpu()
        outputs = outputs.detach().cpu().squeeze(1)
        targets = targets.detach().cpu()

        for idx in range(inputs.shape[0]):
            if idx >= 5:
                break
            vis_img = visualize(img = inputs[idx], pred = outputs[idx], targets = targets[idx])
            if vis_img == None:
                logger.log(logger.WARNING, click.style("Visualization's reults are None", bg='red'))
                vis_imgs = None
                break
            vis_imgs.append(vis_img)
    dist.barrier()

    # Reduce Loss
    all_loss = reduce_loss(device, all_loss)        

    final_results = dict()
    metrics.synced_metric()
    final_results['Evaluation'] = metrics.compute()
    final_results.update({
        'Loss' : {
            'Loss' : all_loss,
        }})
    final_results['State'] = {}

    metrics.reset()
        
    return final_results, vis_imgs