import datetime
import wandb
import matplotlib.pyplot as plt
import copy
from torcheval.metrics import BinaryConfusionMatrix, MulticlassConfusionMatrix
import time
import torch
import torch.distributed as dist
import logging
import os
import click
import re

class Logger:
    def __init__(self, print_freq, epoch, global_rank, save_dir):
        self.logger = logging.getLogger(str(global_rank))
        self.logger.setLevel(logging.DEBUG)
        
        # Filtering Main Process
        filter = logging.Filter("0")

        streamHandler = logging.StreamHandler()
        streamHandler.setLevel(logging.DEBUG)
        streamHandler.addFilter(filter)
        streamHandler.setFormatter(StreamFormatter())
        
        fileHandler = logging.FileHandler(os.path.join(save_dir, 'log.txt'), encoding='utf-8')
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.addFilter(filter)
        fileHandler.setFormatter(FileFormatter())

        self.logger.addHandler(streamHandler)
        self.logger.addHandler(fileHandler)
    
        self.epoch_logger = {}
        self.iter_logger = {}
        self.print_freq = print_freq
        self.meters = None
        self.epoch = epoch
        
    @property
    def INFO(self):
        return logging.INFO
    
    @property
    def DEBUG(self):
        return logging.DEBUG
    
    @property
    def WARNING(self):
        return logging.WARNING
    
    @property
    def ERROR(self):
        return logging.ERRO
    
    def log(self, level, msg):
        self.logger.log(level, msg)
    
    def update_meters(self, meters:dict):
        self.meters = meters
    
    def compute(self):
        results = {}
        for key, values in self.logger:
            results[key] = sum(values)/len(values)

        return results
        
    def is_print_freq(self, n_iter, iter_len):
        return n_iter % self.print_freq ==0 or n_iter == iter_len - 1
    
    def log_every_epcoh(self, meters, epoch, header=''):
        log_msg = [
            header,
            f'Epoch {epoch} '
            "{meters}"
        ]
        log_msg = ' '.join(log_msg)
        # log_msg = click.style(log_msg, fg='cyan', bold=True)

        meters_str = self.meters_string(meters)
        self.log(self.INFO, log_msg.format(meters=meters_str) + '\n')
        
    def meters_string(self, meters, delimeter=' '):
        meters_string = []

        for metric_name, metric_value in meters['Evaluation'].items():
            if metric_name == 'confusion_matrix':
                continue
            meters_string.append(click.style(f"{metric_name}:{metric_value:0.5f}", fg='cyan', bold=True))

        for k, v in meters['Loss'].items():
            if k.lower() == 'loss':
                meters_string.append(click.style(f"{k}:{v:0.5f}", fg='red', bold=True))
            else :
                meters_string.append(f"{k}:{v:0.5f}")
            
        for k, v in meters['State'].items():
            meters_string.append(f"{k}:{v:0.5f}")
            
        meters_string = ' '.join(meters_string)
        
        return meters_string
    
    def log_every_iter(self, dataloader, header=''):
        start_time = time.time()
        end_time = time.time()
        MB = 1024 * 1024
        
        iter_len = len(dataloader)
        log_msg = [
            header,
            "[{step" + f":>{len(str(iter_len))}" + "}/" + f"{iter_len}]",
            "ETA:{eta} |",
            "{meters}",
            "Iter:{iter_time:.3f}s",
            "Load_Data:{data_loading_time:.3f}s"
        ]
        if torch.cuda.is_available():
            log_msg.append('Max_Memory:{memory:.0f}MB')
        log_msg = ' '.join(log_msg)
        
        global_iter_time = []
        
        for n_iter, obj in enumerate(dataloader):
            data_time = time.time() - end_time
            yield n_iter, obj
            
            if dist.get_rank() ==0:                
                iter_time = time.time() - end_time
                global_iter_time.append(iter_time)
                
                if self.is_print_freq(n_iter, iter_len):
                    eta = sum(global_iter_time) / len(global_iter_time) * ((iter_len) - n_iter)
                    eta_string = str(datetime.timedelta(seconds=int(eta)))
                    
                    meters_str =self.meters_string(self.meters)
                    if torch.cuda.is_available():
                        self.log(self.INFO, log_msg.format(
                                step=n_iter + 1, eta=eta_string,
                                meters=meters_str,
                                iter_time=iter_time, data_loading_time=data_time,
                                memory=torch.cuda.max_memory_allocated() / MB))
                    else:
                        self.log(self.INFO, log_msg.format(
                                step=n_iter, eta=eta_string,
                                meters=meters_str,
                                iter_time=iter_time, data_loading_time=data_time))
        
            end_time = time.time()
            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if dist.get_rank() == 0:
            self.log(self.INFO, '{} Total time: {} ({:.4f} s / it)'.format(
                header, total_time_str, total_time / iter_len))
    
def logging_result_in_wandb(result:dict, stage:str, epoch:int, vis_imgs=None, commit=True):
    assert stage in ['train', 'valid', 'test'] 
    
    if 'confusion_matrix' in result.keys():
        fig, ax = plt.subplots()
        cax = ax.matshow(result['confusion_matrix'])
        fig.colorbar(cax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        result['confusion_matrix'] = fig
        
    if vis_imgs is not None:
        wandb_img = []
        for img in vis_imgs:
            
            img = wandb.Image(img, caption='visualize')
            wandb_img.append(img)
        wandb.log({f'{stage}_vis_results' : wandb_img}, step=epoch, commit=False)

    if stage == 'test':
        for metric, value in result.items():
            wandb.run.summary[f"{stage}_{metric}"] = value

    else :
        wandb.log({stage : result}, step=epoch, commit=commit)
         
def logging_result(train_result:dict, valid_result:dict, epoch):
    if 'confusion_matrix' in train_result.keys():
        fig, ax = plt.subplots()
        cax = ax.matshow(train_result['confusion_matrix'])
        fig.colorbar(cax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        train_result['confusion_matrix'] = fig
        
    if 'confusion_matrix' in valid_result.keys():
        fig, ax = plt.subplots()
        cax = ax.matshow(valid_result['confusion_matrix'])
        fig.colorbar(cax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        valid_result['confusion_matrix'] = fig
    
        
    wandb.log({'train' : train_result, 'valid' : valid_result}, step=epoch)
    
    

# ANSI escape sequences for colors and styles
class StreamFormatter(logging.Formatter):
    
    format = ["%(asctime)s", "%(levelname)s", "%(message)s"]

    FORMATS = {
        logging.DEBUG: f"{format[0]} | {click.style(format[1], fg='black', bold=True)} | {format[2]}",
        logging.INFO: f"{format[0]} | {click.style(format[1], fg='green',bold= True)} | {format[2]}",
        logging.WARNING: f"{format[0]} | {click.style(format[1], fg='yellow', bold=True)} | {format[2]}",
        logging.ERROR: f"{format[0]} | {click.style(format[1], fg='red', bold=True)} | {format[2]}",
    }

    def format(self, record):
        change_line= ''
        if record.msg[-1:] == '\n':
            record.msg=record.msg[:-1]
            change_line='\n'
        log_fmt = self.FORMATS.get(record.levelno, self.format)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record) + change_line

    # ANSI escape sequences for colors and styles
class FileFormatter(logging.Formatter):
    """Custom logging formatter with highlighting using ANSI escape sequences"""

    format = ["%(asctime)s", "%(levelname)s", "%(message)s"]

    FORMATS = {
        logging.DEBUG: f"{format[0]} | {format[1]} | {format[2]}",
        logging.INFO: f"{format[0]} | {format[1]} | {format[2]}",
        logging.WARNING: f"{format[0]} | {format[1]} | {format[2]}",
        logging.ERROR: f"{format[0]} | {format[1]} | {format[2]}",
    }

    def format(self, record):
        change_line= ''
        if record.msg[-1:] == '\n':
            record.msg=record.msg[:-1]
            change_line='\n'
        record.msg = re.sub(r'\x1b\[[0-9;]*m', '', record.msg)
        log_fmt = self.FORMATS.get(record.levelno, self.format)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record) + change_line