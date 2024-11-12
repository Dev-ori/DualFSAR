import argparse
import yaml
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Train
    parser.add_argument('--auto_resume', action='store_true',
                        help='auto resume from last chekcpoint in output directory')
    parser.add_argument('--save_ckpt', action='store_true',
                        help='')
    parser.add_argument('--save_ckpt_freq', default=50, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help="훈련 에폭 개수")
    parser.add_argument('--batch', '-b', type=int, default=None,
                        help='batch size for one iter')

    parser.add_argument('--test', action='store_true')

    # Model_EMA                        
    parser.add_argument('--model_ema', action='store_true')
    parser.add_argument('--model_ema_decay', type=float, default=0.9999)
    parser.add_argument('--model_ema_force_cpu', action='store_true')
    
    # Random Seed
    parser.add_argument('--seed', type=int, default=42,
                        help = 'random seed')
    
    
    # WandDB
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_resume', action='store_true', default=None)
    parser.add_argument('--wandb_project_id', type=str, default=None)
    parser.add_argument('--project', type=str, default=None, 
                        help="wandb 프로젝트 이름")
    parser.add_argument('--run_name', type=str, default=None,
                        help="wandb Log 저장 이름")
    
    # Log
    parser.add_argument('--output_dir', type=str, required=True,
                        help='save directory')
    parser.add_argument('--print_freq', type=int, default=5)
    parser.add_argument('--save_vis_freq', type=int, default=5)

    # DDP
    parser.add_argument('--master_addr', type=str, default='localhost', 
                        help="여러 컴퓨터를 이용하여 DDP를 할 때, Master computer의 IP 주소, DDP를 한 컴퓨터에서만 수행할 시 'localhost'로 설정하면 됨")
    parser.add_argument('--master_port', type=str, default='12355',
                        help="DDP에서 통신할 때, 사용할 포트 번호")
    parser.add_argument('--backend', type=str, default='nccl',
                        help="DDP backend { windows : gloo, linux : nccl }")
    parser.add_argument('--world_size', type=int, default=1,
                        help="전체 컴퓨터의 GPU 개수")
    parser.add_argument('--local_gpus', type=int, default=1,
                        help="현재 컴퓨터에서 사용할 GPU 개수")
    parser.add_argument('--global_rank', type=int, default=0,
                        help="몇 번째 컴퓨터 (0번부터 시작)")
    
    # Model
    parser.add_argument('--model', type=str, required=True,
                        help="모델 이름 (필수)")
    parser.add_argument('--checkpoint', type=str, default='',
                        help="model checkpoint")
    parser.add_argument('--model_keys', type=str, default='model|model_state_dict',
                        help="checkpoint model keys (sep using '|')")
    
    
    # Optimizer
    parser.add_argument('--opt', type=str, default='adam',
                        help='torch or custom  optimizer 이름 ')
    parser.add_argument('--opt_lr', type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warup_epoch', type=int, default=-1, metavar='N',
                        help='num of epoch to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--T', type=int, default=None,
                        help="cosine annealing scheduler T_max")
    parser.add_argument('--T_mult', type=int, default=1,
                        help='cosine aneealing with warmup start T_mult argument')
    parser.add_argument('--gamma' , type=float, default=1,
                        help='cosine annealing decay parameter')

    # Loss
    parser.add_argument('--loss', type=str, default='L1Loss',
                        help='implemented torch loss or customized loss')
    parser.add_argument('--loss_scaler', action='store_true',
                        help='use torch\'s GradScaler')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='For gradient cliping')
    # Datasets
    parser.add_argument('--cfg_dataset', type=str, required=True)
    
    
    args = parser.parse_args()
    return args
