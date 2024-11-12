from utils.parser import parse_args
from utils.ddp import DDP_run
from apps.train import train

if __name__ == "__main__":
    args = parse_args()
    
    DDP_run(train, args) # change my_function to train
    
#lsof -nP -iTCP:12355 | grep LISTEN