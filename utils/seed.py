import torch
import numpy as np
import random

def random_seed(seed):
    random.seed(seed)             # Python 내장 랜덤 모듈의 시드 고정
    np.random.seed(seed)          # Numpy 라이브러리의 시드 고정
    torch.manual_seed(seed)       # PyTorch를 사용한 CPU 연산을 위한 시드 고정
    torch.cuda.manual_seed(seed)  # CUDA를 사용한 GPU 연산을 위한 시드 고정
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시 모든 GPU에 대한 시드 고정
    torch.backends.cudnn.deterministic = True  # CUDA의 Deterministic 모드 설정
    torch.backends.cudnn.benchmark = False     # 네트워크의 입력 사이즈가 변하지 않을 때 성능을 개선할 수 있지만, 재현성을 해칠 수 있으므로 False 설정