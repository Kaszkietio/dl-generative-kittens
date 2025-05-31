import numpy as np
import torch
import random


def set_seed(seed: int):
    random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)

def get_device():
    return "cuda:0" if torch.cuda.device_count() > 0 else 'cpu'