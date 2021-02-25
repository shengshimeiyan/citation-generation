import json
import random
import tqdm
from tqdm import tqdm

import numpy as np
import torch

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
def check_gpu():
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f"There is {torch.cuda.device_count()} GPU(s) available.")
        print("Device name:", torch.cuda.get_device_name(0))

    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
        
    return device

def load_data(filepath):
    d = {}
    i = 0
    with open(filepath) as f:
        for line in tqdm(f):
            d[i] = json.loads(line)
            i += 1
    
    return d

