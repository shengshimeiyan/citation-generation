import json
import random
import tqdm
from tqdm import tqdm

import numpy as np
import torch
    
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
    """Given .jsonl file, returns a dictionary of the contents."""
    d = {}
    with open(filepath) as f:
        for i, line in tqdm(enumerate(f)):
            d[i] = json.loads(line)
    
    return d