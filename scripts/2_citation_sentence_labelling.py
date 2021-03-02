import json
import os
import re
import random
import tqdm 
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import AutoTokenizer, AutoModel

from citation_intent_classification import CiteBERT, predict_prob
from utils import check_gpu, load_data

################################################################################

if __name__ == "__main__":
    device = check_gpu()
    MAX_LEN = 512
    BATCH_SIZE = 32
    PRETRAINED_WEIGHTS = "allenai/scibert_scivocab_uncased"
    
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_WEIGHTS)

    model = CiteBERT(PRETRAINED_WEIGHTS, D_out=3)
    model.load_state_dict(torch.load("./models/CiteSciBERT"))
    model.to(device)
    
    filepath = "./misc/citation_sentences/"
    for filename in os.listdir(filepath):
        if not filename.startswith("x"): continue
        
        print(filename)
        d = load_data(os.path.join(filepath, filename))
        shard = pd.DataFrame.from_dict(d).T
        
        print("\nPreprocessing the citation sentences...")
        dataloader = DataLoader(
            shard["citation_sentence"], 
            sampler=SequentialSampler(shard["citation_sentence"]),  
            batch_size=BATCH_SIZE)

        # get input_ids, attention_masks encodings
        input_ids = []
        attention_masks = []
        for batch in tqdm(dataloader):
            b_input_ids, b_attn_mask = tokenizer(
                batch, 
                max_length=MAX_LEN, 
                padding="max_length",
                truncation=True,  
                add_special_tokens=True, 
                return_token_type_ids=False).values()

            input_ids.extend(b_input_ids)
            attention_masks.extend(b_attn_mask)

        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)    
        encodings = TensorDataset(input_ids, attention_masks)
        model_input = DataLoader(
            encodings, 
            sampler=SequentialSampler(encodings), 
            batch_size=BATCH_SIZE)

        prob = predict_prob(model, model_input, device=device)
        shard["background_prob"] = prob[:,0]
        shard["method_prob"] = prob[:,1]
        shard["result_prob"] = prob[:,2]
        shard["intent"] = np.argmax(prob, 1)
        shard["intent"].replace({0: "background", 
                                 1: "method", 
                                 2: "result"}, inplace=True)

        outpath = f"./misc/intent_probabilities_{filename}.jsonl"
        shard.to_json(outpath, orient="records", lines=True)