import json
import re
import random
import time
import tqdm 
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import AutoTokenizer, AutoModel

from utils import set_seed, check_gpu, load_data

################################################################################

def preprocessing(X, pretrained_weights, max_length=512):
    # error: unable to make this function use dataloader
    input_ids = []
    attention_mask = []

    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)
    for text in X:
        encodings = tokenizer.encode(
            text=text, 
            add_special_tokens=True, 
            truncation=True, 
            padding="max_length", 
            max_length=max_length
        )        
        input_ids.append(encodings)

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(np.where(input_ids != 0, 1, 0))

    return input_ids, attention_mask

def create_dataloader(X, pretrained_weights, sampler, 
                      y=None, max_length=512, batch_size=32): 
    inputs, mask = preprocessing(X, pretrained_weights, max_length)
    if y is not None: 
        dataset = TensorDataset(inputs, mask, torch.tensor(y.values))
    else: 
        dataset = TensorDataset(inputs, mask)
    
    return DataLoader(dataset, 
                      sampler=sampler(dataset), 
                      batch_size=batch_size)

###############################################################################

class CiteBERT(nn.Module):
    def __init__(self, pretrained_weights, D_out, H=256, freeze_BERT=True):
        super().__init__()
        self.BERT = AutoModel.from_pretrained(pretrained_weights)
        self.classifier = nn.Sequential(
            nn.Linear(self.BERT.config.hidden_size, H), 
            nn.ReLU(), 
            nn.Linear(H, D_out)
        )

        if freeze_BERT: # do not update params of bert
            for param in self.BERT.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.BERT(input_ids=input_ids, 
                            attention_mask=attention_mask)
        
        logits = self.classifier(outputs[0][:, 0, :]) # only take CLS token
        return logits

def train(model, train_dataloader, valid_dataloader=None, epochs=2, device=None):
    start = time.time()
    for epoch_i in range(epochs):
        print("-"*30)
        print(f"| Epoch | Batch | Train Loss |")
        print("-"*30)
        
        epoch_start = time.time()
        model.train() 

        total_loss, b_loss, b_count = 0, 0, 0
        for step, batch in enumerate(train_dataloader):
            model.zero_grad() 
            
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            b_count += 1
            
            logits = model(b_input_ids, b_attn_mask) 
            
            loss = criterion(logits, b_labels) 
            b_loss += loss.item()
            total_loss += loss.item()

            loss.backward() 

            optimizer.step() 
            scheduler.step() 

            # print training results
            if (step%20 == 0 and step != 0) or (step==len(train_dataloader)-1):
                print(f"| {epoch_i+1:^5} | {step:^5} | {b_loss/b_count:^10.4f} |")
                b_loss, b_count = 0, 0
                
        avg_train_loss = total_loss/len(train_dataloader)
        
        print("-"*30)
        print(f"{'Avg. Train Loss:':^17} {avg_train_loss:^20.4f}")

        # evaluate current model on validation set
        if valid_dataloader: 
            model.eval() 

            valid_loss, precision, macro_avg_f1 = [], [], []
            for batch in valid_dataloader:
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

                with torch.no_grad(): 
                    logits = model(b_input_ids, b_attn_mask) 

                loss = criterion(logits, b_labels) 
                valid_loss.append(loss.item())

                proba = F.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(proba, axis=1)
                report = classification_report(b_labels.tolist(), preds.tolist(), 
                                               output_dict=True, zero_division=0)

                if 2 in b_labels:
                    precision.append(report["2"]["precision"])
                if len(b_labels) > 1:
                    macro_avg_f1.append(report["macro avg"]["f1-score"])
                    
            # print training progress
            print(f"{'Avg. Valid Loss:':^17} {np.mean(valid_loss):^20.4f}")
            print(f"{'Mean Macro Avg. F1:':^20} {np.mean(macro_avg_f1):^14.4f}")
            print(f"{'Mean Precision, Class 2:':^20} {np.mean(precision):^.4f}") 

        print(f"\nTime taken: {round(time.time()-epoch_start, 4)}s")
        print("-"*30)

    print("\nTraining complete.")
    print(f"Time taken: {round(time.time()-start, 4)}s")    
    
def predict_prob(model, test_dataloader, device=None):
    model.eval()

    logits = []
    for batch in tqdm(test_dataloader):
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits.append(model(b_input_ids, b_attn_mask))
    
    logits = torch.cat(logits, dim=0)
    prob = F.softmax(logits, dim=1).cpu().numpy()

    return prob

###############################################################################

if __name__ == "__main__":
    set_seed()
    device = check_gpu()

    
    print("\nPreparing train data...")
    d = load_data("/home/jessica/data/SciCite/train.jsonl")
    data = pd.DataFrame.from_dict(d).T
    data = data.loc[:, ["string", "label"]] 
    
    # replace categorical labels with numbers
    label_map = {"background": 0, "method": 1, "result": 2}
    data["label"].replace(label_map, inplace=True)
    
    # train-valid split
    X_train, X_valid, y_train, y_valid = train_test_split(
        data.string, data.label, test_size=0.3
    )
    
    print("Preparing dataloaders...")
    EPOCHS = 8
    MAX_LEN = 512
    BATCH_SIZE = 32
    PRETRAINED_WEIGHTS="allenai/scibert_scivocab_uncased"
    
    train_dataloader = create_dataloader(
        X_train, pretrained_weights=PRETRAINED_WEIGHTS, y=y_train, 
        sampler=RandomSampler, max_length=MAX_LEN, batch_size=BATCH_SIZE
    )
    valid_dataloader = create_dataloader(
        X_valid, pretrained_weights=PRETRAINED_WEIGHTS, y=y_valid, 
        sampler=SequentialSampler, max_length=MAX_LEN, batch_size=BATCH_SIZE
    )
    
    
    print("Preparing the model...")
    model = CiteBERT(PRETRAINED_WEIGHTS, D_out=3)
    criterion = nn.CrossEntropyLoss() 
    optimizer = AdamW(model.parameters())
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=len(train_dataloader) * EPOCHS)
    model.to(device)
    
    print(f"\nTraining for {EPOCHS} epoch(s)...")
    train(model, train_dataloader, valid_dataloader, epochs=EPOCHS, device=device)
    
    
    EPOCHS = 5 
    print(f"\nTraining new instance of model on total data for {EPOCHS} epoch(s)...")
    total_dataloader = create_dataloader(
        data.string, data.label, pretrained_weights=PRETRAINED_WEIGHTS, 
        sampler=RandomSampler, max_length=MAX_LEN, batch_size=BATCH_SIZE
    )
    
    model = CiteBERT(PRETRAINED_WEIGHTS, D_out=3)
    criterion = nn.CrossEntropyLoss() 
    optimizer = AdamW(model.parameters())
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=len(total_dataloader) * EPOCHS)
    model.to(device)
    
    train(model, total_dataloader, epochs=EPOCHS, device=device)
    
    
    print("\nPreparing the test data...")
    d2 = load_data("/home/jessica/data/SciCite/test.jsonl")
    test = pd.DataFrame.from_dict(d2).T
    test = test.loc[:, ["string", "label"]] 
    test["label"].replace(label_map, inplace=True)
    
    X_test, y_test = test.string, test.label
    test_dataloader = create_dataloader(
        X_test, pretrained_weights=PRETRAINED_WEIGHTS, 
        sampler=SequentialSampler, max_length=MAX_LEN, batch_size=BATCH_SIZE
    )
    
    
    print("\nEvaluating model on test set...")
    prob = predict_prob(model, test_dataloader, device=device)
    y_pred = pd.DataFrame(np.argmax(proba, axis=1))
    print(classification_report(y_test, y_pred, digits=4))
    
    
    state_dict_path = "./models/CiteSciBERT"
    torch.save(model.state_dict(), state_dict_path)
    print(f"\nTrained model weights are saved at '{state_dict_path}'")
