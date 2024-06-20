import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import sklearn.metrics as metrics
import argparse
from torch.utils import data
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class BaseEncoder(nn.Module):

    def __init__(self, len_tokenizer, model='roberta-base'):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(model)
        self.transformer.resize_token_embeddings(len_tokenizer)

    def forward(self, input_ids, attention_mask):
        
        output = self.transformer(input_ids, attention_mask)

        return output

# cross-entropy fine-tuning model
class DittoModel(nn.Module):

    def __init__(self, len_tokenizer, alpha, model='roberta-base'):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(model)
        self.transformer.resize_token_embeddings(len_tokenizer)
        self.config = self.transformer.config
        self.alpha = alpha

        hidden_size = self.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2)
        #self.reduction_layer = torch.nn.Linear(2, 1)

    def forward(self, input_ids, input_ids_aug=None):

        if input_ids_aug is not None:
            enc = self.transformer(torch.cat((input_ids, input_ids_aug)))[0][:, 0, :]
            batch_size = len(input_ids)
            enc1 = enc[:batch_size] # (batch_size, emb_size)
            enc2 = enc[batch_size:] # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha, self.alpha)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
             enc = self.transformer(input_ids)[0][:, 0, :]
        return self.fc(enc)