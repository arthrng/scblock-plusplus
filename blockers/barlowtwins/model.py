"""
Author: Arthur Ning (558688an@eur.nl)
Date: April 30, 2024
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

class BaseEncoder(nn.Module):
    def __init__(self, len_tokenizer, model):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(model)
        self.transformer.resize_token_embeddings(len_tokenizer)

    def forward(self, input_ids, attention_mask):
        # Return output from transformer model
        return self.transformer(input_ids, attention_mask)

# Mean-pooling layer
def mean_pooling(output, attention_mask):
    # Extract embeddings
    embeddings = output[0]

    # Expand
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()

    # Mean-pool embedding
    return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class BarlowTwinsPretrainModel(nn.Module):
    """
    Implementation of SC-Block introduced by Brinkmann et. al (2023)
    
    """
    def __init__(self, len_tokenizer, model='roberta-base'):
        super().__init__()
        
        self.encoder = BaseEncoder(len_tokenizer=len_tokenizer, model=model)
        self.config = self.encoder.transformer.config

    def forward(self, input_ids_left, input_ids_right, attention_mask_left, attention_mask_right):
        # Encode tokens
        output_left = self.encoder(input_ids_left, attention_mask_left)
        output_right = self.encoder(input_ids_right, attention_mask_right)

        # Mean-pool both outputs
        output_left = mean_pooling(output_left, attention_mask_left)
        output_right = mean_pooling(output_right, attention_mask_right)

        # Concatenate two outputs
        output = torch.cat((output_left.unsqueeze(1), output_right.unsqueeze(1)), 1)

        # Normalize the output
        output = F.normalize(output, dim=-1)

        # Return the output
        return output

class BarlowTwinsModel(nn.Module):
    """
    Implementation of SC-Block introduced by Brinkmann et. al (2023)
    
    """
    def __init__(self, len_tokenizer, model='roberta-base'):
        super().__init__()
        
        self.encoder = BaseEncoder(len_tokenizer=len_tokenizer, model=model)
        self.config = self.encoder.transformer.config

    def forward(self, input_ids, attention_mask):
        # Encode tokens
        output = self.encoder(input_ids, attention_mask)

        # Mean-pool both outputs
        output = mean_pooling(output, attention_mask)

        # Normalize the output
        output = F.normalize(output, dim=-1)

        # Return the output
        return output