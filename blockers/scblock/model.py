"""
Author: Arthur Ning (558688an@eur.nl)
Date: April 30, 2024
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

class BaseEncoder(nn.Module):
    def __init__(self, len_tokenizer, model, is_auxiliary=False):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(model)
        self.transformer.resize_token_embeddings(len_tokenizer)

        if is_auxiliary:
          self.transformer.encoder.layer[-1].apply(self.transformer._init_weights)
          self.transformer.encoder.layer[-2].apply(self.transformer._init_weights)
          self.transformer.encoder.layer[-3].apply(self.transformer._init_weights)

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

class ContrastivePretrainModel(nn.Module):
    """
    Implementation of SC-Block introduced by Brinkmann et. al (2023)
    
    """
    def __init__(self, len_tokenizer, model='roberta-base', is_auxiliary=False, has_classification_head=False):
        super().__init__()
        
        self.encoder = BaseEncoder(len_tokenizer=len_tokenizer, model=model, is_auxiliary=is_auxiliary)
        self.config = self.encoder.transformer.config
    
    def reset_weights(self):
        # Reinitialize weights in the last three layers
        self.encoder.transformer.encoder.layer[-1].apply(self.encoder.transformer._init_weights)
        self.encoder.transformer.encoder.layer[-2].apply(self.encoder.transformer._init_weights)
        self.encoder.transformer.encoder.layer[-3].apply(self.encoder.transformer._init_weights)

        # Freeze parameters in all other layers except for the last three
        for param in self.encoder.parameters():
            param.requires_grad = False

        for i in range(1, 4):
            for param in self.encoder.transformer.encoder.layer[-i].parameters():
                param.requires_grad = True

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

class ContrastiveModel(nn.Module):
    """
    Implementation of SC-Block introduced by Brinkmann et. al (2023)
    
    """
    def __init__(self, len_tokenizer, model='roberta-base', is_auxiliary=False):
        super().__init__()
        
        self.encoder = BaseEncoder(len_tokenizer=len_tokenizer, model=model)
        self.config = self.encoder.transformer.config
        self.is_auxiliary = is_auxiliary

    def forward(self, input_ids, attention_mask):
        # Encode tokens
        output = self.encoder(input_ids, attention_mask)

        # Mean-pool both outputs
        output = mean_pooling(output, attention_mask)

        if self.is_auxiliary:
          output = torch.cat((output.unsqueeze(1), output.unsqueeze(1)), 1)

        # Normalize the output
        output = F.normalize(output, dim=-1)

        # Return the output
        return output