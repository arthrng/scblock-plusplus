import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel

def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on the token embeddings provided by the transformer model.

    Parameters:
    - model_output (tuple): Contains token embeddings from the model.
    - attention_mask (torch.Tensor): Mask to identify which tokens are padding.

    Returns:
    - torch.Tensor: Mean-pooled token embeddings.
    """
    token_embeddings = model_output[0]  # Extract token embeddings from the model output
    # Expand the attention mask to match token_embeddings dimensions
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # Compute mean by summing token embeddings and dividing by the sum of the mask
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class BaseEncoder(nn.Module):
    """
    Base encoder class using a transformer model from Hugging Face.
    """
    def __init__(self, len_tokenizer, model='roberta-base'):
        """
        Initialize the BaseEncoder with a pre-trained transformer model.

        Parameters:
        - len_tokenizer (int): Number of tokens in the tokenizer vocabulary.
        - model (str): Pre-trained transformer model to use (e.g., 'roberta-base').
        """
        super(BaseEncoder, self).__init__()
        self.transformer = AutoModel.from_pretrained(model)
        self.transformer.resize_token_embeddings(len_tokenizer)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the transformer model.

        Parameters:
        - input_ids (torch.Tensor): Token IDs of the input sequences.
        - attention_mask (torch.Tensor): Attention mask for padding tokens.

        Returns:
        - tuple: Model output including token embeddings and other outputs.
        """
        return self.transformer(input_ids, attention_mask)

class DittoModel(nn.Module):
    """
    Model for fine-tuning with data augmentation using a transformer-based architecture.
    """
    def __init__(self, len_tokenizer, alpha, model='roberta-base'):
        """
        Initialize the DittoModel with a transformer model and a classification head.

        Parameters:
        - len_tokenizer (int): Number of tokens in the tokenizer vocabulary.
        - alpha (float): Parameter for data augmentation interpolation.
        - model (str): Pre-trained transformer model to use (e.g., 'roberta-base').
        """
        super(DittoModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model)
        self.transformer.resize_token_embeddings(len_tokenizer)
        self.config = self.transformer.config
        self.alpha = alpha

        hidden_size = self.config.hidden_size
        self.fc = nn.Linear(hidden_size, 2)  # Classification head for binary classification

    def forward(self, input_ids, input_ids_aug=None):
        """
        Forward pass with optional data augmentation.

        Parameters:
        - input_ids (torch.Tensor): Token IDs of the original input sequences.
        - input_ids_aug (torch.Tensor, optional): Token IDs of augmented input sequences.

        Returns:
        - torch.Tensor: Predictions from the classification head.
        """
        if input_ids_aug is not None:
            # Concatenate original and augmented inputs
            enc = self.transformer(torch.cat((input_ids, input_ids_aug)))[0][:, 0, :]
            batch_size = len(input_ids)
            enc1 = enc[:batch_size]  # Embeddings for original input
            enc2 = enc[batch_size:]  # Embeddings for augmented input

            # Interpolate between original and augmented embeddings
            aug_lam = np.random.beta(self.alpha, self.alpha)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            # Forward pass with original inputs only
            enc = self.transformer(input_ids)[0][:, 0, :]

        # Apply the classification head
        return self.fc(enc)