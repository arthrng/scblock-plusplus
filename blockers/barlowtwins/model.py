import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

class BaseEncoder(nn.Module):
    """
    Base Encoder class using a transformer model from Hugging Face's Transformers library.
    """
    def __init__(self, len_tokenizer, model):
        """
        Initialize the BaseEncoder with a transformer model.

        Args:
            len_tokenizer (int): The size of the tokenizer.
            model (str): The name of the pre-trained model to use.
        """
        super().__init__()
        
        # Load the pre-trained transformer model
        self.transformer = AutoModel.from_pretrained(model)
        
        # Resize the token embeddings to match the tokenizer length
        self.transformer.resize_token_embeddings(len_tokenizer)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the BaseEncoder.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Transformer model outputs.
        """
        return self.transformer(input_ids, attention_mask)

def mean_pooling(output, attention_mask):
    """
    Mean-pooling function to obtain sentence embeddings from token embeddings.

    Args:
        output (torch.Tensor): Output from the transformer model.
        attention_mask (torch.Tensor): Attention mask.

    Returns:
        torch.Tensor: Mean-pooled sentence embeddings.
    """
    embeddings = output[0]  # Get the token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()  # Expand attention mask
    # Compute the mean-pooling of the embeddings
    return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class BarlowTwinsPretrainModel(nn.Module):
    """
    Implementation of Barlow Twins Pretrain Model.
    """
    def __init__(self, len_tokenizer, model='roberta-base'):
        """
        Initialize the BarlowTwinsPretrainModel.

        Args:
            len_tokenizer (int): The size of the tokenizer.
            model (str): The name of the pre-trained model to use.
        """
        super().__init__()
        self.encoder = BaseEncoder(len_tokenizer=len_tokenizer, model=model)
        self.config = self.encoder.transformer.config

    def forward(self, input_ids_left, input_ids_right, attention_mask_left, attention_mask_right):
        """
        Forward pass for the BarlowTwinsPretrainModel.

        Args:
            input_ids_left (torch.Tensor): Input token IDs for the left branch.
            input_ids_right (torch.Tensor): Input token IDs for the right branch.
            attention_mask_left (torch.Tensor): Attention mask for the left branch.
            attention_mask_right (torch.Tensor): Attention mask for the right branch.

        Returns:
            torch.Tensor: Concatenated and normalized output embeddings.
        """
        # Encode the left and right input tokens
        output_left = self.encoder(input_ids_left, attention_mask_left)
        output_right = self.encoder(input_ids_right, attention_mask_right)

        # Apply mean pooling to the outputs
        output_left = mean_pooling(output_left, attention_mask_left)
        output_right = mean_pooling(output_right, attention_mask_right)

        # Concatenate the outputs along a new dimension
        output = torch.cat((output_left.unsqueeze(1), output_right.unsqueeze(1)), 1)

        # Normalize the concatenated outputs
        output = F.normalize(output, dim=-1)

        return output

class BarlowTwinsModel(nn.Module):
    """
    Implementation of Barlow Twins Model.
    """
    def __init__(self, len_tokenizer, model='roberta-base'):
        """
        Initialize the BarlowTwinsModel.

        Args:
            len_tokenizer (int): The size of the tokenizer.
            model (str): The name of the pre-trained model to use.
        """
        super().__init__()
        self.encoder = BaseEncoder(len_tokenizer=len_tokenizer, model=model)
        self.config = self.encoder.transformer.config

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the BarlowTwinsModel.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Normalized output embeddings.
        """
        # Encode the input tokens
        output = self.encoder(input_ids, attention_mask)
        
        # Apply mean pooling to the output
        output = mean_pooling(output, attention_mask)
        
        # Normalize the output
        output = F.normalize(output, dim=-1)

        return output
