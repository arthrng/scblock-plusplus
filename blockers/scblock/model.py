import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

class BaseEncoder(nn.Module):
    """
    Base Encoder using a transformer model from Hugging Face Transformers.

    Args:
        len_tokenizer (int): The size of the tokenizer's vocabulary.
        model (str): Pretrained transformer model name.
        is_auxiliary (bool): If True, initializes weights of the last three layers of the transformer.
    """
    def __init__(self, len_tokenizer, model, is_auxiliary=False):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model)
        self.transformer.resize_token_embeddings(len_tokenizer)

        if is_auxiliary:
            # Initialize the weights of the last three transformer layers
            self.transformer.encoder.layer[-1].apply(self.transformer._init_weights)
            self.transformer.encoder.layer[-2].apply(self.transformer._init_weights)
            self.transformer.encoder.layer[-3].apply(self.transformer._init_weights)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the transformer.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask to avoid attending to padding tokens.

        Returns:
            tuple: Output of the transformer.
        """
        return self.transformer(input_ids, attention_mask)

def mean_pooling(output, attention_mask):
    """
    Mean pooling of the transformer outputs.

    Args:
        output (tuple): Transformer outputs (last_hidden_state, etc.).
        attention_mask (torch.Tensor): Attention mask to consider valid tokens only.

    Returns:
        torch.Tensor: Mean-pooled embeddings.
    """
    embeddings = output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class ContrastivePretrainModel(nn.Module):
    """
    SC-Block model to train.

    Args:
        len_tokenizer (int): The size of the tokenizer's vocabulary.
        model (str): Pretrained transformer model name.
        is_auxiliary (bool): If True, initializes weights of the last three layers of the transformer.
        has_classification_head (bool): If True, includes a classification head (not currently implemented).
    """
    def __init__(self, len_tokenizer, model='roberta-base', is_auxiliary=False, has_classification_head=False):
        super().__init__()
        self.encoder = BaseEncoder(len_tokenizer=len_tokenizer, model=model, is_auxiliary=is_auxiliary)
        self.config = self.encoder.transformer.config

    def reset_weights(self):
        """
        Reinitialize the weights of the last three transformer layers and freeze the rest.
        """
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
        """
        Forward pass through the model with two inputs for contrastive learning.

        Args:
            input_ids_left (torch.Tensor): Input token IDs for the left input.
            input_ids_right (torch.Tensor): Input token IDs for the right input.
            attention_mask_left (torch.Tensor): Attention mask for the left input.
            attention_mask_right (torch.Tensor): Attention mask for the right input.

        Returns:
            torch.Tensor: Concatenated and normalized outputs from both inputs.
        """
        output_left = self.encoder(input_ids_left, attention_mask_left)
        output_right = self.encoder(input_ids_right, attention_mask_right)

        output_left = mean_pooling(output_left, attention_mask_left)
        output_right = mean_pooling(output_right, attention_mask_right)

        # Concatenate and normalize outputs
        output = torch.cat((output_left.unsqueeze(1), output_right.unsqueeze(1)), 1)
        return F.normalize(output, dim=-1)

class ContrastiveModel(nn.Module):
    """
    SC-Block.

    Args:
        len_tokenizer (int): The size of the tokenizer's vocabulary.
        model (str): Pretrained transformer model name.
        is_auxiliary (bool): If True, the model behaves differently for auxiliary tasks.
    """
    def __init__(self, len_tokenizer, model='roberta-base', is_auxiliary=False):
        super().__init__()
        self.encoder = BaseEncoder(len_tokenizer=len_tokenizer, model=model)
        self.config = self.encoder.transformer.config
        self.is_auxiliary = is_auxiliary

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model for contrastive learning.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask to avoid padding tokens.

        Returns:
            torch.Tensor: Normalized embeddings after mean pooling.
        """
        output = self.encoder(input_ids, attention_mask)
        output = mean_pooling(output, attention_mask)

        if self.is_auxiliary:
            # For auxiliary tasks, duplicate the output
            output = torch.cat((output.unsqueeze(1), output.unsqueeze(1)), 1)

        return F.normalize(output, dim=-1)
