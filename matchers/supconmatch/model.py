import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on the token embeddings, taking the attention mask into account.

    :param model_output: Tuple containing the model's output. The first element contains token embeddings.
    :param attention_mask: Attention mask for the input tokens.
    :return: Mean pooled token embeddings.
    """
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class BaseEncoder(nn.Module):
    """
    Base Encoder using a pre-trained transformer model.
    """

    def __init__(self, len_tokenizer, model='roberta-base'):
        """
        Initialize the BaseEncoder.

        :param len_tokenizer: Length of the tokenizer vocabulary.
        :param model: Name of the pre-trained model.
        """
        super(BaseEncoder).__init__()
        self.transformer = AutoModel.from_pretrained(model)
        self.transformer.resize_token_embeddings(len_tokenizer)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the transformer.

        :param input_ids: Input token IDs.
        :param attention_mask: Attention mask for the inputs.
        :return: Output of the transformer model.
        """
        return self.transformer(input_ids, attention_mask)

class ContrastiveClassifierModel(nn.Module):
    """
    Contrastive Classifier Model using cross-entropy fine-tuning.
    """

    def __init__(self, len_tokenizer, checkpoint_path=None, model='roberta-base', frozen=True):
        """
        Initialize the ContrastiveClassifierModel.

        :param len_tokenizer: Length of the tokenizer vocabulary.
        :param checkpoint_path: Path to the checkpoint file.
        :param model: Name of the pre-trained model.
        :param frozen: Whether to freeze the encoder parameters during training.
        """
        super().__init__()
        self.frozen = frozen
        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config
        self.classification_head = ClassificationHead(self.config)
        self.checkpoint_path = checkpoint_path

        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path)
            self.load_state_dict(checkpoint, strict=False)

        if self.frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids_left, attention_mask_left, input_ids_right, attention_mask_right):
        """
        Forward pass through the model.

        :param input_ids_left: Input IDs for the left side.
        :param attention_mask_left: Attention mask for the left side.
        :param input_ids_right: Input IDs for the right side.
        :param attention_mask_right: Attention mask for the right side.
        :return: Output of the classification head.
        """
        output_left = self.encoder(input_ids_left, attention_mask_left)
        output_right = self.encoder(input_ids_right, attention_mask_right)

        pooled_left = mean_pooling(output_left, attention_mask_left)
        pooled_right = mean_pooling(output_right, attention_mask_right)

        combined_features = torch.cat(
            (pooled_left, pooled_right, torch.abs(pooled_left - pooled_right), pooled_left * pooled_right), dim=-1
        )

        proj_output = self.classification_head(combined_features)
        return torch.sigmoid(proj_output)

class ClassificationHead(nn.Module):
    """
    Classification head for the Contrastive Classifier Model.
    """

    def __init__(self, config):
        """
        Initialize the ClassificationHead.

        :param config: Configuration object of the transformer model.
        """
        super(ClassificationHead).__init__()
        self.hidden_size = 4 * config.hidden_size
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(self.hidden_size, 1)

    def forward(self, features):
        """
        Forward pass through the classification head.

        :param features: Combined features from the left and right inputs.
        :return: Logits for classification.
        """
        x = self.dropout(features)
        return self.out_proj(x)
