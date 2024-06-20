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
class ContrastiveClassifierModel(nn.Module):
    def __init__(self, len_tokenizer, checkpoint_path=None, model='roberta-base', frozen=True):
        super().__init__()

        self.frozen = frozen

        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config
        self.classification_head = ClassificationHead(self.config)
        self.checkpoint_path = checkpoint_path

        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path)
            self.load_state_dict(checkpoint, strict=False)

        # if self.frozen:
        #     for param in self.encoder.parameters():
        #         param.requires_grad = False
        
    def forward(self, input_ids_left, attention_mask_left, input_ids_right, attention_mask_right):
        output_left = self.encoder(input_ids_left, attention_mask_left)
        output_right = self.encoder(input_ids_right, attention_mask_right)

        output_left = mean_pooling(output_left, attention_mask_left)
        output_right = mean_pooling(output_right, attention_mask_right)

        output = torch.cat((output_left, output_right, torch.abs(output_left - output_right), output_left * output_right), dim=-1)

        proj_output = self.classification_head(output)
        proj_output = torch.sigmoid(proj_output)

        return proj_output

class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = 4 * config.hidden_size
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(self.hidden_size, 1)

    def forward(self, features):
        x = self.dropout(features)
        x = self.out_proj(x)
        return x