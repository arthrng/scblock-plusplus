import torch
from blockers.encoder import Encoder
from transformers import AutoTokenizer
import sys
import numpy as np
from blockers.scblock.model import ContrastiveModel

class SCBlockEncoder(Encoder):
    def __init__(self, model_path, dataset_name, max_length=128):
        """Initialize Entity Biencoder"""
        super().__init__(dataset_name)

        self.max_length = max_length
        self.device = torch.device("cuda:{}".format('0') if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())
        print(torch.__version__)
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', additional_special_tokens=('[COL]', '[VAL]'))
        self.model = ContrastiveModel(len_tokenizer=len(self.tokenizer), model='roberta-base').to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)), strict=False)

    def encode_entities(self, entities, excluded_attributes=None):
        """Encode the provided entities"""
        entity_strs = [self.entity_serializer.convert_to_str_representation(entity, excluded_attributes)
                       for entity in entities]

        # Introduce batches here!
        inputs = self.tokenizer(entity_strs, return_tensors='pt', padding=True,
                                truncation=True, max_length=self.max_length).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        
        return inputs, outputs

    def encode_entities_and_return_pooled_outputs(self, entity, excluded_attributes=None):
        torch.cuda.empty_cache()
        inputs, outputs = self.encode_entities(entity, excluded_attributes)
        print('done')


        return outputs.squeeze().tolist()