import torch
from transformers import AutoTokenizer
from blockers.encoder import Encoder
from blockers.barlowtwins.model import BarlowTwinsModel

class BarlowTwinsEncoder(Encoder):
    def __init__(self, model_path, dataset_name, max_length=128):
        """
        Initialize the BarlowTwinsEncoder.

        Args:
            model_path (str): Path to the pretrained Barlow Twins model.
            dataset_name (str): Name of the dataset used for initialization.
            max_length (int, optional): Maximum length of tokenized input sequences. Defaults to 128.
        """
        super().__init__(dataset_name)

        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(
            'roberta-base', additional_special_tokens=('[COL]', '[VAL]')
        )
        self.model = BarlowTwinsModel(len_tokenizer=len(self.tokenizer), model='roberta-base').to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)

    def encode_entities(self, entities, excluded_attributes=None):
        """
        Encode the provided entities using the Barlow Twins model.

        Args:
            entities (list): List of entities to encode.
            excluded_attributes (list, optional): List of attributes to exclude from serialization. Defaults to None.

        Returns:
            tuple: Tokenized inputs and model outputs.
        """
        # Convert entities to string representations
        entity_strs = [
            self.entity_serializer.convert_to_str_representation(entity, excluded_attributes)
            for entity in entities
        ]

        # Tokenize the entity strings
        inputs = self.tokenizer(
            entity_strs, return_tensors='pt', padding=True,
            truncation=True, max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            # Get model outputs
            outputs = self.model(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask']
            )

        return inputs, outputs

    def encode_entities_and_return_pooled_outputs(self, entities, excluded_attributes=None):
        """
        Encode entities and return the pooled outputs as a list.

        Args:
            entities (list): List of entities to encode.
            excluded_attributes (list, optional): List of attributes to exclude from serialization. Defaults to None.

        Returns:
            list: Pooled outputs for each entity.
        """
        torch.cuda.empty_cache()
        _, outputs = self.encode_entities(entities, excluded_attributes)

        # Convert outputs to list
        return outputs.squeeze().tolist()
