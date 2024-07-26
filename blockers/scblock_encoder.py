import torch
from blockers.encoder import Encoder
from transformers import AutoTokenizer
from blockers.scblock.model import ContrastiveModel

class SCBlockEncoder(Encoder):
    def __init__(self, model_path, dataset_name, max_length=128):
        """
        Initialize the SCBlock Encoder with a specified model and dataset.

        Args:
            model_path (str): Path to the pre-trained SCBlock model.
            dataset_name (str): The name of the dataset.
            max_length (int, optional): Maximum sequence length for tokenization. Defaults to 128.
        """
        super().__init__(dataset_name)

        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', additional_special_tokens=('[COL]', '[VAL]'))
        self.model = ContrastiveModel(len_tokenizer=len(self.tokenizer), model='roberta-base').to(self.device)
        
        # Load pre-trained model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)

    def encode_entities(self, entities, excluded_attributes=None):
        """
        Encode the provided entities into embeddings using SCBlock.

        Args:
            entities (list): List of entities to be encoded.
            excluded_attributes (list, optional): Attributes to exclude during serialization. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - inputs (dict): Tokenizer outputs including input_ids and attention_mask.
                - outputs (torch.Tensor): The SCBlock embeddings for the input entities.
        """
        # Convert entities to string representations
        entity_strs = [self.entity_serializer.convert_to_str_representation(entity, excluded_attributes)
                       for entity in entities]

        # Tokenize entity strings
        inputs = self.tokenizer(entity_strs, return_tensors='pt', padding=True,
                                truncation=True, max_length=self.max_length).to(self.device)
        
        # Encode the tokenized inputs
        with torch.no_grad():
            outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        
        return inputs, outputs

    def encode_entities_and_return_pooled_outputs(self, entities, excluded_attributes=None):
        """
        Encode the provided entities and return their pooled embeddings.

        Args:
            entities (list): List of entities to be encoded.
            excluded_attributes (list, optional): Attributes to exclude during serialization. Defaults to None.

        Returns:
            list: The SCBlock embeddings for the input entities, converted to a list.
        """
        # Clear GPU memory cache
        torch.cuda.empty_cache()

        # Encode entities and get outputs
        _, outputs = self.encode_entities(entities, excluded_attributes)

        # Return the embeddings as a list
        return outputs.squeeze().tolist()
