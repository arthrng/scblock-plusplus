import numpy as np
import random
import torch
from sentence_transformers import SentenceTransformer
from blockers.encoder import Encoder

class SBERTEncoder(Encoder):
    def __init__(self, model_path, dataset_name):
        """
        Initialize the SBERT Encoder with a specified model and dataset.

        Args:
            model_path (str): Path or identifier of the pre-trained SBERT model.
            dataset_name (str): The name of the dataset.
        """
        super().__init__(dataset_name)

        # Set seed for reproducibility
        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_path).to(self.device)

    def encode_entities(self, entities, excluded_attributes=None):
        """
        Encode the provided entities into embeddings using SBERT.

        Args:
            entities (list): List of entities to be encoded.
            excluded_attributes (list, optional): Attributes to exclude during serialization. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - inputs (None): SBERT does not use raw inputs directly.
                - outputs (numpy.ndarray): The SBERT embeddings for the input entities.
        """
        # Convert entities to string representations
        entity_strs = [self.entity_serializer.convert_to_str_representation(entity, excluded_attributes)
                       for entity in entities]

        # Encode the entity strings using SBERT
        with torch.no_grad():
            outputs = self.model.encode(entity_strs, show_progress_bar=False)

        return None, outputs

    def encode_entities_and_return_pooled_outputs(self, entities, excluded_attributes=None):
        """
        Encode the provided entities and return their pooled embeddings.

        Args:
            entities (list): List of entities to be encoded.
            excluded_attributes (list, optional): Attributes to exclude during serialization. Defaults to None.

        Returns:
            list: The SBERT embeddings for the input entities, converted to a list.
        """
        # Clear GPU memory cache
        torch.cuda.empty_cache()

        # Encode entities and get outputs
        _, outputs = self.encode_entities(entities, excluded_attributes)

        # Convert outputs to list
        return outputs.tolist()
