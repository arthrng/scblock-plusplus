import numpy as np
import random
import torch
from sentence_transformers import SentenceTransformer
from blockers.encoder import Encoder

class SBERTEncoder(Encoder):
    def __init__(self, model_path, dataset_name):
        """Initialize Entity Biencoder"""
        super().__init__(dataset_name)

        # Make results reproducible
        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_path).to(self.device)

    def encode_entities(self, entities, excluded_attributes=None):
        """Encode the provided entities"""
        entity_strs = [self.entity_serializer.convert_to_str_representation(entity, excluded_attributes)
                       for entity in entities]

        # Introduce batches here!

        inputs = None
        with torch.no_grad():
            outputs = self.model.encode(entity_strs, show_progress_bar=False)

        return inputs, outputs

    def encode_entities_and_return_pooled_outputs(self, entity, excluded_attributes=None):
        torch.cuda.empty_cache()
        inputs, outputs = self.encode_entities(entity, excluded_attributes)

        # Train SBert models always with poolings, hence no additional pooling is necessary
        return outputs.squeeze().tolist()