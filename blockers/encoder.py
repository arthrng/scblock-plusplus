import logging
import sys
sys.path.append('.')
from retrieval.entity_serializer import EntitySerializer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Encoder:
    def __init__(self, dataset_name, context_attributes=None):
        """
        Initialize the Encoder.

        Args:
            dataset_name (str): The name of the dataset.
            context_attributes (list, optional): List of context attributes to include in the entity serialization. Defaults to None.
        """
        self.dataset_name = dataset_name
        self.entity_serializer = EntitySerializer(dataset_name, context_attributes)

    def encode_entities_and_return_pooled_outputs(self, entities, excluded_attributes=None):
        """
        Encode the provided entities and return their pooled outputs.

        This method must be implemented by subclasses.

        Args:
            entities (list): List of entities to be encoded.
            excluded_attributes (list, optional): List of attributes to exclude from serialization. Defaults to None.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        logger = logging.getLogger(__name__)
        logger.warning('encode_entities_and_return_pooled_outputs method not implemented in base Encoder class.')

        raise NotImplementedError('encode_entities_and_return_pooled_outputs method must be implemented by subclasses.')
