import logging
import sys
sys.path.append('.')
from indexing.entity_serializer import EntitySerializer

class Encoder:
    def __init__(self, dataset_name, context_attributes=None):
        self.dataset_name = dataset_name
        self.entity_serializer = EntitySerializer(dataset_name, context_attributes)

    def encode_entities_and_return_pooled_outputs(self, entity, excluded_attributes=None):
        """Fills the provided query table and delivers evidence if expected
            :param entity entity to be encoded
            :param excluded_attributes   Attributes, which will be excluded
        """

        logger = logging.getLogger()
        logger.warning('Method not implemented!')

        raise NotImplementedError('Method not implemented!')