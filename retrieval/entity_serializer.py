"""
This code is based on the code written by:
Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines.
In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer.

The code can be found at:
https://github.com/wbsg-uni-mannheim/SC-Block
"""


import torch.nn.functional as F
import pandas as pd
import numpy as np
import math

class EntitySerializer:
    def __init__(self, dataset_name, context_attributes=None):
        """Initialize the EntitySerializer with schema and context attributes.

        Args:
            dataset_name (str): The dataset name.
            context_attributes (list, optional): List of context attributes. If not provided, defaults are used.
        """
        self.dataset_name = dataset_name

        if dataset_name == 'walmart-amazon':
            self.context_attributes = ['name', 'category', 'brand', 'modelno', 'price']
        elif 'wdcproducts' in dataset_name:
            self.context_attributes = ['brand', 'name', 'price', 'pricecurrency', 'description']
        else:
            raise ValueError('Entity Serialization not defined for dataset name{}'.format(self.dataset_name))

    def convert_to_str_representation(self, entity, excluded_attributes=None, without_special_tokens=False):
        """Convert entity to string representation for text-based processing.

        Args:
            entity (dict): The entity to convert.
            excluded_attributes (list, optional): Attributes to exclude from the string representation.
            without_special_tokens (bool, optional): Whether to include special tokens.

        Returns:
            str: The string representation of the entity.
        """
        entity_str = ''
        selected_attributes = self.context_attributes

        if entity is None:
            raise ValueError('Entity must not be None!')

        if excluded_attributes is not None:
            selected_attributes = [attr for attr in self.context_attributes if attr not in excluded_attributes]

        for attr in selected_attributes:
            attribute_value = self.preprocess_attribute_value(entity, attr)
            if attr == 'description' and attribute_value is not None:
                attribute_value = attribute_value[:100]  # Limit description length
            if attribute_value is not None:
                if without_special_tokens:
                    entity_str = '{} {}'.format(entity_str, attribute_value)
                else:
                    entity_str = '{}[COL] {} [VAL] {} '.format(entity_str, attr, attribute_value)
            else:
                if without_special_tokens:
                    entity_str = '{}'.format(entity_str)
                else:
                    entity_str = '{}[COL] {} [VAL] '.format(entity_str, attr)

        return entity_str

    def preprocess_attribute_value(self, entity, attr):
        """Preprocess attribute values for consistent formatting.

        Args:
            entity (dict): The entity containing attributes.
            attr (str): The attribute to preprocess.

        Returns:
            str or None: The preprocessed attribute value, or None if not applicable.
        """
        attribute_value = None

        if entity is None:
            raise ValueError('Entity must not be None!')

        if attr in entity and len(str(entity[attr])) > 0 \
                and entity[attr] is not None and entity[attr] is not np.nan:
            if isinstance(entity[attr], list) and all(isinstance(element, str) for element in entity[attr]):
                attribute_value = ', '.join(entity[attr])
            elif isinstance(entity[attr], str):
                attribute_value = entity[attr]
            elif isinstance(entity[attr], (np.floating, float)):
                if not math.isnan(entity[attr]):
                    attribute_value = str(entity[attr])
            else:
                attribute_value = str(entity[attr])

        return attribute_value