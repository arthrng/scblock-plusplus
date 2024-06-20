
import torch.nn.functional as F
import random
import pandas as pd
import numpy as np
import math
np.random.seed(42)
import random
random.seed(42)


class EntitySerializer:
    def __init__(self, schema_org_class, context_attributes=None):
        self.schema_org_class = schema_org_class

        if self.schema_org_class == 'amazon-google':
            self.context_attributes = ['name', 'category', 'brand', 'modelno', 'price']
        elif schema_org_class == 'walmart-amazon':
            self.context_attributes = ['name', 'category', 'brand', 'modelno', 'price']
        elif 'wdcproducts' in schema_org_class:
            self.context_attributes = ['brand', 'name', 'price', 'pricecurrency', 'description']
        else:
            raise ValueError('Entity Serialization not defined for schema org class {}'.format(self.schema_org_class))

    def convert_to_str_representation(self, entity, excluded_attributes=None, without_special_tokens=False):
        """Convert to string representation of entity"""
        entity_str = ''
        selected_attributes = self.context_attributes

        if entity is None:
            raise ValueError('Entity must not be None!')

        if excluded_attributes is not None:
            selected_attributes = [attr for attr in self.context_attributes if attr not in excluded_attributes]

        for attr in selected_attributes:
            attribute_value = self.preprocess_attribute_value(entity, attr)
            if attr == 'description' and attribute_value is not None:
                attribute_value = attribute_value[:100]
            if attribute_value is not None:
                if without_special_tokens:
                    entity_str = '{} {}'.format(entity_str, attribute_value)
                else:
                    entity_str = '{}[COL] {} [VAL] {} '.format(entity_str, attr, attribute_value)
            if attribute_value is None:
                if without_special_tokens:
                    entity_str = '{}'.format(entity_str)
                else:
                    entity_str = '{}[COL] {} [VAL] '.format(entity_str, attr)

        return entity_str

    def preprocess_attribute_value(self, entity, attr):
        """Preprocess attribute values"""
        attribute_value = None

        if entity is None:
            raise ValueError('Entity must not be None!')

        if attr in entity and len(str(entity[attr])) > 0 \
                and entity[attr] is not None and entity[attr] is not np.nan:
            if type(entity[attr]) is list and all(type(element) is str for element in entity[attr]):
                attribute_value = ', '.join(entity[attr])
            elif type(entity[attr]) is str:
                attribute_value = entity[attr]
            elif isinstance(entity[attr], np.floating) or type(entity[attr]) is float:
                if not math.isnan(entity[attr]):
                    attribute_value = str(entity[attr])
            else:
                attribute_value = str(entity[attr])

        return attribute_value