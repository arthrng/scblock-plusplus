from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from typing import Optional
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch
import torch.nn.functional as F
import torch
import random
import pandas as pd
import numpy as np
import math
import pickle
import os
np.random.seed(42)
import random
random.seed(42)


class EntitySerializer:
    def __init__(self, schema_org_class, context_attributes=None):
        self.schema_org_class = schema_org_class

        if self.schema_org_class == 'amazon-google':
            self.context_attributes = ['manufacturer', 'name', 'price']
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
    
def serialize_sample_walmartamazon(sample):
    entity_serializer = EntitySerializer('walmart-amazon')
    dict_sample = sample.to_dict()
    dict_sample['name'] = dict_sample['title']
    string = entity_serializer.convert_to_str_representation(dict_sample)

    return string

def serialize_sample_amazongoogle(sample):
    entity_serializer = EntitySerializer('amazon-google')
    dict_sample = sample.to_dict()
    dict_sample['name'] = dict_sample['title']
    string = entity_serializer.convert_to_str_representation(dict_sample)

    return string

def serialize_sample_wdcproducts(sample):
    entity_serializer = EntitySerializer('wdcproducts')
    dict_sample = sample.to_dict()
    dict_sample['name'] = dict_sample['title']
    string = entity_serializer.convert_to_str_representation(dict_sample)

    return string

  

def delete_random_tokens(string_value):
    """Deletes a single token"""
    tokens = string_value.split()
    # Calculate number of to be removed tokens
    #num_remove = int(len(tokens)/10) + 1
    num_remove = 1
    for _ in range(num_remove):
        tokens.pop(random.randint(0, len(tokens)-1))

    return ' '.join(tokens)

class BarlowTwinsPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset, tokenizer='roberta-base', max_length=128):

        self.max_length = max_length
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))

        data = pd.read_pickle(path)
        data = data.reset_index(drop=True)
        data = data.fillna('')
        data = self._prepare_data(data)

        self.data = data
        self.encodings = self.tokenizer(list(data['features']), truncation=True, padding=True) 
        self.features = list(data['features'])

    def __getitem__(self, idx):
        # for every example in batch, return a duplication
        example = self.data.loc[idx].copy()
        pos = self.data.loc[idx].copy()

        # Randomly delete token
        example['features'] = delete_random_tokens(example['features'])
        pos['features'] = delete_random_tokens(pos['features'])

        return example['features'], pos['features']

    def __len__(self):
        return len(self.data)

    def _prepare_data(self, data):
        if self.dataset == 'amazon-google':
            data['features'] = data.apply(serialize_sample_amazongoogle, axis=1)
        elif self.dataset == 'walmart-amazon':
            data['features'] = data.apply(serialize_sample_walmartamazon, axis=1)
        elif 'wdcproducts' in self.dataset:
            data['features'] = data.apply(serialize_sample_wdcproducts, axis=1)

        data = data[['features']]

        return data

