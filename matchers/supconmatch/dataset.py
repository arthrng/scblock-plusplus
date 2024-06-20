from transformers import AutoTokenizer
from torch.utils.data import WeightedRandomSampler
import torch
import torch.nn.functional as F
import torch
import random
import pandas as pd
import math
import numpy as np
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
    
def serialize_sample_amazongoogle(sample, side):
    entity_serializer = EntitySerializer('amazon-google')
    dict_sample = sample.to_dict()
    dict_sample['name'] = dict_sample['{}_title'.format(side)]
    dict_sample['category'] = dict_sample['{}_category'.format(side)]
    dict_sample['brand'] = dict_sample['{}_brand'.format(side)]
    dict_sample['modelno'] = dict_sample['{}_modelno'.format(side)]
    dict_sample['price'] = dict_sample['{}_price'.format(side)]
    string = entity_serializer.convert_to_str_representation(dict_sample)

    return string
    
def serialize_sample_walmartamazon(sample, side):
    entity_serializer = EntitySerializer('walmart-amazon')
    dict_sample = sample.to_dict()
    dict_sample['name'] = dict_sample['{}_title'.format(side)]
    dict_sample['category'] = dict_sample['{}_category'.format(side)]
    dict_sample['brand'] = dict_sample['{}_brand'.format(side)]
    dict_sample['modelno'] = dict_sample['{}_modelno'.format(side)]
    dict_sample['price'] = dict_sample['{}_price'.format(side)]
    string = entity_serializer.convert_to_str_representation(dict_sample)

    return string

def serialize_sample_wdcproducts(sample, side):
    entity_serializer = EntitySerializer('wdcproducts')
    dict_sample = sample.to_dict()
    dict_sample['name'] = dict_sample['{}_title'.format(side)]
    dict_sample['brand'] = dict_sample['{}_brand'.format(side)]
    dict_sample['description'] = dict_sample['{}_description'.format(side)]
    dict_sample['price'] = dict_sample['{}_price'.format(side)]
    dict_sample['pricecurrency'] = dict_sample['{}_pricecurrency'.format(side)]
    string = entity_serializer.convert_to_str_representation(dict_sample)

    return string

class ContrastiveClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset, tokenizer='roberta-base', max_length=128):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset = dataset

        data = pd.read_json(path, lines=True)

        # if dataset == 'amazon-google':
        #     data['description_left'] = ''
        #     data['description_right'] = ''

        data = data.fillna('')
        data = data.reset_index(drop=True)
        data = self._prepare_data(data)

        self.encodings_left = self.tokenizer((list(data['features_left'])), truncation=True, padding=True) 
        self.encodings_right = self.tokenizer((list(data['features_right'])), truncation=True, padding=True) 
        self.labels = list(data['labels'])
        self.data = data

        # Add weights
        class_counts = np.bincount(self.labels)
        class_weights = 1. / class_counts

        # Compute weights for each sample
        self.sample_weights = np.array([class_weights[t] for t in self.labels])
        self.sample_weights = torch.from_numpy(self.sample_weights).float()

        # Create sampler
        self.sampler = WeightedRandomSampler(weights=self.sample_weights, 
                                            num_samples=len(self.sample_weights), 
                                            replacement=True)
        
    def get_sampler(self):
        return self.sampler

    def __getitem__(self, idx):
        example = self.data.loc[idx].copy()

        # Select a random pair of products from the training set
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings_left.items()}
        item2 = {key: torch.tensor(val[idx]) for key, val in self.encodings_right.items()}
        label = torch.tensor(self.labels[idx])

        return item, item2, label

    def __len__(self):
        return len(self.data)
    
    def _prepare_data(self, data):

        if self.dataset == 'amazon-google':
            data['features_left'] = data.apply(serialize_sample_amazongoogle, args=('left',), axis=1)
            data['features_right'] = data.apply(serialize_sample_amazongoogle, args=('right',), axis=1)
        elif self.dataset == 'walmart-amazon':
            data['features_left'] = data.apply(serialize_sample_walmartamazon, args=('left',), axis=1)
            data['features_right'] = data.apply(serialize_sample_walmartamazon, args=('right',), axis=1)
        elif 'wdcproducts' in self.dataset:
            data['features_left'] = data.apply(serialize_sample_wdcproducts, args=('left',), axis=1)
            data['features_right'] = data.apply(serialize_sample_wdcproducts, args=('right',), axis=1)
            
        data = data[['features_left', 'features_right', 'label']]
        data = data.rename(columns={'label': 'labels'})

        return data