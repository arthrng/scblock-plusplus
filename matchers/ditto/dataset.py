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
    def __init__(self, dataset_name, context_attributes=None):
        self.dataset_name = dataset_name

        if self.dataset_name == 'amazon-google':
            self.context_attributes = ['manufacturer', 'name', 'price']
        elif dataset_name == 'walmart-amazon':
            self.context_attributes = ['name', 'category', 'brand', 'modelno', 'price']
        elif 'wdcproducts' in dataset_name:
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

def serialize_sample_wdc(sample, side):
    entity_serializer = EntitySerializer('wdcproducts')
    dict_sample = sample.to_dict()
    dict_sample['name'] = dict_sample['{}_title'.format(side)]
    dict_sample['brand'] = dict_sample['{}_brand'.format(side)]
    dict_sample['description'] = dict_sample['{}_description'.format(side)]
    dict_sample['price'] = dict_sample['{}_price'.format(side)]
    dict_sample['pricecurrency'] = dict_sample['{}_pricecurrency'.format(side)]
    string = entity_serializer.convert_to_str_representation(dict_sample)

    return string

class DittoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset=None, path=None, data=None, tokenizer='roberta-base', max_length=256):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]', '[SEP]', '[CLS]'))
        self.dataset = dataset

        if data is None:
            data = pd.read_json(path, lines=True)
            data = data.fillna('')
            data = data.reset_index(drop=True)
            data = self.prepare_data(data)

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
            # Train boolean
            self.is_training = True
        else:
            self.data = data

            # Train boolean
            self.is_training = False
    def __getitem__(self, idx):
        # Select a random pair of products from the training set
        pair = self.data.loc[idx].copy()
        
        # Get label
        if self.is_training:
            pair_label = torch.tensor(pair['labels'])
            

        # Concatenate the two strings
        left_str = pair['features_left']
        right_str = pair['features_right']

        x = self.tokenizer.encode(text=left_str,
                                  text_pair=right_str,
                                  max_length=self.max_length,
                                  truncation=True)

        # Obtain tokens from the concatenated string
        concat_str = left_str + '[SEP] ' + right_str
        tokens = concat_str.split(' ')
        labels = []
        prev_token = None
        for token in tokens:
            if prev_token == '[COL]':
                labels.append('C')
            elif token not in ['[COL]', '[VAL]', '[SEP]', '[CLS]']:
                labels.append('O')
            else:
                labels.append(token)
            prev_token = token

        # Remove tokens that have at most 4 characters
        new_tokens = []
        for token, label in zip(tokens, labels):
            if label != 'O' or len(token) > 4:
                new_tokens.append(token)
        
        # Construct augmented string
        aug_str = ' '.join(new_tokens)
        left_str, right_str = aug_str.split(' [SEP] ')
        
        x_aug = self.tokenizer.encode(text=left_str,
                                  text_pair=right_str,
                                  max_length=self.max_length,
                                  truncation=True)
        if self.is_training:
            return x, x_aug, pair_label
        else:
            return x, -1

    def __len__(self):
        return len(self.data)
    
    def get_sampler(self):
        return self.sampler
    
    def prepare_data(self, data):
        if 'wdcproducts' in self.dataset:
            data['features_left'] = data.apply(serialize_sample_wdc, args=('left',), axis=1)
            data['features_right'] = data.apply(serialize_sample_wdc, args=('right',), axis=1)
        elif self.dataset == 'amazon-google':
            data['features_left'] = data.apply(serialize_sample_amazongoogle, args=('left',), axis=1)
            data['features_right'] = data.apply(serialize_sample_amazongoogle, args=('right',), axis=1)
        elif self.dataset == 'walmart-amazon':
            data['features_left'] = data.apply(serialize_sample_walmartamazon, args=('left',), axis=1)
            data['features_right'] = data.apply(serialize_sample_walmartamazon, args=('right',), axis=1)
        print(data)
        data = data[['features_left', 'features_right', 'label']]
        data = data.rename(columns={'label': 'labels'})
        return data

    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        if len(batch[0]) == 3:
            x1, x2, y = zip(*batch)

            maxlen = max([len(x) for x in x1+x2])
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
            return torch.LongTensor(x1), \
                   torch.LongTensor(x2), \
                   torch.LongTensor(y)
        else:
            x12, y = zip(*batch)
            maxlen = max([len(x) for x in x12])
            x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]
            return torch.LongTensor(x12), \
                   torch.LongTensor(y)