"""
This code is based on the code written by:
Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines.
In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer.

The code can be found at:
https://github.com/wbsg-uni-mannheim/SC-Block
"""

from transformers import AutoTokenizer
from torch.utils.data import WeightedRandomSampler
import torch
import pandas as pd
import numpy as np

# Append the path for entity serializer import
import sys
sys.path.append('..')
from retrieval.entity_serializer import EntitySerializer

def serialize_sample_amazongoogle(sample, side):
    """
    Serialize sample for amazon-google dataset.

    :param sample: Pandas Series containing the sample data.
    :param side: Either 'left' or 'right' to denote the side of the comparison.
    :return: Serialized string representation of the sample.
    """
    entity_serializer = EntitySerializer('amazon-google')
    dict_sample = sample.to_dict()
    dict_sample['name'] = dict_sample[f'{side}_title']
    dict_sample['category'] = dict_sample[f'{side}_category']
    dict_sample['brand'] = dict_sample[f'{side}_brand']
    dict_sample['modelno'] = dict_sample[f'{side}_modelno']
    dict_sample['price'] = dict_sample[f'{side}_price']
    string = entity_serializer.convert_to_str_representation(dict_sample)
    return string

def serialize_sample_walmartamazon(sample, side):
    """
    Serialize sample for walmart-amazon dataset.

    :param sample: Pandas Series containing the sample data.
    :param side: Either 'left' or 'right' to denote the side of the comparison.
    :return: Serialized string representation of the sample.
    """
    entity_serializer = EntitySerializer('walmart-amazon')
    dict_sample = sample.to_dict()
    dict_sample['name'] = dict_sample[f'{side}_title']
    dict_sample['category'] = dict_sample[f'{side}_category']
    dict_sample['brand'] = dict_sample[f'{side}_brand']
    dict_sample['modelno'] = dict_sample[f'{side}_modelno']
    dict_sample['price'] = dict_sample[f'{side}_price']
    string = entity_serializer.convert_to_str_representation(dict_sample)
    return string

def serialize_sample_wdcproducts(sample, side):
    """
    Serialize sample for wdcproducts dataset.

    :param sample: Pandas Series containing the sample data.
    :param side: Either 'left' or 'right' to denote the side of the comparison.
    :return: Serialized string representation of the sample.
    """
    entity_serializer = EntitySerializer('wdcproducts')
    dict_sample = sample.to_dict()
    dict_sample['name'] = dict_sample[f'{side}_title']
    dict_sample['brand'] = dict_sample[f'{side}_brand']
    dict_sample['description'] = dict_sample[f'{side}_description']
    dict_sample['price'] = dict_sample[f'{side}_price']
    dict_sample['pricecurrency'] = dict_sample[f'{side}_pricecurrency']
    string = entity_serializer.convert_to_str_representation(dict_sample)
    return string

class ContrastiveClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset, tokenizer='roberta-base', max_length=128):
        """
        Initialize the dataset.

        :param path: Path to the dataset file.
        :param dataset: Name of the dataset.
        :param tokenizer: Tokenizer to be used.
        :param max_length: Maximum length for tokenization.
        """
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset = dataset

        data = pd.read_json(path, lines=True)
        data = data.fillna('')
        data = data.reset_index(drop=True)
        data = self._prepare_data(data)

        self.encodings_left = self.tokenizer(list(data['features_left']), truncation=True, padding=True)
        self.encodings_right = self.tokenizer(list(data['features_right']), truncation=True, padding=True)
        self.labels = list(data['labels'])
        self.data = data

        # Compute sample weights
        class_counts = np.bincount(self.labels)
        class_weights = 1. / class_counts
        self.sample_weights = np.array([class_weights[t] for t in self.labels])
        self.sample_weights = torch.from_numpy(self.sample_weights).float()

        # Create sampler
        self.sampler = WeightedRandomSampler(weights=self.sample_weights, 
                                             num_samples=len(self.sample_weights), 
                                             replacement=True)

    def get_sampler(self):
        """
        Get the sampler for weighted random sampling.

        :return: WeightedRandomSampler
        """
        return self.sampler

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        :param idx: Index of the item.
        :return: Tuple containing tokenized left and right samples and the label.
        """
        example = self.data.loc[idx].copy()
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings_left.items()}
        item2 = {key: torch.tensor(val[idx]) for key, val in self.encodings_right.items()}
        label = torch.tensor(self.labels[idx])
        return item, item2, label

    def __len__(self):
        """
        Get the length of the dataset.

        :return: Length of the dataset.
        """
        return len(self.data)

    def _prepare_data(self, data):
        """
        Prepare the data by serializing samples based on the dataset type.

        :param data: DataFrame containing the dataset.
        :return: Processed DataFrame with serialized features.
        """
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
