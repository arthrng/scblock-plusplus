"""
This code is based on the code written by:
Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines.
In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer.

The code can be found at:
https://github.com/wbsg-uni-mannheim/SC-Block
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from torch.utils.data import WeightedRandomSampler

# Append the path for entity serializer import
import sys
sys.path.append('..')
from retrieval.entity_serializer import EntitySerializer

def serialize_sample_walmartamazon(sample, side):
    """
    Serialize a product sample from Walmart or Amazon into a string representation.

    Parameters:
    - sample (pd.Series): A single row from the dataset containing product information.
    - side (str): Indicates whether the sample is for 'left' or 'right'.

    Returns:
    - string: Serialized string representation of the product information.
    """
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
    """
    Serialize a product sample from WDC into a string representation.

    Parameters:
    - sample (pd.Series): A single row from the dataset containing product information.
    - side (str): Indicates whether the sample is for 'left' or 'right'.

    Returns:
    - string: Serialized string representation of the product information.
    """
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
    """
    A custom dataset class for handling data preparation and tokenization for training or evaluation.

    Attributes:
    - dataset (str): Name of the dataset source.
    - path (str): Path to the dataset file.
    - data (pd.DataFrame): Data to be used for the dataset.
    - tokenizer (AutoTokenizer): Tokenizer for encoding text inputs.
    - max_length (int): Maximum length of input sequences.
    - sample_weights (torch.Tensor): Weights for each sample to handle class imbalance.
    - sampler (WeightedRandomSampler): Sampler for weighted random sampling.
    - is_training (bool): Indicates if the dataset is used for training or evaluation.
    """

    def __init__(self, dataset=None, path=None, data=None, tokenizer='roberta-base', max_length=256):
        """
        Initialize the dataset object.

        Parameters:
        - dataset (str): Name of the dataset source.
        - path (str): Path to the dataset file.
        - data (pd.DataFrame): Optional data to use directly.
        - tokenizer (str): Pre-trained tokenizer model name.
        - max_length (int): Maximum length of input sequences.
        """
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]', '[SEP]', '[CLS]'))
        self.dataset = dataset

        if data is None:
            # Load and prepare data if not provided directly
            data = pd.read_json(path, lines=True)
            data = data.fillna('')
            data = data.reset_index(drop=True)
            data = self.prepare_data(data)

            self.labels = list(data['labels'])
            self.data = data

            # Compute class weights for handling class imbalance
            class_counts = np.bincount(self.labels)
            class_weights = 1. / class_counts
            self.sample_weights = np.array([class_weights[t] for t in self.labels])
            self.sample_weights = torch.from_numpy(self.sample_weights).float()

            # Create weighted random sampler
            self.sampler = WeightedRandomSampler(weights=self.sample_weights, 
                                                num_samples=len(self.sample_weights), 
                                                replacement=True)
            self.is_training = True
        else:
            self.data = data
            self.is_training = False

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - Tuple: Contains tokenized sequences and label (if training) or -1 (if evaluation).
        """
        pair = self.data.loc[idx].copy()
        
        if self.is_training:
            pair_label = torch.tensor(pair['labels'])
        
        left_str = pair['features_left']
        right_str = pair['features_right']

        # Tokenize the original strings
        x = self.tokenizer.encode(text=left_str,
                                  text_pair=right_str,
                                  max_length=self.max_length,
                                  truncation=True)

        # Concatenate the two strings for augmented tokenization
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

        # Remove tokens that are less than or equal to 4 characters
        new_tokens = []
        for token, label in zip(tokens, labels):
            if label != 'O' or len(token) > 4:
                new_tokens.append(token)
        
        # Reconstruct augmented string
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
        """
        Return the number of samples in the dataset.

        Returns:
        - int: Number of samples in the dataset.
        """
        return len(self.data)
    
    def get_sampler(self):
        """
        Retrieve the sampler for the dataset.

        Returns:
        - WeightedRandomSampler: The sampler used for training.
        """
        return self.sampler
    
    def prepare_data(self, data):
        """
        Prepare and serialize data based on the dataset type.

        Parameters:
        - data (pd.DataFrame): Raw data to be prepared.

        Returns:
        - pd.DataFrame: Prepared data with serialized features.
        """
        if 'wdcproducts' in self.dataset:
            data['features_left'] = data.apply(serialize_sample_wdc, args=('left',), axis=1)
            data['features_right'] = data.apply(serialize_sample_wdc, args=('right',), axis=1)
        elif self.dataset == 'walmart-amazon':
            data['features_left'] = data.apply(serialize_sample_walmartamazon, args=('left',), axis=1)
            data['features_right'] = data.apply(serialize_sample_walmartamazon, args=('right',), axis=1)
        print(data)
        data = data[['features_left', 'features_right', 'label']]
        data = data.rename(columns={'label': 'labels'})
        return data

    @staticmethod
    def pad(batch):
        """
        Pad and batch data items to ensure uniform sequence length.

        Parameters:
        - batch (list of tuple): List of data items to be padded.

        Returns:
        - Tuple: Padded tensors of sequences and labels.
        """
        if len(batch[0]) == 3:
            x1, x2, y = zip(*batch)

            maxlen = max([len(x) for x in x1 + x2])
            x1 = [xi + [0] * (maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0] * (maxlen - len(xi)) for xi in x2]
            return torch.LongTensor(x1), \
                   torch.LongTensor(x2), \
                   torch.LongTensor(y)
        else:
            x12, y = zip(*batch)
            maxlen = max([len(x) for x in x12])
            x12 = [xi + [0] * (maxlen - len(xi)) for xi in x12]
            return torch.LongTensor(x12), \
                   torch.LongTensor(y)
