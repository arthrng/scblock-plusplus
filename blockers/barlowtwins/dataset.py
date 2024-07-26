from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import random
import pandas as pd
import sys

# Append the path for entity serializer import
sys.path.append('..')
from retrieval.entity_serializer import EntitySerializer

def serialize_sample(sample, dataset):
    """
    Serialize a sample from the dataset using EntitySerializer.

    Args:
        sample (pd.Series): A sample from the dataset.
        dataset (str): The name of the dataset.

    Returns:
        str: Serialized string representation of the sample.
    """
    entity_serializer = EntitySerializer(dataset)
    dict_sample = sample.to_dict()
    dict_sample['name'] = dict_sample['title']
    return entity_serializer.convert_to_str_representation(dict_sample)

def delete_random_tokens(string_value):
    """
    Randomly delete a single token from the input string.

    Args:
        string_value (str): The input string.

    Returns:
        str: The string with a randomly deleted token.
    """
    tokens = string_value.split()
    if tokens:  # Ensure there is at least one token to pop
        tokens.pop(random.randint(0, len(tokens) - 1))
    return ' '.join(tokens)

class BarlowTwinsPretrainDataset(torch.utils.data.Dataset):
    """
    Dataset class for Barlow Twins pretraining.

    Args:
        path (str): Path to the dataset file.
        dataset (str): Name of the dataset.
        tokenizer (str): Name of the tokenizer to use. Default is 'roberta-base'.
        max_length (int): Maximum sequence length for tokenization. Default is 128.
    """
    def __init__(self, path, dataset, tokenizer='roberta-base', max_length=128):
        self.max_length = max_length
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=['[COL]', '[VAL]'])

        # Load and prepare data
        data = pd.read_pickle(path).reset_index(drop=True).fillna('')
        self.data = self._prepare_data(data)

        # Tokenize the data
        self.encodings = self.tokenizer(list(self.data['features']), truncation=True, padding=True)
        self.features = list(self.data['features'])

    def __getitem__(self, idx):
        """
        Get a data sample and its positive pair with a random token deleted.

        Args:
            idx (int): Index of the data sample.

        Returns:
            tuple: A tuple containing the original and positive pair strings.
        """
        example = self.data.iloc[idx].copy()
        pos = self.data.iloc[idx].copy()

        # Delete a random token from both the original and positive examples
        example['features'] = delete_random_tokens(example['features'])
        pos['features'] = delete_random_tokens(pos['features'])

        return example['features'], pos['features']

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)

    def _prepare_data(self, data):
        """
        Prepare and serialize the data.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The prepared data with serialized features.
        """
        if 'wdcproducts' in self.dataset:
            data['features'] = data.apply(serialize_sample, dataset='wdcproducts', axis=1)
        else:
            data['features'] = data.apply(serialize_sample, dataset=self.dataset, axis=1)
        return data[['features']]
