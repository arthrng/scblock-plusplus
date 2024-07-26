"""
Some of the code is based on: https://github.com/wbsg-uni-mannheim/SC-Block
"""

from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import itertools
import torch
import random
import pandas as pd
import numpy as np
import os
import sys

# Append the path for entity serializer import
sys.path.append('..')
from retrieval.entity_serializer import EntitySerializer

def serialize_sample(sample, dataset):
    """
    Serialize a sample using the EntitySerializer.

    Args:
        sample (pd.Series): The sample to be serialized.
        dataset (str): The dataset name.

    Returns:
        str: The serialized string representation of the sample.
    """
    entity_serializer = EntitySerializer(dataset)
    dict_sample = sample.to_dict()
    dict_sample['name'] = dict_sample['title']
    string = entity_serializer.convert_to_str_representation(dict_sample)
    return string

def assign_clusterid(identifier, cluster_id_dict, cluster_id_amount):
    """
    Assign a cluster ID to an identifier.

    Args:
        identifier (str): The identifier.
        cluster_id_dict (dict): The dictionary of cluster IDs.
        cluster_id_amount (int): The current amount of cluster IDs.

    Returns:
        int: The assigned cluster ID.
    """
    try:
        result = cluster_id_dict[identifier]
    except KeyError:
        result = cluster_id_amount
    return result

class RobertaFinetuneDataset(torch.utils.data.Dataset):
    """
    Dataset for fine-tuning Roberta models.
    """
    def __init__(self, entities_path, pairs_path, dataset, tokenizer='roberta-base', max_length=128):
        """
        Initialize the dataset.

        Args:
            entities_path (str): Path to the entities file.
            pairs_path (str): Path to the pairs file.
            dataset (str): The dataset name.
            tokenizer (str): The tokenizer name.
            max_length (int): Maximum length for tokenization.
        """
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset = dataset

        # Read and process the entities data
        data = pd.read_pickle(entities_path, compression='gzip')
        print(data)

        # Read and filter the training pairs
        train_data = pd.read_json(pairs_path, lines=True)
        train_data = train_data[train_data['label'] == 1]

        # Build connected components (clusters) using the training pairs
        bucket_list = []
        for i, row in train_data.iterrows():
            left = f'{row["left_id"]}'
            right = f'{row["right_id"]}'
            found = False
            for bucket in bucket_list:
                if left in bucket and row['label'] == 1:
                    bucket.add(right)
                    found = True
                    break
                elif right in bucket and row['label'] == 1:
                    bucket.add(left)
                    found = True
                    break
            if not found:
                bucket_list.append(set([left, right]))

        cluster_id_amount = len(bucket_list)

        # Assign cluster IDs to entities
        cluster_id_dict = {}
        for i, id_set in enumerate(bucket_list):
            for v in id_set:
                cluster_id_dict[v] = i

        data = data.set_index('id', drop=False)
        data['cluster_id'] = data['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))
        single_entities = data[data['cluster_id'] == cluster_id_amount].copy()

        # Assign labels to single entities
        index = single_entities.index
        if 'wdcproducts' in dataset:
            left_index = [x for x in index if 'left' in x]
            right_index = [x for x in index if 'right' in x]
        elif dataset == 'walmart-amazon':
            left_index = [x for x in index if 'walmart' in x]
            right_index = [x for x in index if 'amazon' in x]

        # Assign increasing integer labels to single entities
        single_entities = single_entities.reset_index(drop=True)
        single_entities['cluster_id'] = single_entities['cluster_id'] + single_entities.index
        single_entities = single_entities.set_index('id', drop=False)

        # Split the single entities
        single_entities_left = single_entities.loc[left_index]
        single_entities_right = single_entities.loc[right_index]

        # Source aware sampling, build one sample per dataset
        data1 = data.copy().drop(single_entities['id'])
        data1 = pd.concat([data1, single_entities_left])
        data1 = pd.concat([data1, single_entities_right])

        data1 = data1.reset_index(drop=True)

        # Encode labels
        label_enc = LabelEncoder()
        cluster_id_set = set(data1['cluster_id'])
        label_enc.fit(list(cluster_id_set))
        data1['labels'] = label_enc.transform(data1['cluster_id'])

        self.label_encoder = label_enc

        data1 = data1.reset_index(drop=True).fillna('')
        data1 = self._prepare_data(data1)
        data1 = data1.sample(n=len(data1))

        self.data1 = data1
        self.encodings = self.tokenizer(list(data1['features']), truncation=True, padding=True, max_length=max_length)
        self.features = list(data1['features'])
        self.labels = list(data1['labels'])
        self.ids = list(data1['id'])

    def __getitem__(self, idx):
        """
        Get a sample and its pair from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: The sample and its pair.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['id'] = self.ids[idx]

        selection = [i for i in [self.labels.index(self.labels[idx])]]
        idx2 = random.choice(selection)

        item2 = {key: torch.tensor(val[idx2]) for key, val in self.encodings.items()}
        item2['labels'] = torch.tensor(self.labels[idx])
        item2['id'] = self.ids[idx]

        return item, item2

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data1)

    def get_features(self):
        """
        Get the features, labels, and ids of the dataset.

        Returns:
            tuple: Features, labels, and ids.
        """
        return self.features, self.labels, self.ids

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
        data = data[['id', 'features', 'labels']]
        return data

class AuxiliaryFinetuneDataset(torch.utils.data.Dataset):
    """
    Dataset for auxiliary fine-tuning.
    """
    def __init__(self, features, labels, ids, num_auxiliary_model, num_auxiliary_models, dataset, tokenizer='roberta-base'):
        """
        Initialize the dataset.

        Args:
            features (list): List of features.
            labels (list): List of labels.
            ids (list): List of ids.
            num_auxiliary_model (int): Index of the auxiliary model.
            num_auxiliary_models (int): Total number of auxiliary models.
            dataset (str): The dataset name.
            tokenizer (str): The tokenizer name.
        """
        self.features = features
        self.labels = labels
        self.ids = ids

        # Split the data among auxiliary models
        self.features_split = np.array_split(np.array(self.features), num_auxiliary_models)
        self.labels_split = np.array_split(np.array(self.labels), num_auxiliary_models)
        self.ids_split = np.array_split(np.array(self.ids), num_auxiliary_models)

        # Prepare training set for the auxiliary network
        features_copy = self.features_split.copy()
        features_copy.pop(num_auxiliary_model)
        self.train_features = list(itertools.chain.from_iterable(features_copy))

        labels_copy = self.labels_split.copy()
        labels_copy.pop(num_auxiliary_model)
        self.train_labels = list(itertools.chain.from_iterable(labels_copy))

        ids_copy = self.ids_split.copy()
        ids_copy.pop(num_auxiliary_model)
        self.train_ids = list(itertools.chain.from_iterable(ids_copy))

        # Encode features
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.encodings = self.tokenizer(self.train_features, truncation=True, padding=True)

        # Prepare heldout set for determining the flooding levels
        self.heldout_features = list(self.features_split[num_auxiliary_model])
        self.heldout_labels = list(self.labels_split[num_auxiliary_model])
        self.heldout_ids = list(self.ids_split[num_auxiliary_model])

    def get_heldout_data(self):
        """
        Get the heldout data for evaluation.

        Returns:
            tuple: Heldout features, labels, and ids.
        """
        return self.heldout_features, self.heldout_labels, self.heldout_ids

    def __getitem__(self, idx):
        """
        Get a sample and its pair from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: The sample and its pair.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        selection = [i for i in [self.labels.index(self.labels[idx])]]
        idx2 = random.choice(selection)

        item2 = {key: torch.tensor(val[idx2]) for key, val in self.encodings.items()}
        item2['labels'] = torch.tensor(self.labels[idx])

        return item, item2

    def __len__(self):
        """
        Get the length of the training dataset.

        Returns:
            int: The length of the training dataset.
        """
        return len(self.train_features)
