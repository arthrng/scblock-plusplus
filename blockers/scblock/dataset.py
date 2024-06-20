from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from typing import Optional
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import itertools
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
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

class RobertaFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, entities_path, pairs_path, dataset, tokenizer='roberta-base', max_length=128):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset = dataset

        # Read training data
        data = pd.read_pickle(entities_path, compression='gzip')
        print(data)
    
        # Read training pairs
        train_data = pd.read_json(pairs_path, lines=True)

        # Only acquire matching record pairs
        train_data = train_data[train_data['label'] == 1]
        
        # build the connected components by applying binning
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

        # Assign labels to connected nodes
        cluster_id_dict = {}
        for i, id_set in enumerate(bucket_list):
            for v in id_set:
                cluster_id_dict[v] = i

        data = data.set_index('id', drop=False)
        data['cluster_id'] = data['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))
        single_entities = data[data['cluster_id'] == cluster_id_amount].copy()

        # Assign labels to single nodes
        index = single_entities.index
        if 'wdcproducts' in dataset:
            left_index = [x for x in index if 'left' in x]
            right_index = [x for x in index if 'right' in x]
        elif dataset == 'amazon-google':
            left_index = [x for x in index if 'amazon' in x]
            right_index = [x for x in index if 'google' in x]
        elif dataset == 'walmart-amazon':
            left_index = [x for x in index if 'walmart' in x]
            right_index = [x for x in index if 'amazon' in x]

        # assing increasing integer label to single nodes
        single_entities = single_entities.reset_index(drop=True)
        single_entities['cluster_id'] = single_entities['cluster_id'] + single_entities.index
        single_entities = single_entities.set_index('id', drop=False)

        # Split the single nodes
        single_entities_left = single_entities.loc[left_index]
        single_entities_right = single_entities.loc[right_index]

        # source aware sampling, build one sample per dataset
        data1 = data.copy().drop(single_entities['id'])
        data1 = pd.concat([data1, single_entities_left])
        data1 = pd.concat([data1, single_entities_right])

        data1 = data1.reset_index(drop=True)

        label_enc = LabelEncoder()
        cluster_id_set = set()
        cluster_id_set.update(data1['cluster_id'])
        label_enc.fit(list(cluster_id_set))
        data1['labels'] = label_enc.transform(data1['cluster_id'])

        self.label_encoder = label_enc
                
        data1 = data1.reset_index(drop=True)
        data1 = data1.fillna('')
        data1 = self._prepare_data(data1)
        data1 = data1.sample(n=len(data1))

        self.data1 = data1
        self.encodings = self.tokenizer(list(data1['features']), truncation=True, padding=True, max_length=max_length) 
        self.features = list(data1['features'])
        self.labels = list(data1['labels'])
        self.ids = list(data1['id'])

    def __getitem__(self, idx):
        # Select a random product from the training set
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['id'] = self.ids[idx]

        # Select another product that matches with the first product offer
        selection = [i for i in [self.labels.index(self.labels[idx])]]
        idx2 = random.choice(selection)

        item2 = {key: torch.tensor(val[idx2]) for key, val in self.encodings.items()}
        item2['labels'] = torch.tensor(self.labels[idx])
        item2['id'] = self.ids[idx]

        # Return both items
        return item, item2
    
    def __len__(self):
        return len(self.data1)
    
    def get_features(self):
        return self.features, self.labels, self.ids
    
    def _prepare_data(self, data):
        if self.dataset == 'walmart-amazon':
            data['features'] = data.apply(serialize_sample_walmartamazon, axis=1)

        elif self.dataset == 'amazon-google':
            data['features'] = data.apply(serialize_sample_amazongoogle, axis=1)

        elif 'wdcproducts' in self.dataset:
            data['features'] = data.apply(serialize_sample_wdcproducts, axis=1)

        data = data[['id', 'features', 'labels']]

        return data

class AuxiliaryFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, ids, num_auxiliary_model, num_auxiliary_models, dataset, tokenizer='roberta-base'):

        self.features = features
        self.labels = labels
        self.ids = ids

        # Split the data
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

        # Prepare heldout set for which we determine the flooding levels
        self.heldout_features = list(self.features_split[num_auxiliary_model])
        self.heldout_labels = list(self.labels_split[num_auxiliary_model])
        self.heldout_ids = list(self.ids_split[num_auxiliary_model])
    
    def get_heldout_data(self):
        return self.heldout_features, self.heldout_labels, self.heldout_ids

    def __getitem__(self, idx):    
        # Select a random product from the training set
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        # Select another product that matches with the first product offer
        selection = [i for i in [self.labels.index(self.labels[idx])]]
        idx2 = random.choice(selection)

        item2 = {key: torch.tensor(val[idx2]) for key, val in self.encodings.items()}
        item2['labels'] = torch.tensor(self.labels[idx])

        # Return both items
        return item, item2
    
    def __len__(self):
        return len(self.train_features)
    
def assign_clusterid(identifier, cluster_id_dict, cluster_id_amount):
    try:
        result = cluster_id_dict[identifier]
    except KeyError:
        result = cluster_id_amount
    return result

def preprocess_dataset(name, left_name, right_name):
    # Get parent directory
    parent_directory = '/content/drive/MyDrive/Master/Thesis/'  #os.path.dirname(os.getcwd()).replace('\\', '/')

    if name in ['walmart-amazon', 'amazon-google']:
        # Read the left and right data sets
        left_df = pd.read_csv(f'{parent_directory}/src/data/raw/{name}/tableA.csv', engine='python')
        right_df = pd.read_csv(f'{parent_directory}/src/data/raw/{name}/tableB.csv', engine='python')
        
        # Read the training, validation and test sets
        train_df = pd.read_csv(f'{parent_directory}/src/data/raw/{name}/train.csv')
        test_df = pd.read_csv(f'{parent_directory}/src/data/raw/{name}/test.csv')
        val_df = pd.read_csv(f'{parent_directory}/src/data/raw/{name}/valid.csv')

        # Convert product ID's to a string
        left_df['id'] = f'{left_name}_' + left_df['id'].astype(str)
        right_df['id'] = f'{right_name}_' + right_df['id'].astype(str)
        
        # Set ID as the index of the dataframes
        left_df = left_df.set_index('id', drop=False)
        right_df = right_df.set_index('id', drop=False)
        left_df = left_df.fillna('')
        right_df = right_df.fillna('')

        # Combine training, validation and test set
        full_df = pd.concat((train_df, pd.concat((val_df, test_df))), ignore_index=True)
        print(full_df)
        full_df = full_df[full_df['label'] == 1]

        full_df['ltable_id'] = f'{left_name}_' + full_df['ltable_id'].astype(str)
        full_df['rtable_id'] = f'{right_name}_' + full_df['rtable_id'].astype(str)
        
        # Assign the products to clusters
        bucket_list = []
        for i, row in full_df.iterrows():
            left = f'{row["ltable_id"]}'
            right = f'{row["rtable_id"]}'
            found_in_bucket = False
            for bucket in bucket_list:
                if left in bucket and row['label'] == 1:
                    bucket.add(right)
                    found_in_bucket = True
                    break
                elif right in bucket and row['label'] == 1:
                    bucket.add(left)
                    found_in_bucket = True
                    break
            if not found_in_bucket:
                bucket_list.append(set([left, right]))

        cluster_id_dict = {}
        
        for i, id_set in enumerate(bucket_list):
            for v in id_set:
                cluster_id_dict[v] = i

        # Convert product ID's to a string
        train_df['ltable_id'] = f'{left_name}_' + train_df['ltable_id'].astype(str)
        train_df['rtable_id'] = f'{right_name}_' + train_df['rtable_id'].astype(str)

        test_df['ltable_id'] = f'{left_name}_' + test_df['ltable_id'].astype(str)
        test_df['rtable_id'] = f'{right_name}_' + test_df['rtable_id'].astype(str)
                            
        val_df['ltable_id'] = f'{left_name}_' + val_df['ltable_id'].astype(str)
        val_df['rtable_id'] = f'{right_name}_' + val_df['rtable_id'].astype(str)

        train_df['label'] = train_df['label'].apply(lambda x: int(x))
        test_df['label'] = test_df['label'].apply(lambda x: int(x))
        val_df['label'] = val_df['label'].apply(lambda x: int(x))

        #valid['pair_id'] = valid['ltable_id'] + '#' + valid['rtable_id']
        # Combine the training set and validation set
        #train_df = pd.concat((train_df, val_df), ignore_index=True)

        # Separate the left and right products in the training set
        train_left = left_df.loc[list(train_df['ltable_id'].values)]
        train_right = right_df.loc[list(train_df['rtable_id'].values)]
        train_labels = [int(x) for x in list(train_df['label'].values)]

        # Separate the left and right products in the validation set
        val_left = left_df.loc[list(val_df['ltable_id'].values)]
        val_right = right_df.loc[list(val_df['rtable_id'].values)]
        val_labels = [int(x) for x in list(val_df['label'].values)]

        # Separate the left and right products in the test set
        test_left = left_df.loc[list(test_df['ltable_id'].values)]
        test_right = right_df.loc[list(test_df['rtable_id'].values)]
        test_labels = [int(x) for x in list(test_df['label'].values)]

        train_left = train_left.reset_index(drop=True)
        train_right = train_right.reset_index(drop=True)

        val_left = val_left.reset_index(drop=True)
        val_right = val_right.reset_index(drop=True)
        
        test_left = test_left.reset_index(drop=True)
        test_right = test_right.reset_index(drop=True)
        
        # Store cluster number for each product
        cluster_id_amount = len(bucket_list)
        
        train_left['cluster_id'] = train_left['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))
        train_right['cluster_id'] = train_right['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))
        val_left['cluster_id'] = val_left['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))
        val_right['cluster_id'] = val_right['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))
        test_left['cluster_id'] = test_left['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))
        test_right['cluster_id'] = test_right['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))
        
        # Join the left products and the right products in the training set
        train_df = train_left.add_prefix('left_').join(train_right.add_prefix('right_'))
        train_df['label'] = train_labels

        # Join the left products and the right products in the validation set
        val_df = val_left.add_prefix('left_').join(val_right.add_prefix('right_'))
        val_df['label'] = val_labels

        # Join the left products and the right products in the test set
        test_df = test_left.add_prefix('left_').join(test_right.add_prefix('right_'))
        test_df['label'] = test_labels

        # Store the data sets
        os.makedirs(os.path.dirname(f'{parent_directory}/src/data/interim/'), exist_ok=True)
        train_df.to_json(f'{parent_directory}/src/data/interim/{name}-train.json.gz', compression='gzip', lines=True, orient='records')
        val_df.to_json(f'{parent_directory}/src/data/interim/{name}-val.json.gz', compression='gzip', lines=True, orient='records')
        test_df.to_json(f'{parent_directory}/src/data/interim/{name}-test.json.gz', compression='gzip', lines=True, orient='records')

        # Get set with just ID's
        merged_ids = set()
        merged_ids.update(train_df['left_id'])
        merged_ids.update(train_df['right_id']) 
        
        entity_set = left_df[left_df['id'].isin(merged_ids)]
        entity_set = pd.concat((entity_set, right_df[right_df['id'].isin(merged_ids)]))

        # In next line all connected components are assigned the same label
        # Note, that all single nodes are assigned the same label here
        entity_set['cluster_id'] = entity_set['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))

        # assign increasing integer label to single nodes
        single_entities = entity_set[entity_set['cluster_id'] == cluster_id_amount].copy()
        single_entities = single_entities.reset_index(drop=True)
        single_entities['cluster_id'] = single_entities['cluster_id'] + single_entities.index
        
        entity_set = entity_set.drop(single_entities['id'])
        entity_set = pd.concat((entity_set, single_entities))
        entity_set = entity_set.reset_index(drop=True)

        print(f'Amount of entity descriptions: {len(entity_set)}')
        print(f'Amount of clusters: {len(entity_set["cluster_id"].unique())}')

        os.makedirs(os.path.dirname(f'{parent_directory}/src/data/processed/'), exist_ok=True)
        entity_set.to_pickle(f'{parent_directory}/src/data/processed/{name}-train.pkl.gz', compression='gzip')
    else:
        pass

    print(f'FINISHED BULDING {name} DATASETS\n')


if __name__ == '__main__':
    preprocess_dataset('amazon-google', 'amazon', 'google')