import pandas as pd
import os
import random
from tqdm import tqdm
    
def assign_cluster_id(identifier, cluster_id_dict, cluster_id_amount):
    try:
        result = cluster_id_dict[identifier]
    except KeyError:
        result = cluster_id_amount
    return result

def preprocess_dataset(dataset_name, left_name, right_name):
    # Get parent directory
    parent_directory = os.path.dirname(os.getcwd()).replace('\\', '/') #'/content/drive/MyDrive/Master/Thesis/' 

    # Read the left and right data sets
    left_df = pd.read_csv(f'{parent_directory}/src/data/raw/{dataset_name}/tableA.csv', engine='python')
    right_df = pd.read_csv(f'{parent_directory}/src/data/raw/{dataset_name}/tableB.csv', engine='python')
    
    # Read the training, validation and test sets
    train_df = pd.read_csv(f'{parent_directory}/src/data/raw/{dataset_name}/train.csv')
    test_df = pd.read_csv(f'{parent_directory}/src/data/raw/{dataset_name}/test.csv')
    val_df = pd.read_csv(f'{parent_directory}/src/data/raw/{dataset_name}/valid.csv')

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
    
    # Convert the training, validation and test set to the DeepMatcher format, which we can use to train the models
    train_df = convert_to_pairs_format(df=train_df, 
                                             split='train', 
                                             left_df=left_df, 
                                             right_df=right_df, 
                                             parent_directory=parent_directory, 
                                             dataset_name=dataset_name, 
                                             left_name=left_name, 
                                             right_name=right_name,
                                             cluster_id_dict=cluster_id_dict, 
                                             cluster_id_amount=len(bucket_list))
    val_df = convert_to_pairs_format(df=val_df, 
                                           split='val', 
                                           left_df=left_df, 
                                           right_df=right_df, 
                                           parent_directory=parent_directory, 
                                           dataset_name=dataset_name, 
                                           left_name=left_name, 
                                           right_name=right_name,
                                           cluster_id_dict=cluster_id_dict, 
                                           cluster_id_amount=len(bucket_list))
    test_df = convert_to_pairs_format(df=test_df, 
                                            split='test', 
                                            left_df=left_df, 
                                            right_df=right_df, 
                                            parent_directory=parent_directory, 
                                            dataset_name=dataset_name, 
                                            left_name=left_name, 
                                            right_name=right_name,
                                            cluster_id_dict=cluster_id_dict, 
                                            cluster_id_amount=len(bucket_list))
    
    # Convert the training, validation and test set to a format with all the products
    dataframes = {'train' : train_df,
                  'val' : val_df,
                  'test' : test_df}
    for split, df in dataframes.items():
        convert_to_entity_format(df=df,
                                 split=split,
                                 left_df=left_df,
                                 right_df=right_df,
                                 parent_directory=parent_directory,
                                 dataset_name=dataset_name,
                                 cluster_id_dict=cluster_id_dict,
                                 cluster_id_amount=len(bucket_list))


def convert_to_pairs_format(df, split, left_df, right_df, parent_directory, dataset_name, left_name, right_name, cluster_id_dict, cluster_id_amount):
    # Convert product ID's to a string
    df['ltable_id'] = f'{left_name}_' + df['ltable_id'].astype(str)
    df['rtable_id'] = f'{right_name}_' + df['rtable_id'].astype(str)

    # Add binary labels to the pairs
    df['label'] = df['label'].apply(lambda x: int(x))

    # Separate the left and right products in the dataframe of the split
    split_left = left_df.loc[list(df['ltable_id'].values)]
    split_right = right_df.loc[list(df['rtable_id'].values)]
    split_labels = [int(x) for x in list(df['label'].values)]

    split_left = split_left.reset_index(drop=True)
    split_right = split_right.reset_index(drop=True)
    
    # Assign cluster ID's to each product 
    split_left['cluster_id'] = split_left['id'].apply(assign_cluster_id, args=(cluster_id_dict, cluster_id_amount))
    split_right['cluster_id'] = split_right['id'].apply(assign_cluster_id, args=(cluster_id_dict, cluster_id_amount))
    
    # Join the left products and the right products in the set
    df = split_left.add_prefix('left_').join(split_right.add_prefix('right_'))
    df['label'] = split_labels

    # Store the data sets
    os.makedirs(os.path.dirname(f'{parent_directory}/src/data/processed/'), exist_ok=True)
    df.to_json(f'{parent_directory}/src/data/processed/{dataset_name}-{split}-pairs.json.gz', 
               compression='gzip', 
               lines=True, 
               orient='records')

    # Return dataframe
    return df

def convert_to_entity_format(df, split, left_df, right_df, parent_directory, dataset_name, cluster_id_dict, cluster_id_amount):
    # Get set with just ID's of the products
    merged_ids = set()
    merged_ids.update(df['left_id'])
    merged_ids.update(df['right_id']) 
    
    # Intiliaze entity set
    entity_set = left_df[left_df['id'].isin(merged_ids)]
    entity_set = pd.concat((entity_set, right_df[right_df['id'].isin(merged_ids)]))

    # Assign all connected components the same label
    entity_set['cluster_id'] = entity_set['id'].apply(assign_cluster_id, args=(cluster_id_dict, cluster_id_amount))

    # Assign increasing integer label to single nodes
    single_entities = entity_set[entity_set['cluster_id'] == cluster_id_amount].copy()
    single_entities = single_entities.reset_index(drop=True)
    single_entities['cluster_id'] = single_entities['cluster_id'] + single_entities.index
    
    print(len(single_entities))
    entity_set = entity_set.drop(single_entities['id'])
    entity_set = pd.concat((entity_set, single_entities))
    entity_set = entity_set.reset_index(drop=True)

    print(f'Amount of entity descriptions: {len(entity_set)}')
    print(f'Amount of clusters: {len(entity_set["cluster_id"].unique())}')

    # Store the entity set
    os.makedirs(os.path.dirname(f'{parent_directory}/src/data/processed/'), exist_ok=True)
    entity_set.to_pickle(f'{parent_directory}/src/data/processed/{dataset_name}-{split}-entities.pkl.gz', compression='gzip')

def convert_wdcproducts_to_deepmatcher(corner_case_perc):
    # Get parent directory
    parent_directory = os.path.dirname(os.getcwd()).replace('\\', '/') #'/content/drive/MyDrive/Master/Thesis/'
    
    # Initialize splits
    splits = ['train', 'valid', 'test']
    ds_sets = {}
    for split in splits:
        # Initialize path to raw data file
        if split == 'test':
            path = f'{parent_directory}/src/data/raw/wdcproducts/{corner_case_perc}pair/wdcproducts{corner_case_perc}cc{100-corner_case_perc}rnd050un_gs.json.gz'
        else:
            path = f'{parent_directory}/src/data/raw/wdcproducts/{corner_case_perc}pair/wdcproducts{corner_case_perc}cc{100-corner_case_perc}rnd000un_{split}_small.json.gz'

        # Import data
        df_split = pd.read_json(path, lines=True)
        ds_sets[split] = df_split.sort_values(by=['label'], ascending=False)

    # Determine unique cluster ids - all clusters appear left and right
    cluster_ids = [ds_sets[split]['cluster_id_left'].unique() for split in ds_sets]
    cluster_ids = set([cluster_id for cluster_id_list in cluster_ids for cluster_id in cluster_id_list])
    not_test_cluster_ids = [ds_sets[split]['cluster_id_left'].unique() for split in ds_sets if split in ['train', 'valid']]
    not_test_cluster_ids = set([cluster_id for cluster_id_list in not_test_cluster_ids for cluster_id in cluster_id_list])

    unique_ids = set()
    for split in ds_sets:
        unique_ids.update(ds_sets[split]['id_left'].unique())

    # Determine left offer ids per cluster id
    for cluster_id in tqdm(cluster_ids):
        # Get Offer ids for train & valid
        offer_ids = [ds_sets[split].loc[(ds_sets[split]['cluster_id_left'] == cluster_id)]['id_left'].unique()
                        for split in ds_sets]
        offer_ids = list(set([offer_id for offer_id_list in offer_ids for offer_id in offer_id_list]))
        left_offer_id = offer_ids.pop()  # Leading offer id

        for split in ds_sets:
            # Replace left hand side offer ids by leading offer id
            for offer_id in offer_ids:
                ds_sets[split].loc[ds_sets[split]['id_left'] == offer_id, 'id_left'] = left_offer_id

            # Replace right hand side leading offer ids with random offer id from the same cluster
            if len(list(offer_ids)) > 0:
                ds_sets[split].loc[ds_sets[split]['id_right'] == left_offer_id, 'id_right'] = random.choice(list(offer_ids))
                ds_sets[split] = ds_sets[split].drop_duplicates()

    # Remove pairs where both records are supposed to be in the query table
    unique_left_ids = set()
    for split in ds_sets:
        unique_left_ids.update(ds_sets[split]['id_left'].unique())

    for split in ds_sets:
        ds_sets[split] = ds_sets[split].loc[~ds_sets[split]['id_right'].isin(unique_left_ids)]


    #unique_ids = set()
    for cluster_id in tqdm(cluster_ids):
        offer_ids = [ds_sets[split].loc[(ds_sets[split]['cluster_id_left'] == cluster_id) & (
                    ds_sets[split]['cluster_id_right'] == cluster_id)]['id_left'].unique() for split in ds_sets]
        offer_ids = set([offer_id for offer_id_list in offer_ids for offer_id in offer_id_list])
        if len(offer_ids) > 1:
            print('I am here!')
        elif len(offer_ids) == 0:
            print('Now I am here!')

    table_A_records = []
    table_B_records = []
    split_info = {'train': [], 'valid': [], 'test': []}
    for split in splits:
        for index, row in ds_sets[split].iterrows():
            left_record = {'id': row['id_left'], 'brand': row['brand_left'], 'title': row['title_left'],
                            'description': row['description_left'], 'price': row['price_left'],
                            'pricecurrency': row['priceCurrency_left'],
                            'cluster_id': row['cluster_id_left']}
            table_A_records.append(left_record)

            right_record = {'id': row['id_right'],'brand': row['brand_right'], 'title': row['title_right'],
                            'description': row['description_right'],'price': row['price_right'],
                            'pricecurrency': row['priceCurrency_right'],
                            'cluster_id': row['cluster_id_right']}
            table_B_records.append(right_record)

            matching_info = {'ltable_id': row['id_left'], 'rtable_id': row['id_right'], 'label': row['label'], 'cluster_id': row['cluster_id_left']}
            split_info[split].append(matching_info)

    # Create Data Frames
    df_table_a = pd.DataFrame(table_A_records)
    df_table_b = pd.DataFrame(table_B_records)

    # #Add matches to train/ validation to make sure that each cluster is seen at least once.
    df_train = pd.DataFrame(split_info['train']).drop(columns=['cluster_id'])
    df_valid = pd.DataFrame(split_info['valid']).drop(columns=['cluster_id'])
    df_test = pd.DataFrame(split_info['test']).drop(columns=['cluster_id'])

    # Drop duplicates from data tables
    df_table_a = df_table_a.drop_duplicates(subset=['id'])
    df_table_b = df_table_b.drop_duplicates(subset=['id'])

    # Save Data Frames
    path = f'{parent_directory}/src/data/raw/wdcproducts{corner_case_perc}pair'

    if not os.path.exists(path):
        os.makedirs(path)

    df_table_a = df_table_a.set_index('id')
    df_table_a.to_csv(path_or_buf='{}/tableA.csv'.format(path), sep=',')

    df_table_b = df_table_b.set_index('id')
    df_table_b.to_csv(path_or_buf='{}/tableB.csv'.format(path), sep=',')

    df_train.to_csv(path_or_buf='{}/train.csv'.format(path), sep=',', index=False)
    df_valid.to_csv(path_or_buf='{}/valid.csv'.format(path), sep=',', index=False)
    df_test.to_csv(path_or_buf='{}/test.csv'.format(path), sep=',', index=False)

if __name__ == '__main__':
    # Set seed
    random.seed(42)

    # Convert WDC-B to DeepMatcher format
    corner_case_percs = [20, 80]
    for corner_case_perc in corner_case_percs:
        convert_wdcproducts_to_deepmatcher(corner_case_perc)

    # Initialize dataset names
    dataset_names = {'amazon-google' : ['amazon', 'google'],
                     'walmart-amazon' : ['walmart', 'amazon'],
                     #'wdcproducts20pair' : ['left', 'right'],
                     #'wdcproducts80pai278r' : ['left', 'right']
                     }
    for dataset_name, seperated_dataset_name in dataset_names.items():
        preprocess_dataset(dataset_name, seperated_dataset_name[0], seperated_dataset_name[1])


#  # Convert product ID's to a string
#     train_df['ltable_id'] = f'{left_name}_' + train_df['ltable_id'].astype(str)
#     train_df['rtable_id'] = f'{right_name}_' + train_df['rtable_id'].astype(str)

#     test_df['ltable_id'] = f'{left_name}_' + test_df['ltable_id'].astype(str)
#     test_df['rtable_id'] = f'{right_name}_' + test_df['rtable_id'].astype(str)
                        
#     val_df['ltable_id'] = f'{left_name}_' + val_df['ltable_id'].astype(str)
#     val_df['rtable_id'] = f'{right_name}_' + val_df['rtable_id'].astype(str)

#     train_df['label'] = train_df['label'].apply(lambda x: int(x))
#     test_df['label'] = test_df['label'].apply(lambda x: int(x))
#     val_df['label'] = val_df['label'].apply(lambda x: int(x))

#     #valid['pair_id'] = valid['ltable_id'] + '#' + valid['rtable_id']
#     # Combine the training set and validation set
#     #train_df = pd.concat((train_df, val_df), ignore_index=True)

#     # Separate the left and right products in the training set
#     train_left = left_df.loc[list(train_df['ltable_id'].values)]
#     train_right = right_df.loc[list(train_df['rtable_id'].values)]
#     train_labels = [int(x) for x in list(train_df['label'].values)]

#     # Separate the left and right products in the validation set
#     val_left = left_df.loc[list(val_df['ltable_id'].values)]
#     val_right = right_df.loc[list(val_df['rtable_id'].values)]
#     val_labels = [int(x) for x in list(val_df['label'].values)]

#     # Separate the left and right products in the test set
#     test_left = left_df.loc[list(test_df['ltable_id'].values)]
#     test_right = right_df.loc[list(test_df['rtable_id'].values)]
#     test_labels = [int(x) for x in list(test_df['label'].values)]

#     train_left = train_left.reset_index(drop=True)
#     train_right = train_right.reset_index(drop=True)

#     val_left = val_left.reset_index(drop=True)
#     val_right = val_right.reset_index(drop=True)
    
#     test_left = test_left.reset_index(drop=True)
#     test_right = test_right.reset_index(drop=True)
    
#     # Store cluster number for each product
#     cluster_id_amount = len(bucket_list)
    
#     train_left['cluster_id'] = train_left['id'].apply(assign_cluster_id, args=(cluster_id_dict, cluster_id_amount))
#     train_right['cluster_id'] = train_right['id'].apply(assign_cluster_id, args=(cluster_id_dict, cluster_id_amount))
#     val_left['cluster_id'] = val_left['id'].apply(assign_cluster_id, args=(cluster_id_dict, cluster_id_amount))
#     val_right['cluster_id'] = val_right['id'].apply(assign_cluster_id, args=(cluster_id_dict, cluster_id_amount))
#     test_left['cluster_id'] = test_left['id'].apply(assign_cluster_id, args=(cluster_id_dict, cluster_id_amount))
#     test_right['cluster_id'] = test_right['id'].apply(assign_cluster_id, args=(cluster_id_dict, cluster_id_amount))
    
#     # Join the left products and the right products in the training set
#     train_df = train_left.add_prefix('left_').join(train_right.add_prefix('right_'))
#     train_df['label'] = train_labels

#     # Join the left products and the right products in the validation set
#     val_df = val_left.add_prefix('left_').join(val_right.add_prefix('right_'))
#     val_df['label'] = val_labels

#     # Join the left products and the right products in the test set
#     test_df = test_left.add_prefix('left_').join(test_right.add_prefix('right_'))
#     test_df['label'] = test_labels

#     # Store the data sets
#     os.makedirs(os.path.dirname(f'{parent_directory}/src/data/processed/'), exist_ok=True)
#     train_df.to_json(f'{parent_directory}/src/data/processed/{dataset_name}-train-pairs.json.gz', 
#                         compression='gzip', 
#                         lines=True, 
#                         orient='records')
#     val_df.to_json(f'{parent_directory}/src/data/processed/{dataset_name}-val-pairs.json.gz', 
#                     compression='gzip', 
#                     lines=True, 
#                     orient='records')
#     test_df.to_json(f'{parent_directory}/src/data/processed/{dataset_name}-test-pairs.json.gz', 
#                     compression='gzip', 
#                     lines=True, 
#                     orient='records')