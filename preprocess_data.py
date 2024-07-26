import pandas as pd
import os
from tqdm import tqdm

def assign_cluster_id(identifier, cluster_id_dict, cluster_id_amount):
    """
    Assigns a cluster ID to a given identifier based on the cluster_id_dict. If the identifier is not found,
    assigns a default cluster ID.

    Args:
    identifier (str): The identifier to assign a cluster ID to.
    cluster_id_dict (dict): Dictionary mapping identifiers to cluster IDs.
    cluster_id_amount (int): The total number of clusters.

    Returns:
    int: The assigned cluster ID.
    """
    try:
        result = cluster_id_dict[identifier]
    except KeyError:
        result = cluster_id_amount
    return result

def preprocess_dataset(dataset_name, left_name, right_name):
    """
    Preprocesses a dataset by reading raw data, combining training, validation, and test sets,
    assigning clusters, and converting the data to the DeepMatcher format.

    Args:
    dataset_name (str): Name of the dataset to preprocess.
    left_name (str): Name of the left table.
    right_name (str): Name of the right table.
    """
    parent_directory = os.path.dirname(os.getcwd()).replace('\\', '/')

    # Read the left and right data sets
    left_df = pd.read_csv(f'{parent_directory}/src/data/raw/{dataset_name}/tableA.csv', engine='python')
    right_df = pd.read_csv(f'{parent_directory}/src/data/raw/{dataset_name}/tableB.csv', engine='python')

    # Read the training, validation, and test sets
    train_df = pd.read_csv(f'{parent_directory}/src/data/raw/{dataset_name}/train.csv')
    test_df = pd.read_csv(f'{parent_directory}/src/data/raw/{dataset_name}/test.csv')
    val_df = pd.read_csv(f'{parent_directory}/src/data/raw/{dataset_name}/valid.csv')

    # Convert product IDs to strings and set as index
    left_df['id'] = f'{left_name}_' + left_df['id'].astype(str)
    right_df['id'] = f'{right_name}_' + right_df['id'].astype(str)
    left_df = left_df.set_index('id', drop=False).fillna('')
    right_df = right_df.set_index('id', drop=False).fillna('')

    # Combine training, validation, and test sets
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df = full_df[full_df['label'] == 1]

    # Convert IDs to string format
    full_df['ltable_id'] = f'{left_name}_' + full_df['ltable_id'].astype(str)
    full_df['rtable_id'] = f'{right_name}_' + full_df['rtable_id'].astype(str)

    # Assign products to clusters
    bucket_list = []
    for i, row in full_df.iterrows():
        left = row['ltable_id']
        right = row['rtable_id']
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

    cluster_id_dict = {v: i for i, id_set in enumerate(bucket_list) for v in id_set}

    # Convert to DeepMatcher format
    train_df = convert_to_pairs_format(train_df, 'train', left_df, right_df, parent_directory, dataset_name, left_name, right_name, cluster_id_dict, len(bucket_list))
    val_df = convert_to_pairs_format(val_df, 'val', left_df, right_df, parent_directory, dataset_name, left_name, right_name, cluster_id_dict, len(bucket_list))
    test_df = convert_to_pairs_format(test_df, 'test', left_df, right_df, parent_directory, dataset_name, left_name, right_name, cluster_id_dict, len(bucket_list))

    # Convert to entity format
    dataframes = {'train': train_df, 'val': val_df, 'test': test_df}
    for split, df in dataframes.items():
        convert_to_entity_format(df, split, left_df, right_df, parent_directory, dataset_name, cluster_id_dict, len(bucket_list))

def convert_to_pairs_format(df, split, left_df, right_df, parent_directory, dataset_name, left_name, right_name, cluster_id_dict, cluster_id_amount):
    """
    Converts a dataframe to a pairs format suitable for DeepMatcher.

    Args:
    df (pd.DataFrame): Dataframe to convert.
    split (str): The data split (train, val, test).
    left_df (pd.DataFrame): Dataframe for the left table.
    right_df (pd.DataFrame): Dataframe for the right table.
    parent_directory (str): Parent directory path.
    dataset_name (str): Name of the dataset.
    left_name (str): Name of the left table.
    right_name (str): Name of the right table.
    cluster_id_dict (dict): Dictionary mapping identifiers to cluster IDs.
    cluster_id_amount (int): The total number of clusters.

    Returns:
    pd.DataFrame: Converted dataframe.
    """
    df['ltable_id'] = f'{left_name}_' + df['ltable_id'].astype(str)
    df['rtable_id'] = f'{right_name}_' + df['rtable_id'].astype(str)
    df['label'] = df['label'].apply(int)

    split_left = left_df.loc[df['ltable_id'].values].reset_index(drop=True)
    split_right = right_df.loc[df['rtable_id'].values].reset_index(drop=True)
    split_labels = df['label'].values.tolist()

    split_left['cluster_id'] = split_left['id'].apply(assign_cluster_id, args=(cluster_id_dict, cluster_id_amount))
    split_right['cluster_id'] = split_right['id'].apply(assign_cluster_id, args=(cluster_id_dict, cluster_id_amount))

    df = split_left.add_prefix('left_').join(split_right.add_prefix('right_'))
    df['label'] = split_labels

    # Store the datasets
    os.makedirs(os.path.dirname(f'{parent_directory}/src/data/processed/'), exist_ok=True)
    df.to_json(f'{parent_directory}/src/data/processed/{dataset_name}-{split}-pairs.json.gz', compression='gzip', lines=True, orient='records')

    return df

def convert_to_entity_format(df, split, left_df, right_df, parent_directory, dataset_name, cluster_id_dict, cluster_id_amount):
    """
    Converts a dataframe to an entity format suitable for DeepMatcher.

    Args:
    df (pd.DataFrame): Dataframe to convert.
    split (str): The data split (train, val, test).
    left_df (pd.DataFrame): Dataframe for the left table.
    right_df (pd.DataFrame): Dataframe for the right table.
    parent_directory (str): Parent directory path.
    dataset_name (str): Name of the dataset.
    cluster_id_dict (dict): Dictionary mapping identifiers to cluster IDs.
    cluster_id_amount (int): The total number of clusters.
    """
    merged_ids = set(df['left_id']).union(df['right_id'])

    entity_set = pd.concat([left_df[left_df['id'].isin(merged_ids)], right_df[right_df['id'].isin(merged_ids)]])
    entity_set['cluster_id'] = entity_set['id'].apply(assign_cluster_id, args=(cluster_id_dict, cluster_id_amount))

    single_entities = entity_set[entity_set['cluster_id'] == cluster_id_amount].copy()
    single_entities = single_entities.reset_index(drop=True)
    single_entities['cluster_id'] = single_entities['cluster_id'] + single_entities.index

    entity_set = entity_set.drop(single_entities['id'])
    entity_set = pd.concat([entity_set, single_entities]).reset_index(drop=True)

    print(f'Amount of entity descriptions: {len(entity_set)}')
    print(f'Amount of clusters: {len(entity_set["cluster_id"].unique())}')

    os.makedirs(os.path.dirname(f'{parent_directory}/src/data/processed/'), exist_ok=True)
    entity_set.to_pickle(f'{parent_directory}/src/data/processed/{dataset_name}-{split}-entities.pkl.gz', compression='gzip')

def convert_wdcproducts_to_deepmatcher(corner_case_perc):
    """
    Converts the WDC products dataset to a format suitable for DeepMatcher.

    Args:
    corner_case_perc (int): Percentage of corner cases.
    """
    parent_directory = os.path.dirname(os.getcwd()).replace('\\', '/')
    splits = ['train', 'valid', 'test']
    ds_sets = {}

    for split in splits:
        path = f'{parent_directory}/src/data/raw/wdcproducts/{corner_case_perc}pair/wdcproducts{corner_case_perc}cc{100-corner_case_perc}rnd{"050un_gs" if split == "test" else "000un_"+split+"_small"}.json.gz'
        df_split = pd.read_json(path, lines=True)
        ds_sets[split] = df_split.sort_values(by=['label'], ascending=False)

    cluster_ids = {cluster_id for split in ds_sets for cluster_id in ds_sets[split]['cluster_id_left'].unique()}
    unique_ids = {id_ for split in ds_sets for id_ in ds_sets[split]['id_left'].unique()}

    for cluster_id in tqdm(cluster_ids):
        cluster_df = pd.concat([ds_sets[split][(ds_sets[split]['cluster_id_left'] == cluster_id) & (ds_sets[split]['cluster_id_right'] == cluster_id)] for split in splits])
        unique_ids.update(cluster_df['id_left'].unique())

    left_df = ds_sets['train'][['id_left']].drop_duplicates().reset_index(drop=True)
    right_df = ds_sets['train'][['id_right']].drop_duplicates().reset_index(drop=True)

    left_df['id_left'] = 'left_' + left_df['id_left'].astype(str)
    right_df['id_right'] = 'right_' + right_df['id_right'].astype(str)

    full_df = pd.concat([ds_sets[split] for split in splits], ignore_index=True)

    convert_to_pairs_format(full_df, 'full', left_df, right_df, parent_directory, 'wdcproducts', 'left', 'right')
    convert_to_entity_format(full_df, 'full', left_df, right_df, parent_directory, 'wdcproducts', cluster_ids, len(cluster_ids))
