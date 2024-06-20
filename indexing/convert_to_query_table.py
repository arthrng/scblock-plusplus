import logging
import os
from collections import defaultdict

import pandas as pd

from evidence import RetrievalEvidence
from querytable import RetrievalQueryTable


def convert_table_to_query_table(dataset):
    """ Convert Test set Table A of deepmatcher benchmark to query table
    :param dataset string org class represents the dataset name"""

    parent_directory = os.path.dirname(os.getcwd()).replace('\\', '/') #'/content/drive/MyDrive/Master/Thesis' 
    path_to_table_a = f'{parent_directory}/src/data/raw/{dataset}/tableA.csv'
    path_to_test_set = f'{parent_directory}/src/data/raw/{dataset}/test.csv' 
    path_to_train_set = f'{parent_directory}/src/data/raw/{dataset}/train.csv'
    path_to_valid_set = f'{parent_directory}/src/data/raw/{dataset}/valid.csv'

    # Add all records as evidences to query table
    test_record_dict = defaultdict(list)
    for split in ['train', 'valid', 'test']:
        path = f'{parent_directory}/src/data/raw/{dataset}/{split}.csv' 
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                line_values = line.split(',')
                if line_values[0] == 'ltable_id':
                    # Skip first line
                    continue
                test_record_dict[line_values[0]].append({'row_id': line_values[1], 'label': line_values[2], 'split': split})

    # Extract seen records
    seen_evidences_records = set()
    seen_entity_records = set()
    #seen_pair = []
    for path in [path_to_train_set, path_to_valid_set]:
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                line_values = line.split(',')
                if line_values[0] == 'ltable_id':
                    continue
                if line_values[2] == '0':
                    seen_evidences_records.add(line_values[1])
                    seen_entity_records.add(line_values[0])

    # Build Query Table
    query_table_ids = {'amazon-google': 10000,
                       'walmart-amazon': 11000,
                       'wdcproducts20pair': 12000,
                       'wdcproducts80pair': 13000}
    
    qt_id = query_table_ids[dataset]
    assembling_strategy = 'Test {} of data set {}'.format('tableA', dataset)
    gt_table = dataset

    verified_evidences = []
    table = []
    evidence_id = 1

    df_table = pd.read_csv(path_to_table_a).fillna('')
    context_attributes = df_table.columns.to_list()[1:]
    if 'wdcproducts' in dataset:
        context_attributes = list(map(lambda x: x.replace('title', 'name'), context_attributes))

    seen_counter = {'both_seen': 0, 'left_seen': 0, 'right_seen': 0, 'none_seen': 0}

    for index, row in df_table.iterrows():
        if len(table) >= 100:
            # Check data sets into query tables with 100 records
            query_table = RetrievalQueryTable(qt_id, '', assembling_strategy,
                                              gt_table, dataset,
                                              context_attributes,  # Exclude id
                                              table, verified_evidences)
            query_table.save(with_evidence_context=False)

            # Initialize variables for new query table
            verified_evidences = []
            table = []
            evidence_id = 1
            qt_id += 1

        entity = row.to_dict()
        entity = dict((k.lower(), v) for k, v in entity.items())

        entity['entityId'] = entity['id']
        print(entity)
        del entity['id']

        if dataset in ['amazon-google', 'walmart-amazon']:
            entity['name'] = entity['title']
            del entity['title']
        elif 'wdcproducts' in dataset:
            entity['name'] = entity['title']
            del entity['title']
            del entity['cluster_id']
            context_attributes = list(map(lambda x: x.replace('Pant', 'Ishan'), context_attributes))
        #added_pairs = []
        for reference in test_record_dict[str(entity['entityId'])]:
            table_identifier = '{}_{}.json.gz'.format(dataset, 'tableA').lower() #if table_name == 'tableB' else 'tableB'
            print(reference['split'])
            evidence = RetrievalEvidence(evidence_id, qt_id, entity['entityId'],
                                         table_identifier, reference['row_id'], None, reference['split'])
            evidence.scale = int(reference['label'])
            evidence.signal = int(reference['label']) == 1

            if str(entity['entityId']) in seen_entity_records and reference['row_id'] in seen_evidences_records:
                evidence.seen_training = 'seen'
                if int(reference['label']) == 1 and reference['split'] == 'test':
                    seen_counter['both_seen'] += 1
            elif str(entity['entityId']) in seen_entity_records:
                evidence.seen_training = 'left_seen'
                if int(reference['label']) == 1 and reference['split'] == '/test':
                    seen_counter['left_seen'] += 1
            elif reference['row_id'] in seen_evidences_records:
                evidence.seen_training = 'right_seen'
                if int(reference['label']) == 1 and reference['split'] == 'test':
                    seen_counter['right_seen'] += 1
            else:
                evidence.seen_training = 'unseen'
                if int(reference['label']) == 1 and reference['split'] == 'test':
                    seen_counter['none_seen'] += 1

            verified_evidences.append(evidence)
                #added_pairs.append(pair_id)
            evidence_id += 1

        # Add all entities of table a to query table
        table.append(entity)

    # Save final query table
    query_table = RetrievalQueryTable(qt_id, '', assembling_strategy,
                                      gt_table, dataset,
                                      context_attributes,  # Exclude id
                                      table, verified_evidences)
    query_table.save(with_evidence_context=False)
    logging.info('Converted {} of {} to query table'.format('tableA', dataset))

    print(seen_counter)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    dataset_names = ['amazon-google', 'walmart-amazon', 'wdcproducts20pair', 'wdcproducts80pair']
    for dataset_name in dataset_names:
        convert_table_to_query_table(dataset_name)