"""
This code is based on the code written by:
Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines.
In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer.

The code can be found at:
https://github.com/wbsg-uni-mannheim/SC-Block
"""

from elasticsearch import Elasticsearch, helpers
import logging
import gzip
from multiprocessing import Pool
import os
import time
from tqdm import tqdm
import json

import sys
sys.path.append('.')
from retrieval.entity_extraction import extract_entity
from retrieval.entity_serializer import EntitySerializer
from es_helper import determine_es_index_name
from retrieval.retrieval_strategy import load_es_index_configuration

def load_data(dataset, worker=1, tokenizer=None, no_test=True, with_language_detection=False, duplicate_check=False, entity_length_check=False):
    """Load data into Elasticsearch and index entities.

    Args:
        dataset (str): Name of the dataset.
        worker (int): Number of worker processes for parallel processing.
        tokenizer (str or None): Tokenizer for processing text (if applicable).
        no_test (bool): Whether to skip test mode.
        with_language_detection (bool): Whether to include language detection.
        duplicate_check (bool): Whether to check for duplicate entities.
        entity_length_check (bool): Whether to check entity length.
    """
    logger = logging.getLogger()

    # Connect to Elasticsearch
    _es = Elasticsearch(
        [{'host': 'localhost', 'port': 9200, 'scheme': 'http'}],
        basic_auth=("elastic", 'PhB9-HXKmUGaiHKY7AE-'),
        timeout=15, max_retries=3, retry_on_timeout=True
    )

    if not _es.ping():
        raise ValueError("Connection to Elasticsearch failed")

    start = time.time()
    clusters = None

    # Load mapping and create index
    mapping = load_es_index_configuration(tokenizer)
    entity_index_name = determine_es_index_name(dataset, tokenizer=tokenizer)

    if no_test:
        if _es.indices.exists(index=entity_index_name):
            _es.indices.delete(index=entity_index_name)
        _es.indices.create(index=entity_index_name, body=mapping)
    
    # Initialize statistics
    index_statistics = {
        'tables_added': 0, 
        'tables_not_added': 0, 
        'entities_added': 0, 
        'entities_not_added': 0
    }

    directory = f'./data/corpus/{dataset}'

    # Prepare parallel processing
    results = []
    if worker > 0:
        pool = Pool(worker)
    collected_filenames = []

    for filename in os.listdir(directory):
        if clusters is not None:
            file_table = filename.lower().split('_')[1]
            if file_table[:3] not in clusters or file_table not in clusters[file_table[:3]]:
                continue

        collected_filenames.append(filename)
        if len(collected_filenames) > 50:
            process_files(
                directory, collected_filenames, entity_index_name, dataset,
                clusters, with_language_detection, duplicate_check, entity_length_check,
                worker, results, pool
            )
            collected_filenames = []

    if collected_filenames:
        process_files(
            directory, collected_filenames, entity_index_name, dataset,
            clusters, with_language_detection, duplicate_check, entity_length_check,
            worker, results, pool
        )

    # Wait for all tasks to finish
    pbar = tqdm(total=len(results))
    logger.info('Waiting for all tasks to finish!')

    while results:
        results, entity_index = send_actions_to_elastic(
            _es, results, 0, index_statistics, pbar, no_test, worker
        )

    pbar.close()

    if worker > 0:
        pool.close()
        pool.join()

    # Report statistics
    execution_time = time.time() - start
    logger.info('Added entities: {}'.format(index_statistics['entities_added']))
    logger.info('Not added entities: {}'.format(index_statistics['entities_not_added']))
    logger.info('Added tables: {}'.format(index_statistics['tables_added']))
    logger.info('Not added tables: {}'.format(index_statistics['tables_not_added']))
    logger.info('Indexing time: {} sec'.format(execution_time))


def process_files(directory, files, entity_index_name, dataset, clusters, with_language_detection, duplicate_check, entity_length_check, worker, results, pool):
    """Process a batch of files and create indexing actions.

    Args:
        directory (str): Directory containing files.
        files (list): List of filenames to process.
        entity_index_name (str): Elasticsearch index name.
        dataset (str): Name of the dataset.
        clusters (dict or None): Clustering information.
        with_language_detection (bool): Whether to include language detection.
        duplicate_check (bool): Whether to check for duplicate entities.
        entity_length_check (bool): Whether to check entity length.
        worker (int): Number of worker processes.
        results (list): List to store asynchronous results.
    """
    if worker == 0:
        results.append(create_table_index_action(
            directory, files, entity_index_name, dataset, clusters,
            with_language_detection, duplicate_check, entity_length_check
        ))
    else:
        results.append(pool.apply_async(create_table_index_action, (
            directory, files, entity_index_name, dataset, clusters,
            with_language_detection, duplicate_check, entity_length_check
        )))


def send_actions_to_elastic(_es, results, entity_index, index_statistics, pbar, no_test, worker):
    """Send actions to Elasticsearch and update statistics.

    Args:
        _es (Elasticsearch): Elasticsearch client.
        results (list): List of asynchronous results.
        entity_index (int): Current entity index.
        index_statistics (dict): Statistics dictionary.
        pbar (tqdm): Progress bar.
        no_test (bool): Whether to skip test mode.
        worker (int): Number of worker processes.

    Returns:
        tuple: Updated results list and entity index.
    """
    logger = logging.getLogger()
    collected_results = []
    actions = []

    for result in results:
        new_actions, new_statistics = (result.get() if result.ready() else (None, None)) if worker > 0 else result

        if new_actions and new_statistics:
            logger.debug('Retrieved {} actions'.format(len(new_actions)))
            for action in new_actions:
                action['_id'] = entity_index
                actions.append(action)
                entity_index += 1

            index_statistics['entities_added'] += new_statistics['entities_added']
            index_statistics['entities_not_added'] += new_statistics['entities_not_added']
            collected_results.append(result)
            pbar.update(1)

    if actions and no_test:
        helpers.bulk(client=_es, actions=actions, chunk_size=1000, request_timeout=60)

    results = [result for result in results if result not in collected_results]

    return results, entity_index


def create_table_index_action(directory, files, entity_index, dataset, clusters, with_language_detection, duplicate_check, entity_length_check):
    """Creates entity documents for indexing into Elasticsearch.

    Args:
        directory (str): Directory containing files.
        files (list): List of filenames to process.
        entity_index (int): Current entity index.
        dataset (str): Name of the dataset.
        clusters (dict or None): Clustering information.
        with_language_detection (bool): Whether to include language detection.
        duplicate_check (bool): Whether to check for duplicate entities.
        entity_length_check (bool): Whether to check entity length.

    Returns:
        tuple: List of indexing actions and statistics.
    """
    logger = logging.getLogger()
    entity_serializer = EntitySerializer(dataset)

    actions = []
    index_statistics = {'entities_added': 0, 'entities_not_added': 0}

    for filename in files:
        file_path = os.path.join(directory, filename)

        if clusters is not None:
            file_table = filename.lower().split('_')[1]
            if file_table[:3] not in clusters or file_table not in clusters[file_table[:3]]:
                continue
        else:
            file_table = None

        try:
            with gzip.open(file_path, 'rb') as file:
                for line in file:
                    raw_entity = json.loads(line)
                    if clusters is not None and file_table:
                        if raw_entity['row_id'] not in clusters[file_table[:3]].get(file_table, []):
                            index_statistics['entities_not_added'] += 1
                            continue

                    if 'name' in raw_entity and raw_entity['name']:
                        entity = extract_entity(raw_entity, dataset)
                        entity_wo_description = {k: v for k, v in entity.items() if k != 'description'}

                        if 'name' in entity and (not entity_length_check or len(entity) > 1):
                            if duplicate_check:
                                if entity_wo_description not in actions:
                                    actions.append(entity_wo_description)
                                    entity.update({
                                        'table': filename.lower(),
                                        'row_id': raw_entity['row_id'],
                                        'page_url': raw_entity['page_url'],
                                        'all_attributes': entity_serializer.convert_to_str_representation(entity)
                                    })
                                    actions.append({'_index': entity_index, '_source': entity})
                                    index_statistics['entities_added'] += 1
                                else:
                                    index_statistics['entities_not_added'] += 1
                            else:
                                entity.update({
                                    'table': filename.lower(),
                                    'row_id': raw_entity['row_id'],
                                    'page_url': raw_entity['page_url'],
                                    'all_attributes': entity_serializer.convert_to_str_representation(entity)
                                })
                                actions.append({'_index': entity_index, '_source': entity})
                                index_statistics['entities_added'] += 1
                        else:
                            index_statistics['entities_not_added'] += 1
                    else:
                        logger.debug(
                            'Entity does not have a name attribute: {} - not added to index: {}'
                            .format(str(raw_entity), filename)
                        )
                        index_statistics['entities_not_added'] += 1

        except gzip.BadGzipFile as e:
            logger.warning('{} - Cannot open file {}'.format(e, filename))

    logger.debug('Added {} actions'.format(index_statistics['entities_added']))

    return actions, index_statistics


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    dataset_names = ['walmart-amazon', 'wdcproducts20pair', 'wdcproducts80pair']
    for dataset_name in dataset_names:
        load_data(dataset=dataset_name)
