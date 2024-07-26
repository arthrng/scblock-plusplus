"""
This code is based on the code written by:
Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines.
In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer.

The code can be found at:
https://github.com/wbsg-uni-mannheim/SC-Block
"""

import gzip
import json
import logging
import sys
from elasticsearch import helpers

# Append the path to the current working directory
sys.path.append('.')
from retrieval.value_normalizer import normalize_value, get_datatype

def determine_es_index_name(schema_org_class, table=False, tokenizer=None, clusters=False):
    """Determine the Elasticsearch index name based on parameters.

    Args:
        schema_org_class (str): The schema.org class name.
        table (bool, optional): Whether the index is for a table or not. Defaults to False.
        tokenizer (str, optional): The tokenizer used. Defaults to None.
        clusters (bool, optional): Whether to include clustering in the index name. Defaults to False.

    Returns:
        str: The constructed index name.
    """
    if table:
        index_name = '{}_{}'.format(schema_org_class, 'table_index')
    else:
        index_name = '{}_{}'.format(schema_org_class, 'entity_index')

    if tokenizer is not None:
        # Append tokenizer information to the index name if provided
        index_name = '{}_{}'.format(index_name, tokenizer)

    if clusters:
        # Append clustering information to the index name if specified
        index_name = '{}_{}'.format(index_name, 'only_clustered')

    logging.getLogger().info('Index name {}'.format(index_name))
    print(index_name)

    return index_name

def index_table(path_to_table_corpus, schema_org_class, table_file_name, table_index_name, es):
    """Index a table from a gzip-compressed JSON file into Elasticsearch.

    Args:
        path_to_table_corpus (str): Path to the table corpus directory.
        schema_org_class (str): The schema.org class name.
        table_file_name (str): The name of the table file.
        table_index_name (str): The name of the Elasticsearch index.
        es (Elasticsearch): An Elasticsearch client instance.
    """
    logger = logging.getLogger()
    file_path = '{}{}/{}'.format(path_to_table_corpus, schema_org_class, table_file_name)
    actions = []

    with gzip.open(file_path, 'rb') as file:
        entity_index_number = 0
        for line in file.readlines():
            entity_index_number += 1
            raw_entity = json.loads(line)

            if 'name' in raw_entity:
                entity = {'table': table_index_name, 'row_id': raw_entity['row_id'],
                          'page_url': raw_entity['page_url']}

                # Normalize/unpack raw_entity if necessary
                for key in raw_entity.keys():
                    if isinstance(raw_entity[key], str):
                        entity[key] = normalize_value(raw_entity[key], get_datatype(key), raw_entity)
                    elif isinstance(raw_entity[key], dict):
                        if 'name' in raw_entity[key]:
                            entity[key] = normalize_value(raw_entity[key]['name'], get_datatype(key), raw_entity)
                        else:
                            for property_key in raw_entity[key].keys():
                                entity[property_key] = normalize_value(raw_entity[key][property_key], get_datatype(key), raw_entity)
                    elif isinstance(raw_entity[key], list):
                        if all(isinstance(element, str) for element in raw_entity[key]):
                            entity[key] = [normalize_value(element, get_datatype(key), raw_entity) for element in raw_entity[key]]
                        elif all(isinstance(element, dict) for element in raw_entity[key]) and all('name' in element for element in raw_entity[key]):
                            entity[key] = [normalize_value(element['name'], get_datatype(key), raw_entity) for element in raw_entity[key]]

                actions.append({'_index': table_index_name, '_source': entity})

            else:
                logger.warning(
                    'TABLE INDEX ERROR - Entity does not have a name attribute: {} - not added to index: {}'
                        .format(str(raw_entity), table_index_name))

    mapping = {"settings": {"number_of_shards": 1}, "mappings": {"date_detection": False}}
    es.indices.create(index=table_index_name, ignore=400, body=json.dumps(mapping))
    try:
        helpers.bulk(client=es, actions=actions, request_timeout=30)
    except helpers.BulkIndexError as e:
        logger.warning(e)
    logger.info('Table {} indexed'.format(table_file_name))