import logging
import os
import json

import yaml
from elasticsearch import Elasticsearch


def load_es_index_configuration(tokenizer):
    # Load es index configuration from yaml
    with open('./config/indexing/es_index.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if tokenizer is None:
        return config['index_configuration']['default']
    else:
        return config['index_configuration'][tokenizer]

def load_ground_truth_tables_for_filtering(schema_org_class):
    """ :return list of relevant ground truth tables.
        Evidences that originate from these ground truth tables will not be considered."""
    path_to_ground_truth_tables = './config/ground_truth_tables.json'
    with open(path_to_ground_truth_tables) as file:
        logging.info('Load ground truth tables')
        ground_truth_tables = json.load(file)
        if schema_org_class in ground_truth_tables:
            return ground_truth_tables[schema_org_class]
        else:
            return []

class RetrievalStrategy:
    """Strategy class for all open book table augmentation strategies"""

    def __init__(self, schema_org_class, name, clusters=False):
        self.logger = logging.getLogger()
        self.schema_org_class = schema_org_class
        self.name = name
        self.clusters = clusters

        # Connect to Elasticsearch
        self._es = Elasticsearch([{'host': 'localhost', 'port':9200, 'scheme': 'http'}], basic_auth=("elastic", 'PhB9-HXKmUGaiHKY7AE-'), timeout=15, max_retries=3, retry_on_timeout=True)
        self.path_to_table_corpus = f'./data/corpus/{schema_org_class}'
        self.neural_search = False
        self.tokenizer = None
        self.model = None

        self.ground_truth_tables = load_ground_truth_tables_for_filtering(schema_org_class)

    def retrieve_evidence(self, query_table, evidence_count, entity_id):
        """Fills the provided query table and delivers evidence if expected
            :param query_table querytable which will be filled
            :param evidence_count   Number of evidences provided by strategy
        """
        logger = logging.getLogger()
        logger.warning('Method not implemented!')

        raise NotImplementedError('Method not implemented!')

    def filter_evidences_by_ground_truth_tables(self, evidences):
        filtered_evidences = [evidence for evidence in evidences
                              if evidence.table not in self.ground_truth_tables]

        return filtered_evidences

    def query_tables_index(self, row, context_attributes, evidence_count, index):
        # TO-DO: Check for context attributes!
        matching_attributes = (attribute for attribute in row.keys()
                               if attribute != 'entityId'
                               and attribute in context_attributes
                               and row[attribute] is not None
                               and not type(row[attribute]) is list)
        should_match = [{'match': {attribute: {'query': row[attribute]}}} for attribute in matching_attributes]

        # List attributes
        matching_list_attributes = (attribute for attribute in row.keys() if
                                    row[attribute] is list and attribute in context_attributes)
        should_match_list = [{'match': {attribute: {'query': ' '.join(row[attribute])}}}
                             for attribute in matching_list_attributes]
        should_match.extend(should_match_list)

        # Uncomment following block to make sure that all hits contain the target attribute
        # if target_attribute is not None:
        #     must_exist = {'exists': {'field': target_attribute}}
        # else:
        must_exist = {'match_all': {}}

        query_body = {
            'size': evidence_count,
            'query':
                {
                    'bool':
                        {'should': should_match,
                         'must': must_exist
                         }
                }
        }

        return self._es.search(body=json.dumps(query_body), index=index, request_timeout=60)

    def query_tables_index_by_all_attributes(self, entity_string, evidence_count, index):

        #should_match = [{'match': {'all_attributes': {'query': entity_string}}}]
        should_match = [{'match': {'all_attributes': entity_string}}]

        query_body = {
            'size': evidence_count,
            'query':
                {
                    'bool':
                        {'should': should_match
                         }
                }
        }

        return self._es.search(body=json.dumps(query_body), index=index, request_timeout=60)

    def query_tables_index_by_table_row_id(self, table, row_id, index):
        # To-Do: Do not analyze table and row_id during indexing to enable exact matches
        query_body = {
            'size': 3,
            'query':
                {
                    'bool':
                        {'should': [{'match': {'table': {'query': table}}}, {'match': {'row_id': {'query': row_id}}}]
                         }
                }
        }

        search_result = self._es.search(body=json.dumps(query_body), index=index, request_timeout=60)
        for hit in search_result['hits']['hits']:
            if hit['_source']['table'] == table and hit['_source']['row_id'] == row_id:
                record = hit['_source']
                record['_id'] = hit['_id']
                return record

        # Return None if exactly matching hit was not found
        logging.info('No exact match found!')
        return None

    def query_tables_index_by_table_id(self, table, index):
        # To-Do: Do not analyze table and row_id during indexing to enable exact matches
        query_body = {
            'size': 3,
            'query':
                {
                    'bool':
                        {'should': [{'match': {'table': {'query': table}}}]
                         }
                }
        }

        search_result = self._es.search(body=json.dumps(query_body), index=index, request_timeout=60)
        for hit in search_result['hits']['hits']:
            if hit['_source']['table'] == table:
                return hit['_source']

        # Return None if exactly matching hit was not found
        return None

    def query_for_unique_values(self, field, index):
        query_body = {
            'size': 3,
            'aggs': {
                'langs': {
                    'terms': {'field': field, 'size': 500}
                }
            }
        }

        search_result = self._es.search(body=json.dumps(query_body), index=index, request_timeout=60)

        return search_result

    def query_tables_index_by_id(self, ids, index):
        query_body = {
            'size': len(ids),
            'query': {
                'terms': {
                    '_id': [str(identifier) for identifier in ids]
                }
            }
        }
        sorted_hits = []
        query_results = self._es.search(body=json.dumps(query_body), index=index, request_timeout=60)

        hits = query_results['hits']['hits'].copy()
        for identifier in ids:
            found_hit = None
            for hit in hits:
                if str(identifier) == hit['_id']:
                    found_hit = hit
                    sorted_hits.append(hit)
                    break

            if found_hit is not None:
                hits.remove(found_hit)

        query_results['hits']['hits'] = sorted_hits
        return query_results


    def get_no_index_entities(self, index):
        return int(self._es.cat.count(index, params={"format": "json"})[0]['count'])