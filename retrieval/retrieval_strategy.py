"""
This code is based on the code written by:
Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines.
In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer.

The code can be found at:
https://github.com/wbsg-uni-mannheim/SC-Block
"""

import logging
import json
import yaml
from elasticsearch import Elasticsearch

def load_es_index_configuration(tokenizer):
    """
    Load Elasticsearch index configuration from a YAML file.
    
    Parameters:
    - tokenizer: Optional tokenizer name to select a specific configuration. If None, the default configuration is used.

    Returns:
    - dict: The Elasticsearch index configuration.
    """
    with open('./config/indexing/es_index.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # Return the configuration for the specified tokenizer or the default configuration if tokenizer is None
    return config['index_configuration'].get(tokenizer, config['index_configuration']['default'])

def load_ground_truth_tables_for_filtering(dataset_name):
    """
    Load relevant ground truth tables for filtering evidences from a JSON file.
    
    Parameters:
    - dataset_name: The name of the dataset for which ground truth tables are needed.

    Returns:
    - list: A list of ground truth tables related to the specified dataset. If no relevant tables are found, an empty list is returned.
    """
    path_to_ground_truth_tables = './config/ground_truth_tables.json'
    with open(path_to_ground_truth_tables) as file:
        logging.info('Load ground truth tables')
        ground_truth_tables = json.load(file)
        # Return the list of ground truth tables for the specified dataset name or an empty list if not found
        return ground_truth_tables.get(dataset_name, [])

class RetrievalStrategy:
    """
    Strategy class for all open book table augmentation strategies. This class handles various methods for querying
    and retrieving data from an Elasticsearch index.
    """
    def __init__(self, dataset_name, name, clusters=False):
        """
        Initialize the RetrievalStrategy class.

        Parameters:
        - dataset_name: The name of the dataset.
        - name: The name of the strategy.
        - clusters: A boolean indicating whether clusters are used in the strategy. Default is False.
        """
        self.logger = logging.getLogger(__name__)
        self.dataset_name = dataset_name
        self.name = name
        self.clusters = clusters

        # Connect to Elasticsearch with the provided host, port, and authentication details
        self._es = Elasticsearch(
            [{'host': 'localhost', 'port': 9200, 'scheme': 'http'}],
            basic_auth=("elastic", 'PhB9-HXKmUGaiHKY7AE-'),
            timeout=15, max_retries=3, retry_on_timeout=True
        )
        
        self.path_to_table_corpus = f'./data/corpus/{dataset_name}'
        self.neural_search = False
        self.tokenizer = None
        self.model = None

        # Load the ground truth tables for filtering evidences based on the dataset name
        self.ground_truth_tables = load_ground_truth_tables_for_filtering(dataset_name)

    def retrieve_evidence(self, query_table, evidence_count, entity_id):
        """
        Retrieve evidence for the provided query table.

        Parameters:
        - query_table: The query table to be filled.
        - evidence_count: The number of evidences to be provided by the strategy.
        - entity_id: The entity ID for which evidence is being retrieved.

        Raises:
        - NotImplementedError: If the method is not implemented.
        """
        self.logger.warning('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def filter_evidences_by_ground_truth_tables(self, evidences):
        """
        Filter out evidences originating from ground truth tables.

        Parameters:
        - evidences: List of evidence objects to be filtered.

        Returns:
        - list: Filtered list of evidence objects that do not originate from ground truth tables.
        """
        return [evidence for evidence in evidences if evidence.table not in self.ground_truth_tables]

    def query_tables_index(self, row, context_attributes, evidence_count, index):
        """
        Query the Elasticsearch tables index with specified context attributes.

        Parameters:
        - row: The row of data to be used for matching.
        - context_attributes: The attributes to be considered for the query.
        - evidence_count: The number of evidences to retrieve.
        - index: The Elasticsearch index to query.

        Returns:
        - dict: The search results from Elasticsearch.
        """
        # Generate match queries for attributes that are present in the row and context attributes
        matching_attributes = (
            attr for attr in row.keys()
            if attr != 'entityId' and attr in context_attributes and row[attr] is not None and not isinstance(row[attr], list)
        )
        should_match = [{'match': {attr: {'query': row[attr]}}} for attr in matching_attributes]

        # Generate match queries for list attributes
        matching_list_attributes = (
            attr for attr in row.keys() if isinstance(row[attr], list) and attr in context_attributes
        )
        should_match_list = [{'match': {attr: {'query': ' '.join(row[attr])}}} for attr in matching_list_attributes]
        should_match.extend(should_match_list)

        # Create a match all query for must clause
        must_exist = {'match_all': {}}

        # Construct the query body
        query_body = {
            'size': evidence_count,
            'query': {
                'bool': {
                    'should': should_match,
                    'must': must_exist
                }
            }
        }

        # Execute the search query and return the results
        return self._es.search(body=json.dumps(query_body), index=index, request_timeout=60)

    def query_tables_index_by_all_attributes(self, entity_string, evidence_count, index):
        """
        Query the Elasticsearch tables index by matching all attributes.

        Parameters:
        - entity_string: The entity string to match against all attributes.
        - evidence_count: The number of evidences to retrieve.
        - index: The Elasticsearch index to query.

        Returns:
        - dict: The search results from Elasticsearch.
        """
        should_match = [{'match': {'all_attributes': entity_string}}]

        # Construct the query body
        query_body = {
            'size': evidence_count,
            'query': {
                'bool': {
                    'should': should_match
                }
            }
        }

        # Execute the search query and return the results
        return self._es.search(body=json.dumps(query_body), index=index, request_timeout=60)

    def query_tables_index_by_table_row_id(self, table, row_id, index):
        """
        Query the Elasticsearch tables index by table and row ID.

        Parameters:
        - table: The table ID to match.
        - row_id: The row ID to match.
        - index: The Elasticsearch index to query.

        Returns:
        - dict: The matched record if found, otherwise None.
        """
        query_body = {
            'size': 3,
            'query': {
                'bool': {
                    'should': [
                        {'match': {'table': {'query': table}}},
                        {'match': {'row_id': {'query': row_id}}}
                    ]
                }
            }
        }

        # Execute the search query
        search_result = self._es.search(body=json.dumps(query_body), index=index, request_timeout=60)
        for hit in search_result['hits']['hits']:
            # Check for an exact match of table and row ID
            if hit['_source']['table'] == table and hit['_source']['row_id'] == row_id:
                record = hit['_source']
                record['_id'] = hit['_id']
                return record

        self.logger.info('No exact match found!')
        return None

    def query_tables_index_by_table_id(self, table, index):
        """
        Query the Elasticsearch tables index by table ID.

        Parameters:
        - table: The table ID to match.
        - index: The Elasticsearch index to query.

        Returns:
        - dict: The matched record if found, otherwise None.
        """
        query_body = {
            'size': 3,
            'query': {
                'bool': {
                    'should': [{'match': {'table': {'query': table}}}]
                }
            }
        }

        # Execute the search query
        search_result = self._es.search(body=json.dumps(query_body), index=index, request_timeout=60)
        for hit in search_result['hits']['hits']:
            if hit['_source']['table'] == table:
                return hit['_source']

        return None

    def query_for_unique_values(self, field, index):
        """
        Query for unique values in a specified field.

        Parameters:
        - field: The field for which unique values are to be retrieved.
        - index: The Elasticsearch index to query.

        Returns:
        - dict: The search results containing unique values for the specified field.
        """
        query_body = {
            'size': 3,
            'aggs': {
                'unique_values': {
                    'terms': {'field': field, 'size': 500}
                }
            }
        }

        # Execute the search query and return the results
        search_result = self._es.search(body=json.dumps(query_body), index=index, request_timeout=60)
        return search_result

    def query_tables_index_by_id(self, ids, index):
        """
        Query the Elasticsearch tables index by document IDs.

        Parameters:
        - ids: The list of document IDs to match.
        - index: The Elasticsearch index to query.

        Returns:
        - dict: The search results sorted by the order of IDs provided.
        """
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
        """
        Get the number of entities in the specified index.

        Parameters:
        - index: The Elasticsearch index to query.

        Returns:
        - int: The count of entities in the index.
        """
        return int(self._es.cat.count(index, params={"format": "json"})[0]['count'])
