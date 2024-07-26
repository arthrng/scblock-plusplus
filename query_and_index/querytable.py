"""
This code is based on the code written by:
Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines.
In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer.

The code can be found at:
https://github.com/wbsg-uni-mannheim/SC-Block
"""

import copy
import logging
import json
import os
import itertools
import sys

# Append the current directory to the system path for module imports
sys.path.append('.')

# Import necessary modules for evidence retrieval
from retrieval.evidence import RetrievalEvidence

# Configure logging
logger = logging.getLogger()

def load_query_table(raw_json):
    """
    Load and initialize verified evidences from the provided raw JSON data.

    Parameters:
    - raw_json: The raw JSON data containing query table information.

    Returns:
    - A RetrievalQueryTable object initialized with the loaded data.
    """
    verified_evidences = []

    for raw_evidence in raw_json['verifiedEvidences']:
        context = raw_evidence.get('context')
        evidence = RetrievalEvidence(
            raw_evidence['id'], raw_evidence['queryTableId'], raw_evidence['entityId'],
            raw_evidence['table'], raw_evidence['rowId'], context, raw_evidence['split']
        )

        # Set optional attributes if they exist in raw evidence
        if 'seenTraining' in raw_evidence:
            evidence.seen_training = raw_evidence['seenTraining']
        if 'scale' in raw_evidence:
            evidence.scale = raw_evidence['scale']
        if 'signal' in raw_evidence:
            evidence.verify(raw_evidence['signal'])
            if 'scale' not in raw_evidence:
                evidence.determine_scale(raw_json['table'])
        if 'cornerCase' in raw_evidence:
            evidence.corner_case = raw_evidence['cornerCase']

        if evidence.query_table_id == raw_json['id'] and evidence not in verified_evidences:
            verified_evidences.append(evidence)

    return RetrievalQueryTable(
        raw_json['id'], 'retrieval', raw_json['assemblingStrategy'],
        raw_json['gtTable'], raw_json['schemaOrgClass'],
        raw_json['contextAttributes'], raw_json['table'], verified_evidences
    )

def load_query_table_from_file(path):
    """
    Load a query table from a provided file path and return a new QueryTable object.

    Parameters:
    - path: The file path to the JSON file containing the query table data.

    Returns:
    - A RetrievalQueryTable object initialized with the data from the file.
    """
    with open(path, encoding='utf-8') as gs_file:
        logger.info('Load query table from ' + path)
        query_table = load_query_table(json.load(gs_file))
        if not isinstance(query_table, (BaseQueryTable, RetrievalQueryTable)):
            logger.warning(f'Not able to load query table from {path}')
        return query_table

def load_query_tables(query_table_paths):
    """
    Load all query tables from the specified paths.

    Parameters:
    - query_table_paths: List of file paths to the query table JSON files.

    Returns:
    - A list of loaded query tables.
    """
    return [load_query_table_from_file(path) for path in query_table_paths]

def load_query_tables_by_class(query_table_paths, dataset_name):
    """
    Load all query tables of a specific class and type.

    Parameters:
    - query_table_paths: List of file paths to the query table JSON files.
    - dataset_name: The name of the dataset.

    Returns:
    - A list of loaded query tables filtered by class and type.
    """
    query_tables = []
    for gt_table in get_gt_tables(query_table_paths, dataset_name):
        for path in get_query_table_paths(query_table_paths, dataset_name, gt_table):
            query_tables.append(load_query_table_from_file(path))
    return query_tables

def get_dataset_names():
    """
    Get a list of all dataset names for the query tables.

    Returns:
    - A list of dataset names.
    """
    dataset_names = []
    path_to_classes = os.path.join(os.environ['DATA_DIR'], 'querytables')
    if os.path.isdir(path_to_classes):
        dataset_names = [
            dataset_name for dataset_name in os.listdir(path_to_classes)
            if dataset_name != 'deprecated' and 'test' not in dataset_name
        ]
    return dataset_names

def get_gt_tables(path):
    """
    Get a list of ground truth tables by schema.org class.

    Parameters:
    - path: The directory path to look for ground truth tables.

    Returns:
    - A list of ground truth tables.
    """
    gt_tables = []
    if os.path.isdir(path):
        gt_tables = [gt_table for gt_table in os.listdir(path) if gt_table != 'deprecated']
    return gt_tables

def get_query_table_paths(path, gt_table):
    """
    Get paths to query tables for a given ground truth table.

    Parameters:
    - path: The base directory path.
    - gt_table: The ground truth table name.

    Returns:
    - The path to the query tables.
    """
    return os.path.join(path, gt_table)

def get_all_query_table_paths(type):
    """
    Get paths to all query tables for a specified type.

    Parameters:
    - type: The type of query table to retrieve paths for.

    Returns:
    - A list of paths to query tables.
    """
    query_table_paths = []
    for dataset_name in get_dataset_names():
        for gt_table in get_gt_tables(type, dataset_name):
            query_table_paths.extend(get_query_table_paths(type, dataset_name, gt_table))
    return query_table_paths

def create_context_attribute_permutations(query_table):
    """
    Create all possible query table permutations based on the context attributes of the provided query table.

    Parameters:
    - query_table: The query table object to create permutations for.

    Returns:
    - A list of query tables with different permutations of context attributes.
    """
    permutations = []
    for i in range(len(query_table.context_attributes)):
        permutations.extend(itertools.permutations(query_table.context_attributes, i))

    # Remove permutations that do not contain the 'name' attribute
    permutations = [
        permutation for permutation in permutations if 'name' in permutation
        and permutation != query_table.context_attributes
    ]

    query_tables = []
    for permutation in permutations:
        new_query_table = copy.deepcopy(query_table)
        removable_attributes = [attr for attr in new_query_table.context_attributes if attr not in permutation]
        for attr in removable_attributes:
            new_query_table.remove_context_attribute(attr)
        query_tables.append(new_query_table)

    return query_tables

class BaseQueryTable:
    """
    Base class for query tables.
    """
    def __init__(self, identifier, type, assembling_strategy, gt_table, dataset_name,
                 context_attributes, table, verified_evidences):
        self.identifier = identifier
        self.type = type
        self.assembling_strategy = assembling_strategy
        self.gt_table = gt_table
        self.dataset_name = dataset_name
        self.context_attributes = context_attributes
        self.table = table
        self.verified_evidences = verified_evidences
        self.retrieved_evidences = None

    def __str__(self):
        return self.to_json(with_evidence_context=False)

    def to_json(self, with_evidence_context, with_retrieved_evidences=False):
        """
        Convert query table to JSON format.

        Parameters:
        - with_evidence_context: Include context in the evidence.
        - with_retrieved_evidences: Include retrieved evidences in the JSON.

        Returns:
        - The JSON representation of the query table.
        """
        encoded_evidence = {}
        for key in self.__dict__.keys():
            if key == 'identifier':
                encoded_evidence['id'] = self.__dict__['identifier']
            elif key == 'verified_evidences':
                encoded_evidence['verifiedEvidences'] = [
                    evidence.to_json(with_evidence_context) for evidence in self.verified_evidences
                ]
            elif key == 'retrieved_evidences':
                if with_retrieved_evidences and self.retrieved_evidences is not None:
                    encoded_evidence['retrievedEvidences'] = [
                        evidence.to_json(with_evidence_context, without_score=False) for evidence in self.retrieved_evidences
                    ]
            else:
                camel_cased_key = ''.join([key_part.capitalize() for key_part in key.split('_')])
                camel_cased_key = camel_cased_key[0].lower() + camel_cased_key[1:]
                encoded_evidence[camel_cased_key] = self.__dict__[key]
        return encoded_evidence

    def no_known_positive_evidences(self, entity_id):
        """
        Calculate number of known positive evidences for a given entity ID.

        Parameters:
        - entity_id: The entity ID to check.

        Returns:
        - The count of known positive evidences for the entity.
        """
        return sum(1 for evidence in self.verified_evidences if evidence.signal and evidence.entity_id == entity_id)

    def has_verified_evidences(self):
        """
        Check if there are any verified evidences.

        Returns:
        - True if there are verified evidences, False otherwise.
        """
        return len(self.verified_evidences) > 0

    def determine_path_to_query_table(self):
        """
        Determine the file path to save the query table.

        Returns:
        - The file path to save the query table.
        """
        gt_table = self.gt_table.lower().replace(" ", "_")
        file_name = f'gs_querytable_{gt_table}_{self.identifier}.json'
        return os.path.join('./data/querytables', self.dataset_name, file_name)

    def save(self, with_evidence_context, with_retrieved_evidences=False):
        """
        Save query table to disk.

        Parameters:
        - with_evidence_context: Include context in the evidence.
        - with_retrieved_evidences: Include retrieved evidences in the JSON.
        """
        path_to_query_table = self.determine_path_to_query_table()

        # Create directories if they do not exist
        file_name = os.path.basename(path_to_query_table)
        dir_path = os.path.dirname(path_to_query_table)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        # Save query table to file
        with open(path_to_query_table, 'w', encoding='utf-8') as f:
            json.dump(self.to_json(with_evidence_context, with_retrieved_evidences), f, indent=2, ensure_ascii=False)
            logger.info(f'Save query table {path_to_query_table}')

    def calculate_evidence_statistics_of_row(self, entity_id):
        """
        Calculate and export query table statistics per entity.

        Parameters:
        - entity_id: The entity ID to calculate statistics for.

        Returns:
        - A dictionary with statistics of the entity.
        """
        row = next(row for row in self.table if row['entityId'] == entity_id)

        evidences = sum(1 for evidence in self.verified_evidences if evidence.entity_id == row['entityId'])
        correct_value_entity = sum(1 for evidence in self.verified_evidences if evidence.entity_id == row['entityId'] and evidence.scale == 3)
        rel_value_entity = sum(1 for evidence in self.verified_evidences if evidence.entity_id == row['entityId'] and evidence.scale == 2)
        correct_entity = sum(1 for evidence in self.verified_evidences if evidence.entity_id == row['entityId'] and evidence.scale == 1)

        return {
            'gt_table': self.gt_table,
            'dataset_name': self.dataset_name,
            'entity_id': row['entityId'],
            'evidences': evidences,
            'correct_value_entity': correct_value_entity,
            'rel_value_entity': rel_value_entity,
            'correct_entity': correct_entity
        }

class RetrievalQueryTable(BaseQueryTable):
    """
    Class for retrieval query tables.
    """
    def __init__(self, identifier, type, assembling_strategy, gt_table, dataset_name,
                 context_attributes, table, verified_evidences):
        super().__init__(identifier, type, assembling_strategy, gt_table, dataset_name,
                         context_attributes, table, verified_evidences)