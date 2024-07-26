"""
This code is based on the code written by:
Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines.
In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer.

The code can be found at:
https://github.com/wbsg-uni-mannheim/SC-Block
"""

import logging
from evidence import RetrievalEvidence
from entity_serializer import EntitySerializer
from retrieval_strategy import RetrievalStrategy

import sys
sys.path.append('.')
from query_and_index.es_helper import determine_es_index_name

class QueryByEntity(RetrievalStrategy):

    def __init__(self, dataset, clusters=False, rank_evidences_by_table=False, all_attributes=False, tokenizer=None, switched=False):
        """
        Initialize the QueryByEntity retrieval strategy.

        Parameters:
        - dataset: The dataset to use.
        - clusters: Whether to use clustering.
        - rank_evidences_by_table: Whether to rank evidences by table.
        - all_attributes: Whether to serialize all attributes.
        - tokenizer: The tokenizer to use.
        - switched: Whether to use the switched mode.
        """
        name = 'query_by_entity' if not rank_evidences_by_table else 'query_by_entity_rank_by_table'
        super().__init__(dataset, name, clusters=clusters, switched=switched)
        
        self.rank_evidences_by_table = rank_evidences_by_table
        self.model_name = 'BM25'
        
        self.all_attributes = all_attributes
        if self.all_attributes:
            self.entity_serializer = EntitySerializer(dataset)
            self.model_name = f'{self.model_name}-all_attributes'
        
        self.tokenizer = tokenizer
        if self.tokenizer:
            self.model_name = f'{self.model_name}-{tokenizer}'

    def retrieve_evidence(self, query_table, evidence_count, entity_id):
        """
        Retrieve evidences for the given query table and entity ID.

        Parameters:
        - query_table: The query table to retrieve evidences from.
        - evidence_count: The number of evidences to retrieve.
        - entity_id: The entity ID to retrieve evidences for.

        Returns:
        - A list of retrieved evidences.
        """
        logger = logging.getLogger()
        evidence_id = 1
        evidences = []

        # Iterate through the query table rows
        for row in query_table.table:
            if entity_id is not None and entity_id != row['entityId'] and not self.rank_evidences_by_table:
                continue

            # Determine the Elasticsearch index name
            index_name = determine_es_index_name(self.dataset_name, clusters=self.clusters, tokenizer=self.tokenizer, switched=self.switched)
            
            # Serialize the entity to a string representation if all attributes are to be used
            if self.all_attributes:
                entity_str = self.entity_serializer.convert_to_str_representation(row)
                entity_result = self.query_tables_index_by_all_attributes(entity_str, evidence_count, index_name)
            else:
                entity_result = self.query_tables_index(row, query_table.context_attributes, evidence_count, index_name)
            
            logger.info(f'Found {len(entity_result["hits"]["hits"])} results for entity {row["entityId"]} of query table {query_table.identifier}!')

            first_score = None  # Normalize BM25 scores with the first score

            for hit in entity_result['hits']['hits']:
                if first_score is None:
                    first_score = hit['_score']
                
                # Handle cases where the target attribute value is not found
                found_value = None
                if query_table.type == 'augmentation' and query_table.target_attribute in hit['_source']:
                    found_value = hit['_source'][query_table.target_attribute]

                row_id = hit['_source']['row_id']
                table_name = hit['_source']['table']

                # Create evidence
                evidence = RetrievalEvidence(evidence_id, query_table.identifier, row['entityId'], table_name, row_id, hit['_source'])

                score = hit['_score'] / first_score
                evidence.scores[self.name] = score
                evidence.similarity_score = score

                evidences.append(evidence)
                logger.debug(f'Added evidence {evidence_id} to query table')
                evidence_id += 1

        # Optionally rank evidences by the most frequently retrieved table
        if self.rank_evidences_by_table:
            tables = [evidence.table for evidence in evidences]
            table_counts = [[table, tables.count(table)] for table in set(tables)]
            table_counts.sort(key=lambda x: x[1], reverse=True)

            collected_evidences = evidences.copy()
            evidences = []
            for table, _ in table_counts:
                evidences.extend([evidence for evidence in collected_evidences if evidence.table == table])

        return evidences
