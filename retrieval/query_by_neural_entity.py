"""
This code is based on the code written by:
Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines.
In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer.

The code can be found at:
https://github.com/wbsg-uni-mannheim/SC-Block
"""

import logging
import faiss
import numpy as np
import torch
import os

import sys
sys.path.append('.')
from retrieval.evidence import RetrievalEvidence
from query_and_index.es_helper import determine_es_index_name
from query_and_index.faiss_collector import determine_path_to_faiss_index
from blockers.select_blocker import select_blocker
from retrieval.retrieval_strategy import RetrievalStrategy

class QueryByNeuralEntity(RetrievalStrategy):

    def __init__(self, dataset_name, bi_encoder_name, clusters, model_name, base_model, with_projection, projection, pooling, similarity):
        """
        Initializes the QueryByNeuralEntity object with the provided parameters.

        Args:
            dataset_name (str): The Schema.org class.
            bi_encoder_name (str): The name of the bi-encoder to use.
            clusters (dict): Clustering information.
            model_name (str): The name of the model.
            base_model (str): The base model name.
            with_projection (bool): Whether to use projection.
            projection (str): Projection configuration.
            pooling (str): Pooling strategy (e.g., 'mean', 'max').
            similarity (str): Similarity metric ('cos' for cosine similarity, 'euclidean' for Euclidean distance).
        """
        super().__init__(dataset_name, 'query_by_neural_entity', clusters=clusters)
        self.logger = logging.getLogger(__name__)
        self.pooling = pooling
        self.similarity = similarity
        self.model_name = model_name

        normalize = self.similarity == 'cos'
        bi_encoder_config = {
            'name': bi_encoder_name,
            'model_name': model_name,
            'base_model': base_model,
            'with_projection': with_projection,
            'projection': projection,
            'pooling': pooling,
            'normalize': normalize
        }
        self.entity_biencoder = select_blocker(
            bi_encoder_config['name'],
            bi_encoder_config['model_name'],
            dataset_name
        )

        self.rank_evidences_by_table = False

        # Load Faiss index
        path_to_faiss_index = determine_path_to_faiss_index(
            dataset_name, model_name, pooling, similarity, clusters
        )
        if not os.path.exists(path_to_faiss_index):
            self.logger.error(f'Faiss index file does not exist at {path_to_faiss_index}')
            raise FileNotFoundError(f'Faiss index file not found: {path_to_faiss_index}')
        
        self.logger.info(f'Loading Faiss index from {path_to_faiss_index}')
        self.index = faiss.read_index(path_to_faiss_index)

    def retrieve_evidence(self, query_table, evidence_count, entity_id):
        """
        Retrieves evidence from the query table based on neural embeddings and Faiss index.

        Args:
            query_table (QueryTable): The table containing query entities.
            evidence_count (int): Number of evidence items to retrieve.
            entity_id (int or None): Specific entity ID to filter by.

        Returns:
            list: A list of evidence objects.
        """
        self.logger.info(f'Retrieving evidence for query table: {query_table.identifier}')
        evidence_id = 1
        evidences = []
        entity_vectors = []

        # Generate entity embeddings
        if entity_id is None:
            if query_table.type == 'retrieval':
                entity_vectors = self.entity_biencoder.encode_entities_and_return_pooled_outputs(query_table.table)
            elif query_table.type == 'augmentation':
                entity_vectors = self.entity_biencoder.encode_entities_and_return_pooled_outputs(
                    query_table.table, [query_table.target_attribute]
                )
            else:
                raise ValueError(f'Unknown Query Table Type: {query_table.type}')
        else:
            for row in query_table.table:
                if entity_id != row['entityId'] and not self.rank_evidences_by_table:
                    continue

                if query_table.type == 'retrieval':
                    pooled_output = self.entity_biencoder.encode_entities_and_return_pooled_outputs([row])[0]
                elif query_table.type == 'augmentation':
                    pooled_output = self.entity_biencoder.encode_entities_and_return_pooled_outputs([row], [query_table.target_attribute])[0]
                else:
                    raise ValueError(f'Unknown Query Table Type: {query_table.type}')
                
                entity_vectors.append(pooled_output)

        # Query Faiss index
        torch.cuda.empty_cache()
        entity_vectors = np.array(entity_vectors).astype('float32')
        D, I = self.index.search(entity_vectors, evidence_count)

        # Determine Elasticsearch index name
        index_name = determine_es_index_name(self.dataset_name, clusters=self.clusters)

        for i in range(len(I)):
            try:
                entity_result = self.query_tables_index_by_id(I[i], index_name)
            except Exception as e:
                self.logger.error(f'Error querying Elasticsearch: {e}')
                continue

            hits = entity_result['hits']['hits']
            for hit in hits[:evidence_count]:
                found_value = None
                if query_table.type == 'augmentation' and query_table.target_attribute in hit['_source']:
                    found_value = hit['_source'][query_table.target_attribute]
                
                rowId = hit['_source']['row_id']
                table_name = hit['_source']['table']
                new_entity_id = entity_id if entity_id is not None else query_table.table[i]['entityId']

                evidence = RetrievalEvidence(
                    evidence_id, query_table.identifier, new_entity_id, table_name, rowId, hit['_source']
                )

                # Determine similarity score
                similarity_not_found = True
                for distance, instance in zip(D[i], I[i]):
                    if int(instance) == int(hit['_id']):
                        evidence.scores[self.name] = distance.item()
                        evidence.similarity_score = distance.item()
                        similarity_not_found = False
                        break

                if similarity_not_found:
                    self.logger.warning(f'Could not find similarity score for entity {hit["_id"]} in Faiss index!')

                evidences.append(evidence)
                self.logger.debug(f'Added evidence {evidence_id} to query table')
                evidence_id += 1

        return evidences
