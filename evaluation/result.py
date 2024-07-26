"""
This code is based on the code written by:
Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines.
In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer.

The code can be found at:
https://github.com/wbsg-uni-mannheim/SC-Block
"""

import json
import os

class Result:
    def __init__(self, querytable, retrieval_strategy, similarity_reranker, k_interval, type, voting, split, seen):
        self.querytable = querytable
        self.retrieval_strategy = self._get_retrieval_strategy_info(retrieval_strategy)
        self.similarity_reranker = similarity_reranker.name if similarity_reranker else None
        self.k_interval = k_interval
        self.ranking_level = type
        self.voting_strategy = voting
        self.split = split
        self.seen = seen

        # Initialize result storage dictionaries
        self._initialize_result_dicts()

    def _get_retrieval_strategy_info(self, retrieval_strategy):
        """
        Extract relevant information from the retrieval strategy.
        """
        if retrieval_strategy:
            return {
                'name': retrieval_strategy.name,
                'model_name': getattr(retrieval_strategy, 'model_name', None),
                'pooling': getattr(retrieval_strategy, 'pooling', None),
                'similarity': getattr(retrieval_strategy, 'similarity', None)
            }
        return None

    def _initialize_result_dicts(self):
        """
        Initialize result storage dictionaries for various metrics and attributes.
        """
        self.precision_per_entity = {}
        self.recall_per_entity = {}
        self.f1_per_entity = {}
        self.not_annotated_per_entity = {}
        self.fusion_accuracy = {}
        self.no_retrieved_verified_evidences = {}
        self.no_retrieved_evidences = {}
        self.no_verified_evidences = {}
        self.seen_training = {}
        self.corner_cases = {}
        self.retrieved_corner_cases = {}
        self.serialization = {}

        self.different_values = {}
        self.different_evidences = {}
        self.different_tables = {}
        self.found_values = {}
        self.predicted_values = {}
        self.target_values = {}

        for k in self.k_interval:
            self.precision_per_entity[k] = {}
            self.recall_per_entity[k] = {}
            self.f1_per_entity[k] = {}
            self.not_annotated_per_entity[k] = {}
            self.fusion_accuracy[k] = {}
            self.no_retrieved_verified_evidences[k] = {}
            self.no_retrieved_evidences[k] = {}
            self.no_verified_evidences[k] = {}
            self.seen_training[k] = {}
            self.corner_cases[k] = {}
            self.retrieved_corner_cases[k] = {}

            self.different_values[k] = {}
            self.different_evidences[k] = {}
            self.different_tables[k] = {}
            self.found_values[k] = {}
            self.predicted_values[k] = {}

    def save_result(self, file_name, with_evidences=False):
        """
        Save the results to a JSON file.

        :param file_name: Name of the file to save results
        :param with_evidences: Whether to include evidence details in the output
        """
        path_to_results = f'./results/{self.querytable.schema_org_class}'
        os.makedirs(path_to_results, exist_ok=True)

        file_path = os.path.join(path_to_results, file_name)

        with open(file_path, 'a+', encoding='utf-8') as f:
            unpacked_results = self.unpack(with_evidences)
            for unpacked_result in unpacked_results:
                json.dump(unpacked_result, f)
                f.write('\n')

    def unpack(self, with_evidences=False):
        """
        Unpack results into a list of dictionaries.

        :param with_evidences: Whether to include evidence details in the unpacked results
        :return: List of dictionaries containing unpacked results
        """
        results = []
        template_row = self._get_template_row()

        for k in self.k_interval:
            k_row = template_row.copy()
            k_row['k'] = k
            for entity_id in self.f1_per_entity[k].keys():
                row = k_row.copy()
                row.update(self._get_entity_results(k, entity_id, with_evidences))
                results.append(row)

        return results

    def _get_template_row(self):
        """
        Create a template row dictionary based on query table type.

        :return: Dictionary containing template row information
        """
        common_data = {
            'ranking_level': self.ranking_level,
            'querytableId': self.querytable.identifier,
            'schemaOrgClass': self.querytable.schema_org_class,
            'gt_table': self.querytable.gt_table,
            'retrieval_strategy': self.retrieval_strategy['name'] if self.retrieval_strategy else None,
            'model_name': self.retrieval_strategy.get('model_name'),
            'pooling': self.retrieval_strategy.get('pooling'),
            'similarity': self.retrieval_strategy.get('similarity'),
            'split': self.split,
            'seen': self.seen,
            'similarity_reranker': self.similarity_reranker,
            'assemblingStrategy': self.querytable.assembling_strategy,
            'contextAttributes': ', '.join(self.querytable.context_attributes)
        }

        if self.querytable.type == 'retrieval':
            return common_data

        if self.querytable.type == 'augmentation':
            return {**common_data,
                    'useCase': self.querytable.use_case,
                    'targetAttribute': self.querytable.target_attribute,
                    'voting_strategy': self.voting_strategy}

        raise ValueError(f'Query Table Type {self.querytable.type} is not defined!')

    def _get_entity_results(self, k, entity_id, with_evidences):
        """
        Extract results for a specific entity and k value.

        :param k: Top-k value
        :param entity_id: Entity identifier
        :param with_evidences: Whether to include evidence details
        :return: Dictionary containing results for the specified entity
        """
        results = {
            'entityId': entity_id,
            'serialization': self.serialization.get(entity_id),
            'precision': self.precision_per_entity[k].get(entity_id),
            'recall': self.recall_per_entity[k].get(entity_id),
            'f1': self.f1_per_entity[k].get(entity_id),
            'not_annotated': self.not_annotated_per_entity[k].get(entity_id),
            'retrieved_verified_evidences': self.no_retrieved_verified_evidences[k].get(entity_id),
            'retrieved_evidences': self.no_retrieved_evidences[k].get(entity_id),
            'seen_training': self.seen_training[k].get(entity_id),
            'corner_cases': self.corner_cases[k].get(entity_id),
            'retrieved_corner_cases': self.retrieved_corner_cases[k].get(entity_id),
            'verified_evidences': self.no_verified_evidences[k].get(entity_id),
            'different_tables': self.different_tables[k].get(entity_id)
        }

        if with_evidences:
            results['different_evidences'] = self.different_evidences[k].get(entity_id)

        if self.querytable.type == 'augmentation':
            results.update({
                'fusion_accuracy': self.fusion_accuracy[k].get(entity_id),
                'different_values': self.different_values[k].get(entity_id),
                'found_values': self.found_values[k].get(entity_id),
                'target_value': self.target_values.get(entity_id),
                'predicted_value': self.predicted_values[k].get(entity_id)
            })

        # Calculate and add evidence statistics
        evidence_statistics = self.querytable.calculate_evidence_statistics_of_row(entity_id)
        results.update({
            'evidences': evidence_statistics[0],
            'correct_entity': evidence_statistics[3],
            'not_correct_entity': evidence_statistics[4]
        })

        if self.querytable.type == 'augmentation':
            results.update({
                'correct_value_entity': evidence_statistics[1],
                'rel_value_entity': evidence_statistics[2]
            })

        return results
