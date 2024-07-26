"""
This code is based on the code written by:
Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines.
In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer.

The code can be found at:
https://github.com/wbsg-uni-mannheim/SC-Block
"""

import itertools
import logging
import time
from tqdm import tqdm
import sys

# Add the current directory to the system path to import local modules
sys.path.append('.')
from evaluation.result import Result
from retrieval.entity_serializer import EntitySerializer

def evaluate_query_table(query_table, retrieval_strategy, similarity_reranker,
                         k_interval, voting='weighted', collect_result_context=False, split=None, experiment_type='retrieval'):
    """
    Evaluate the performance of a query table by calculating precision and recall for the provided evidences.
    
    :param Querytable query_table: The query table to evaluate.
    :param str experiment_type: Type of experiment, either 'retrieval' or 'augmentation'.
    :param RetrievalStrategy retrieval_strategy: The strategy used for retrieval.
    :param list[int] k_interval: List of integers specifying the intervals for evaluating the retrieved evidences.
    :param str voting: Voting strategy, either 'simple' (majority vote) or 'weighted' (based on similarity scores).
    :param bool collect_result_context: Whether to collect context information for results.
    :param str split: The dataset split to consider ('train', 'valid', 'test', or None).
    :param str experiment_type: Specifies the type of experiment ('retrieval' or 'augmentation').
    """
    logger = logging.getLogger()
    logger.info(f'Evaluate query table {query_table.identifier}: {query_table.assembling_strategy}')

    # Ranking levels to use
    ranking_lvls = ['3,2,1 - Correct Entity']  # Current ranking level configuration
    results = []
    splits = ['train', 'valid', 'test', None]  # Dataset splits
    seen_values = ['seen', 'left_seen', 'right_seen', 'unseen', 'all']  # Seen status

    # Aggregate scores to similarity score for each evidence
    for evidence in query_table.retrieved_evidences:
        evidence.aggregate_scores_to_similarity_score()

    entity_serializer = EntitySerializer(query_table.schema_org_class, None)

    for ranking_lvl in ranking_lvls:
        for split, seen in itertools.product(splits, seen_values):
            if split is None and seen in ['both_seen', 'left_seen', 'right_seen', 'none_seen']:
                continue

            result = Result(query_table, retrieval_strategy, similarity_reranker, k_interval,
                            ranking_lvl, voting, split, seen)

            if not query_table.has_verified_evidences():
                logger.warning(f'No verified evidences found for query table {query_table.identifier}!')
                return results

            if experiment_type == 'augmentation':
                relevance_classification = {'3 - Correct Value and Entity': [3],
                                            '3,2 - Relevant Value and Correct Entity': [3, 2],
                                            '3,2,1 - Correct Entity': [3, 2, 1]}[ranking_lvl]

                positive_evidences = [evidence for evidence in query_table.verified_evidences
                                     if evidence.scale in relevance_classification]
                negative_evidences = [evidence for evidence in query_table.verified_evidences
                                     if evidence.scale not in relevance_classification]

                time.sleep(15)

                # Filter evidences - Remove ground truth tables
                positive_evidences = retrieval_strategy.filter_evidences_by_ground_truth_tables(positive_evidences)
                negative_evidences = retrieval_strategy.filter_evidences_by_ground_truth_tables(negative_evidences)
            else:
                positive_evidences = [evidence for evidence in query_table.verified_evidences
                                      if evidence.signal]
                negative_evidences = [evidence for evidence in query_table.verified_evidences
                                      if not evidence.signal]

                # Filter by split and seen status
                if split in ['train', 'valid', 'test']:
                    positive_evidences = [evidence for evidence in positive_evidences
                                          if evidence.split == split]
                    negative_evidences = [evidence for evidence in negative_evidences if evidence.split == split]

                if seen in ['both_seen', 'left_seen', 'right_seen', 'none_seen']:
                    positive_evidences = [evidence for evidence in positive_evidences
                                          if evidence.seen_training == seen]
                    negative_evidences = [evidence for evidence in negative_evidences
                                          if evidence.seen_training == seen]

            for row in tqdm(query_table.table):
                all_retrieved_evidences = [evidence for evidence in query_table.retrieved_evidences if
                                           evidence.entity_id == row['entityId']]

                if split in ['train', 'valid', 'test']:
                    all_retrieved_evidences = [evidence for evidence in all_retrieved_evidences if
                                               (evidence in positive_evidences) or (evidence in negative_evidences)]

                all_retrieved_evidences.sort(key=lambda evidence: evidence.similarity_score, reverse=True)

                if query_table.type == 'augmentation':
                    result.target_values[row['entityId']] = row[query_table.target_attribute]

                no_verified_evidences = sum([1 for evidence in positive_evidences if evidence.entity_id == row['entityId']])
                scores_per_k = [calculate_aggregated_evidence_scores(all_retrieved_evidences, positive_evidences, k)
                                for k in k_interval]

                for k, rel_retrieved_evidences, no_retrieved_evidences, no_retrieved_verified_evidences in scores_per_k:
                    if k == 1 and experiment_type == 'augmentation' and voting == 'weighted':
                        continue

                    # Update precision and recall values
                    precision = 0
                    recall = 0
                    f1 = 0
                    no_not_annotated = 0

                    result.serialization[row['entityId']] = entity_serializer.convert_to_str_representation(row)
                    result.no_retrieved_verified_evidences[k][row['entityId']] = no_retrieved_verified_evidences
                    result.no_retrieved_evidences[k][row['entityId']] = no_retrieved_evidences
                    result.no_verified_evidences[k][row['entityId']] = no_verified_evidences
                    result.not_annotated_per_entity[k][row['entityId']] = no_not_annotated

                    no_retrieved_corner_case_evidences = 0
                    result_evidences = []

                    for retrieved_evidence in rel_retrieved_evidences[:k]:
                        if retrieved_evidence.context is not None:
                            result_evidence = retrieved_evidence.context.copy()
                            result_evidence['similarity_score'] = retrieved_evidence.similarity_score
                            result_evidence['relevant_evidence'] = retrieved_evidence in positive_evidences
                            result_evidence['table'] = retrieved_evidence.table
                            result_evidence['row_id'] = retrieved_evidence.row_id
                            result_evidences.append(result_evidence)

                    found_positive_evidences = []

                    if len(found_positive_evidences) > 0:
                        result.seen_training[k][row['entityId']] = found_positive_evidences[0].seen_training
                    else:
                        result.seen_training[k][row['entityId']] = None

                    result.corner_cases[k][row['entityId']] = 0
                    result.retrieved_corner_cases[k][row['entityId']] = no_retrieved_corner_case_evidences
                    result.different_evidences[k][row['entityId']] = result_evidences
                    result.different_tables[k][row['entityId']] = None

                    if experiment_type == 'augmentation':
                        values = []
                        similarities = []

                        for evidence in rel_retrieved_evidences[:k]:
                            if evidence.value is not None:
                                value = (', '.join([v for v in evidence.value if v is not None])
                                         if isinstance(evidence.value, list)
                                         else str(evidence.value))
                                values.append(value)
                                similarities.append(evidence.similarity_score)

                        if voting == 'simple':
                            value_counts = [(value, values.count(value)) for value in set(values)]
                        elif voting == 'weighted':
                            dict_value_similarity = {}
                            total_similarity = 0
                            initial_value_counts = {value: values.count(value) for value in set(values)}
                            for value, similarity_score in zip(values, similarities):
                                if value not in dict_value_similarity:
                                    dict_value_similarity[value] = 0
                                dict_value_similarity[value] += similarity_score
                                total_similarity += similarity_score

                            dict_value_norm_similarity = {value: sim/initial_value_counts[value]
                                                           for value, sim in dict_value_similarity.items()}
                            if total_similarity > 0:
                                value_counts = [(value, sim/total_similarity) for value, sim in dict_value_norm_similarity.items()]
                            else:
                                value_counts = [(value, 0) for value in dict_value_norm_similarity.keys()]
                        else:
                            raise ValueError(f'Unknown voting strategy {voting}.')

                        dict_value_counts = [{'value': value_count[0], 'count': value_count[1]} for value_count in value_counts]
                        value_counts.sort(key=lambda x: x[1], reverse=True)

            results.append(result)

    return results

def calculate_aggregated_evidence_scores(all_retrieved_evidences, positive_evidences, k):
    """Calculate aggregated evidence scores"""
    rel_retrieved_evidences = all_retrieved_evidences[:k]
    no_retrieved_evidences = len(rel_retrieved_evidences)

    no_retrieved_verified_evidences = sum([1 for evidence in rel_retrieved_evidences if evidence in positive_evidences])
    
    return k, rel_retrieved_evidences, no_retrieved_evidences, no_retrieved_verified_evidences
