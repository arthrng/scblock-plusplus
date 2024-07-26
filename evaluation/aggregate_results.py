import itertools
import json
import os
import logging
from typing import List, Dict, Any

from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def aggregate_results(results: List[Any], k: int, execution_times: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate results from multiple retrieval experiments.

    Args:
        results (List[Any]): A list of result objects to be aggregated.
        k (int): The value of 'k' used in the retrieval process.
        execution_times (Dict[str, Any]): Dictionary containing execution times.

    Returns:
        Dict[str, Any]: Aggregated results including metrics like precision, recall, F1 score.
    """
    splits = ['train', 'valid', 'test', None]
    seen_values = ['seen', 'left_seen', 'right_seen', 'unseen', 'all']

    # Initialize counters
    no_retrieved_verified_evidences = {f'{split}_{seen_value}': 0 for split, seen_value in itertools.product(splits, seen_values)}
    no_retrieved_evidences = {f'{split}_{seen_value}': 0 for split, seen_value in itertools.product(splits, seen_values)}
    no_verified_evidences = {f'{split}_{seen_value}': 0 for split, seen_value in itertools.product(splits, seen_values)}
    no_corner_cases = {f'{split}_{seen_value}': 0 for split, seen_value in itertools.product(splits, seen_values)}
    no_retrieved_corner_cases = {f'{split}_{seen_value}': 0 for split, seen_value in itertools.product(splits, seen_values)}

    aggregated_result = {'model_name': None, 'pooling': None, 'retrieval_strategy': None, 'similarity_reranker': None, 'voting_strategy': None}
    
    for split, seen_value in itertools.product(splits, seen_values):
        if split is None and seen_value in ['seen', 'unseen']:
            continue

        subset_results = [result for result in results if getattr(result, 'seen') == seen_value and getattr(result, 'split') == split]

        for result in tqdm(subset_results, desc=f"Processing {split}_{seen_value}"):
            try:
                no_retrieved_verified_evidences[f'{split}_{seen_value}'] += sum(result.no_retrieved_verified_evidences[k].values())
                no_retrieved_evidences[f'{split}_{seen_value}'] += sum(result.no_retrieved_evidences[k].values())
                no_verified_evidences[f'{split}_{seen_value}'] += sum(result.no_verified_evidences[k].values())
                no_corner_cases[f'{split}_{seen_value}'] += sum(result.corner_cases[k].values())
                no_retrieved_corner_cases[f'{split}_{seen_value}'] += sum(result.retrieved_corner_cases[k].values())

                dict_result = result.__dict__
                for key in aggregated_result:
                    if aggregated_result[key] is None:
                        aggregated_result[key] = dict_result.get(key, None)
                    else:
                        if aggregated_result[key] != dict_result.get(key, None):
                            raise ValueError(f'Inconsistent setup found: {aggregated_result[key]} vs {dict_result.get(key, None)} for {key}')
            except AttributeError as e:
                logger.error(f"Error processing result: {e}")

    options = [f'{split}_{seen_value}' for split, seen_value in itertools.product(splits, seen_values)]
    
    for option in options:
        if option in ['None_seen', 'None_unseen']:
            continue

        retrieved_verified = no_retrieved_verified_evidences.get(option, 0)
        retrieved = no_retrieved_evidences.get(option, 0)
        verified = no_verified_evidences.get(option, 0)
        corner_case_retrieved = no_retrieved_corner_cases.get(option, 0)
        corner_cases = no_corner_cases.get(option, 0)

        aggregated_result[f'no_retrieved_verified_{option}'] = retrieved_verified
        aggregated_result[f'no_retrieved_{option}'] = retrieved
        aggregated_result[f'no_verified_{option}'] = verified

        aggregated_result[f'precision_{option}'] = (retrieved_verified / retrieved) if retrieved > 0 else 0
        aggregated_result[f'recall_{option}'] = (retrieved_verified / verified) if verified > 0 else 0
        aggregated_result[f'f1_{option}'] = (2 * aggregated_result[f'precision_{option}'] * aggregated_result[f'recall_{option}'] /
                                              (aggregated_result[f'precision_{option}'] + aggregated_result[f'recall_{option}'])) \
                                              if (aggregated_result[f'precision_{option}'] + aggregated_result[f'recall_{option}']) > 0 else 0
        aggregated_result[f'corner_case_recall_{option}'] = (corner_case_retrieved / corner_cases) if corner_cases > 0 else 0

    aggregated_result['schema_org_class'] = results[0].querytable.schema_org_class if results else None
    aggregated_result['k'] = k
    aggregated_result.update(execution_times)

    return aggregated_result

def save_aggregated_result(aggregated_result: Dict[str, Any], file_name: str):
    """
    Save aggregated results to a JSON file.

    Args:
        aggregated_result (Dict[str, Any]): Aggregated results to be saved.
        file_name (str): Name of the file where results will be saved.
    """
    path_to_results = f"./results/{aggregated_result.get('schema_org_class', 'unknown')}"
    os.makedirs(path_to_results, exist_ok=True)

    path_to_results = os.path.join(path_to_results, f'aggregated_{file_name}')

    with open(path_to_results, 'a+', encoding='utf-8') as f:
        json.dump(aggregated_result, f)
        f.write('\n')
