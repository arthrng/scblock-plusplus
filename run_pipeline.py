import logging
import time
from datetime import datetime
import numpy as np
import torch
from torch.multiprocessing import set_start_method
from random import randrange, seed as random_seed
import os
import sys
from tqdm import tqdm
from evaluation.aggregate_results import aggregate_results, save_aggregated_result
from evaluation.evaluate_query_tables import evaluate_query_table
from retrieval.query_by_neural_entity import QueryByNeuralEntity
from query_and_index.querytable import load_query_table_from_file, get_gt_tables, get_query_table_paths
from matchers.select_matcher import select_matcher

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append('.')

def set_seed(seed):
    """Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`."""
    random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_experiments_from_configuration(dataset_name, retrieval_strategy, matching_strategy):
    """Runs experiments based on the provided configuration."""
    logger = logging.getLogger()

    # Get schema corresponding to the dataset
    context_attributes = get_context_attributes(dataset_name)
    experiment_type = 'retrieval'

    # Load query tables
    query_table_paths = load_query_table_paths(dataset_name)

    string_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Values of k to be tested
    k_range = [1, 5, 10, 20, 40, 80]
    logger.info(f'Will test the following values of k: {k_range}')

    save_results_with_evidences = True
    clusters = False

    file_name = get_results_file_name(retrieval_strategy, matching_strategy)

    print(retrieval_strategy['bi-encoder'])

    # Run experiments by combining pipelines and query tables
    for k in k_range:
        results, execution_times = run_experiments(retrieval_strategy, matching_strategy, query_table_paths, dataset_name, k, context_attributes, clusters=clusters)
        if results is not None:
            for result in results:
                result.save_result(file_name, save_results_with_evidences)
            aggregated_result = aggregate_results(results, k, execution_times)
            print(aggregated_result)
            save_aggregated_result(aggregated_result, file_name)

    logger.info('Finished running experiments!')

def get_context_attributes(dataset_name):
    """Returns context attributes based on the dataset name."""
    if dataset_name == 'walmart-amazon':
        return ['name', 'category', 'brand', 'modelno', 'price']
    elif 'wdcproducts' in dataset_name:
        return ['name', 'brand', 'description', 'price', 'pricecurrency']
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

def load_query_table_paths(dataset_name):
    """Loads query table paths for the given dataset_name."""
    query_table_paths = []
    for gt_table in get_gt_tables(f'./data/querytables/{dataset_name}'):
        query_table_paths.append(get_query_table_paths(f'./data/querytables/{dataset_name}', gt_table))
    return query_table_paths

def get_results_file_name(retrieval_strategy, matching_strategy):
    """Generates a file name for the results based on the retrieval and matching strategies."""
    if matching_strategy is None:
        return f"results_{retrieval_strategy['file_name']}.json"
    else:
        return f"results_{retrieval_strategy['file_name']}_{matching_strategy['name']}.json"

def run_experiments(retrieval_strategy, matching_strategy, query_table_paths, dataset_name, evidence_count, context_attributes=None, clusters=False):
    """Run Pipeline on query tables."""
    time.sleep(randrange(30))
    logger = logging.getLogger()

    # Initialize strategy
    retrieval_strategy = QueryByNeuralEntity(
        dataset_name,
        retrieval_strategy['bi-encoder'],
        clusters,
        retrieval_strategy['model_name'],
        retrieval_strategy['base_model'],
        retrieval_strategy['with_projection'],
        retrieval_strategy['projection'],
        retrieval_strategy['pooling'],
        retrieval_strategy['similarity']
    )
    matching_strategy = select_matcher(matching_strategy, dataset_name, context_attributes)
    
    logger.info('Run experiments on {} query tables'.format(len(query_table_paths)))
    results = []
    execution_times = []

    materialized_pairs = []

    for query_table_path in tqdm(query_table_paths):
        query_table = load_query_table_from_file(query_table_path)
        query_table.retrieved_evidences, execution_times_per_run = retrieve_evidences_with_pipeline(query_table, retrieval_strategy, evidence_count, matching_strategy)
        materialized_pairs.extend(query_table.materialize_pairs())
        execution_times.append(execution_times_per_run)
        k_intervals = [evidence_count]
        new_results = evaluate_query_table(query_table, retrieval_strategy, matching_strategy, k_intervals, collect_result_context=True)
        results.extend(new_results)

    aggregated_execution_times = {key: sum([execution_time[key] for execution_time in execution_times]) for key in execution_times[0]}

    logger.info('Finished running experiments on subset of query tables!')
    return results, aggregated_execution_times

def retrieve_evidences_with_pipeline(query_table, retrieval_strategy, evidence_count, similarity_re_ranker, entity_id=None):
    """Retrieve evidences using the pipeline strategy."""
    execution_times = {}
    start_time = time.time()

    evidences = retrieval_strategy.retrieve_evidence(query_table, evidence_count, entity_id)
    retrieval_time = time.time()
    execution_times['retrieval_time'] = retrieval_time - start_time
    
    # Filter evidences by ground truth tables
    evidences = retrieval_strategy.filter_evidences_by_ground_truth_tables(evidences)

    # Run re-ranker
    if similarity_re_ranker is not None:
        evidences = similarity_re_ranker.re_rank_evidences(query_table, evidences)
        similarity_re_ranking_time = time.time()
        execution_times['sim_reanker_time'] = similarity_re_ranking_time - retrieval_time

    execution_times['complete_execution_time'] = time.time() - start_time
    print(execution_times['complete_execution_time'])

    return evidences, execution_times

def collect_results_of_finished_experiments(async_results, file_name, evidence_count, with_evidences=True, with_extended_results=False):
    """Collect results and write them to file."""
    logger = logging.getLogger()
    collected_results = []
    for async_result in async_results:
        if async_result.ready():
            results, execution_times = async_result.get()
            collected_results.append(async_result)

            # Save query table to file
            if results is not None:
                logger.info('Will collect {} results now!'.format(len(results)))
                if with_extended_results:
                    for result in results:
                        result.save_result(file_name, with_evidences)

                aggregated_result = aggregate_results(results, evidence_count, execution_times)
                save_aggregated_result(aggregated_result, file_name)

    async_results = [async_result for async_result in async_results if async_result not in collected_results]

    return async_results

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    set_start_method('spawn')

    dataset_names = ['walmart-amazon', 'wdcproducts20pair', 'wdcproducts80pair']
    encoders = {
        'scblock': ['scblock', 'scblock-with-adaflood', 'scblock-with-flooding-0.005'],
        'barlowtwins': ['barlowtwins'],
        'sbert': ['sbert']
    }
    alphas = {
        'amazon-google': 0.2,
        'walmart-amazon': 0.4,
        'wdcproducts20pair': 0.6,
        'wdcproducts80pair': 0.1
    }

    for dataset_name in dataset_names:
        for encoder_name, model_names in encoders.items():
            for model_name in model_names:
                print(encoder_name, model_name)
                if encoder_name == 'sbert':
                    model_path = f'./blockers/{encoder_name}/saved_blockers/{dataset_name}'
                else:
                    model_path = f'./blockers/{encoder_name}/saved_blockers/{dataset_name}/{model_name}.pt'

                retrieval_strategy = {
                    'name': 'query_by_neural_entity',
                    'bi-encoder': encoder_name,
                    'file_name': model_name,
                    'model_name': model_path,  # /{file_name}.pt
                    'base_model': 'roberta-base',
                    'with_projection': False,
                    'projection': 768,
                    'pooling': True,
                    'similarity': 'cos'
                }

                matcher = 'supconmatch'
                file_name = 'supconmatch.pt'
                matching_strategy = {
                    'name': matcher,
                    'model_name': f'./matchers/{matcher}/saved_matchers/{dataset_name}/{file_name}',
                    'base_model': 'roberta-base',
                    'matcher': True,
                    'alpha': alphas[dataset_name]
                }

                run_experiments_from_configuration(dataset_name, retrieval_strategy, matching_strategy)
