"""
This code is based on the code written by:
Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines.
In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer.

The code can be found at:
https://github.com/wbsg-uni-mannheim/SC-Block
"""

import logging
import os
import time
from multiprocessing import Process, Queue
from elasticsearch import Elasticsearch
from tqdm import tqdm
from es_helper import determine_es_index_name
from faiss_collector import FaissIndexCollector

sys.path.append('.')
import sys
from retrieval.query_by_entity import QueryByEntity
from blockers.select_blocker import select_blocker


sys.path.append('.')

def load_data(dataset, blocker_name, model_name, model_path, base_model, pooling, similarity_measure, dimensions, clusters, batch_size):
    """Load data, generate embeddings, and index entities using FAISS.

    Args:
        dataset (str): Name of the dataset.
        blocker_name (str): Name of the blocker.
        model_name (str): Name of the model.
        model_path (str): Path to the model file.
        base_model (str): Base model name.
        pooling (bool): Whether to use pooling.
        similarity_measure (str): Similarity measure to use.
        dimensions (int): Dimensions of the embeddings.
        clusters (bool): Whether to use clustering.
        batch_size (int): Number of entities to process per batch.
    """
    logger = logging.getLogger()

    encoder_configuration = {
        'blocker_name': blocker_name, 
        'model_name': model_name,
        'model_path': model_path, 
        'base_model': base_model, 
        'projection': dimensions, 
        'pooling': pooling, 
        'normalize': True, 
        'similarity_measure': similarity_measure, 
        'dimensions': dimensions
    }

    logger.info('Chosen Model {} for indexing schema org class {}'.format(model_path, dataset))

    start_time = time.time()

    strategy = QueryByEntity(dataset)
    _es = Elasticsearch(
        [{'host': 'localhost', 'port': 9200, 'scheme': 'http'}], 
        basic_auth=("elastic", 'PhB9-HXKmUGaiHKY7AE-'), 
        timeout=15, 
        max_retries=3, 
        retry_on_timeout=True
    )
    
    entity_index_name = determine_es_index_name(dataset, clusters=clusters)
    logger.info('Creating FAISS index for ES index {}'.format(entity_index_name))
    
    no_entities = int(_es.cat.count(index=entity_index_name, params={"format": "json"})[0]['count'])
    final_step = int(no_entities / batch_size) + 1

    # Initialize Faiss collector
    faiss_collector = FaissIndexCollector(
        dataset, model_path, pooling, similarity_measure, final_step,
        dimensions, clusters
    )

    input_q = Queue()
    output_q = Queue()
    processes = []

    # Get GPU workers
    gpu_workers = os.environ.get('CUDA_VISIBLE_DEVICES', '1').split(',')
    logger.info('Using GPUs: {}'.format(gpu_workers))

    for gpu_n in gpu_workers:
        for i in range(2):  # Start multiple processes - 1 per GPU
            p = Process(target=generate_embeddings,
                        args=(encoder_configuration, dataset, input_q, output_q, gpu_n))
            p.start()
            processes.append(p)

    for i in tqdm(range(final_step)):
        start = i * batch_size
        end = min(start + batch_size, no_entities)

        # Retrieve entities
        entities = strategy.query_tables_index_by_id(range(start, end), entity_index_name)
        if len(entities['hits']['hits']) != end - start:
            logger.warning('Did not receive all entities from {}!'.format(entity_index_name))
        
        # Encode entities
        entities = [entity['_source'] for entity in entities['hits']['hits']]
        input_q.put((i, entities))

        # Collect results
        collect = True
        wait_and_persist_indices = i % 50000 == 0
        if input_q.qsize() > 80 or output_q.qsize() > 20:
            faiss_collector = collect_faiss_entities(output_q, faiss_collector, wait_and_persist_indices, wait_and_persist_indices)
            logger.info('Input Size: {} - Output Size: {}'.format(input_q.qsize(), output_q.qsize()))

        if input_q.qsize() > 80:
            logger.info('{} Configurations available for processing'.format(input_q.qsize()))
            time.sleep(input_q.qsize() * 0.01)

    # Final collection and saving
    while True:
        faiss_collector = collect_faiss_entities(output_q, faiss_collector, False, False)
        if faiss_collector.next_representation == final_step:
            faiss_collector = collect_faiss_entities(output_q, faiss_collector, True, True)
            break

    for p in processes:
        p.terminate()
        p.join()
        p.close()

    input_q.close()
    output_q.close()

    indexing_time = time.time() - start_time

    logger.info('Added {} entities to FAISS indices'.format(faiss_collector.collected_entities))
    logger.info('Indexing time: {}'.format(indexing_time))


def generate_embeddings(encoder_config, dataset, input_q, output_q, gpu_n):
    """Generate embeddings of entities.

    Args:
        encoder_config (dict): Configuration for the encoder.
        dataset (str): Name of the dataset.
        input_q (Queue): Queue to get entities from.
        output_q (Queue): Queue to put encoded entities in.
        gpu_n (str): GPU number to use.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_n)

    # Initialize Bi-Encoder
    blocker = select_blocker(
        encoder_config['blocker_name'], 
        encoder_config['model_path'], 
        dataset
    )
    collection_identifier = f'entity_vector_{encoder_config["pooling"]}_norm'

    while True:
        i, entities = input_q.get()

        # Encode entities
        pooled_outputs = blocker.encode_entities_and_return_pooled_outputs(entities)

        if len(entities) == 1:
            entities[0][collection_identifier] = pooled_outputs
        else:
            for entity, pooled_output in zip(entities, pooled_outputs):
                entity[collection_identifier] = pooled_output

        output_q.put((i, entities))


def collect_faiss_entities(output_q, faiss_collector, wait_for_encodings, persist_indices):
    """Collect entities in FAISS indices and return the number of collected entities.

    Args:
        output_q (Queue): Queue to get encoded entities from.
        faiss_collector (FaissIndexCollector): The FAISS index collector.
        wait_for_encodings (bool): Whether to wait for encodings to complete.
        persist_indices (bool): Whether to persist the indices.

    Returns:
        FaissIndexCollector: Updated FAISS index collector.
    """
    logger = logging.getLogger()

    if not output_q.empty():
        faiss_collector.unsaved_representations.append(output_q.get(wait_for_encodings))
        while output_q.qsize() > 10:
            if not output_q.empty():
                faiss_collector.unsaved_representations.append(output_q.get(wait_for_encodings))
            logger.debug('Output size during collection: {}'.format(output_q.qsize()))

        entities = faiss_collector.next_savable_entities()

        while entities is not None:
            faiss_collector.collected_entities += len(entities)
            faiss_collector.initialize_entity_representations()
            for entity in entities:
                faiss_collector.collect_entity_representation(entity)
            faiss_collector.add_entity_representations_to_indices()
            entities = faiss_collector.next_savable_entities()
        
        logger.info('Finished processing entities.')

    if persist_indices:
        faiss_collector.save_indices()

    return faiss_collector

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Configuration parameters
    blocker_name = 'scblock'
    model_name = 'scblock-with-adaflood'
    dataset = 'wdcproducts80pair'
    
    # Load data and perform indexing
    load_data(
        dataset=dataset, 
        blocker_name=blocker_name,
        model_name=model_name, 
        model_path=f'./blockers/{blocker_name}/saved_blockers/{dataset}/{model_name}.pt',
        base_model='roberta-base', 
        pooling=True, 
        similarity_measure='cos',
        dimensions=768,
        clusters=False, 
        batch_size=128
    )
