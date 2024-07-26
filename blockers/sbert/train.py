import csv
import logging
import math
import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
from sentence_transformers.readers import InputExample

sys.path.append('..')
from retrieval.entity_serializer import EntitySerializer

def sbert_finetuning(dataset, model_pretrained_checkpoint, epochs, output_dir):
    """
    Fine-tunes a Sentence-BERT model on a specific dataset.

    Args:
        dataset (str): Name of the dataset.
        model_pretrained_checkpoint (str): Pretrained model checkpoint to start from.
        epochs (int): Number of epochs to train.
        output_dir (str): Directory to save the fine-tuned model.
    """

    # Setup logging
    logger = logging.getLogger()
    logger.info('Selected data set {}'.format(dataset))

    # Configuration for training
    train_batch_size = 128
    num_epochs = 10
    model_save_path = output_dir
    logger.warning(model_save_path)

    # Determine the model path
    if dataset in model_pretrained_checkpoint:
        model_name = '{}/models/open_book/{}'.format(os.environ['DATA_DIR'], model_pretrained_checkpoint)
        logger.info('Path to model: ' + model_name)
    
    logger.info('Path to model: ' + model_pretrained_checkpoint)

    # Initialize the transformer model
    word_embedding_model = models.Transformer('roberta-base')
    word_embedding_model.max_seq_length = 128

    # Add special tokens if the dataset is not in the pretrained checkpoint
    if dataset not in model_pretrained_checkpoint:
        special_tokens_dict = {'additional_special_tokens': ['[COL]', '[VAL]']}
        word_embedding_model.tokenizer.add_special_tokens(special_tokens_dict)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        logger.info('Added special tokens [COL], [VAL]')

    # Add pooling layer to the model
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    # Create the SentenceTransformer model
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Load and prepare the dataset
    logger.info("Read {} dataset".format(dataset))
    record_dict = {}
    source_field = '__source'
    left_source = 'table_a'
    right_source = 'table_b'
    path_to_data = f'./data/raw/{dataset}'
    entity_serializer = EntitySerializer(schema_org_class=dataset)

    # Load records from tableA.csv
    with open(f'{path_to_data}/tableA.csv', newline='', encoding="utf-8") as f:
        for record in csv.DictReader(f):
            record['id'] = f'{left_source}-{record["id"]}'
            if dataset in ['amazon-google', 'walmart-amazon']:
                record['name'] = record.pop('title')
            record[source_field] = left_source
            string_representation = entity_serializer.convert_to_str_representation(record)
            record_dict[record['id']] = string_representation
    logger.info(f'Loaded {len(record_dict)} records from tableA')

    # Load records from tableB.csv
    with open(f'{path_to_data}/tableB.csv', newline='', encoding="utf-8") as f:
        for record in csv.DictReader(f):
            record['id'] = f'{right_source}-{record["id"]}'
            record[source_field] = right_source
            string_representation = entity_serializer.convert_to_str_representation(record)
            record_dict[record['id']] = string_representation
    logger.info(f'Loaded {len(record_dict)} records from tableB')

    # Prepare training and validation samples
    train_samples = []
    valid_samples = []
    valid_left = []
    valid_right = []
    valid_labels = []

    # Load training samples from train.csv
    with open(f'{path_to_data}/train.csv', newline='', encoding="utf-8") as f:
        for record in csv.DictReader(f):
            left_record = record_dict[f'{left_source}-{record["ltable_id"]}']
            right_record = record_dict[f'{right_source}-{record["rtable_id"]}']
            score = float(record['label'])
            train_samples.append(InputExample(texts=[left_record, right_record], label=score))
    logger.info(f'Loaded {len(train_samples)} training samples')

    # Load validation samples from valid.csv
    with open(f'{path_to_data}/valid.csv', newline='', encoding="utf-8") as f:
        for record in csv.DictReader(f):
            left_record = record_dict[f'{left_source}-{record["ltable_id"]}']
            right_record = record_dict[f'{right_source}-{record["rtable_id"]}']
            valid_left.append(left_record)
            valid_right.append(right_record)
            valid_labels.append(int(record['label']))
            score = float(record['label'])
            valid_samples.append(InputExample(texts=[left_record, right_record], label=score))

    # Create DataLoader for training
    train_loader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)
    logger.warning('Loss is not defined!')

    # Prepare evaluators
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(valid_samples, name=f'{dataset}-valid')
    binary_acc_evaluator = BinaryClassificationEvaluator(
        sentences1=valid_left,
        sentences2=valid_right,
        labels=valid_labels,
        name="test-dev",
    )

    # Evaluate the model before training
    results = binary_acc_evaluator(model)
    
    # Configure the training
    warmup_steps = math.ceil(len(train_loader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        save_best_model=True,
        output_path=model_save_path,
        show_progress_bar=True
    )

    # Save the trained model
    model.save(model_save_path)

if __name__ == "__main__":
    dataset_names = ['amazon-google', 'walmart-amazon', 'wdcproducts20pair', 'wdcproducts80pair']
    for dataset_name in dataset_names:
        sbert_finetuning(
            dataset=dataset_name,
            model_pretrained_checkpoint='roberta-base',
            epochs=20,
            output_dir=f'./blockers/sbert/saved_blockers/{dataset_name}'
        )