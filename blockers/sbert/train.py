"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""
import csv

import numpy as np

from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
import logging
import os


#### Just some code to print debug information to stdout

#### /print debug information to stdout
# src.strategy.open_book.entity_serialization import EntitySerializer

class EntitySerializer:
    def __init__(self, schema_org_class):
        self.schema_org_class = schema_org_class

        if self.schema_org_class == 'amazon-google':
            self.context_attributes = ['manufacturer', 'name', 'price']
        elif schema_org_class == 'walmart-amazon':
            self.context_attributes = ['name', 'category', 'brand', 'modelno', 'price']
        elif 'wdcproducts' in schema_org_class:
            self.context_attributes = ['brand', 'name', 'price', 'pricecurrency', 'description']
        else:
            raise ValueError('Entity Serialization not defined for schema org class {}'.format(self.schema_org_class))

    def convert_to_str_representation(self, entity, excluded_attributes=None, without_special_tokens=False):
        """Convert to string representation of entity"""
        entity_str = ''
        selected_attributes = self.context_attributes

        if entity is None:
            raise ValueError('Entity must not be None!')

        if excluded_attributes is not None:
            selected_attributes = [attr for attr in self.context_attributes if attr not in excluded_attributes]

        for attr in selected_attributes:
            attribute_value = self.preprocess_attribute_value(entity, attr)
            if attr == 'description' and attribute_value is not None:
                attribute_value = attribute_value[:100]
            if attribute_value is not None:
                if without_special_tokens:
                    entity_str = '{} {}'.format(entity_str, attribute_value)
                else:
                    entity_str = '{}[COL] {} [VAL] {} '.format(entity_str, attr, attribute_value)
            if attribute_value is None:
                if without_special_tokens:
                    entity_str = '{}'.format(entity_str)
                else:
                    entity_str = '{}[COL] {} [VAL] '.format(entity_str, attr)

        return entity_str

    def preprocess_attribute_value(self, entity, attr):
        """Preprocess attribute values"""
        attribute_value = None

        if entity is None:
            raise ValueError('Entity must not be None!')

        if attr in entity and len(str(entity[attr])) > 0 \
                and entity[attr] is not None and entity[attr] is not np.nan:
            if type(entity[attr]) is list and all(type(element) is str for element in entity[attr]):
                attribute_value = ', '.join(entity[attr])
            elif type(entity[attr]) is str:
                attribute_value = entity[attr]
            elif isinstance(entity[attr], np.floating) or type(entity[attr]) is float:
                if not math.isnan(entity[attr]):
                    attribute_value = str(entity[attr])
            else:
                attribute_value = str(entity[attr])

        return attribute_value
    
def sbert_finetuning(dataset, model_pretrained_checkpoint, epochs, output_dir):

    logger = logging.getLogger()
    logger.info('Selected data set {}'.format(dataset))
    # Check if dataset exsits. If not, download and extract it

    # Read the dataset
    train_batch_size = 128
    num_epochs = epochs
    #model_save_path = '{}/models/open_book/finetuned_sbert_{}_{}_{}_dense_{}_extended_subset_pairs'.format(os.environ['DATA_DIR'], model_name.replace('/', ''),
    #                                                                                pooling, loss, dataset)
    model_save_path = output_dir
    logger.warning(model_save_path)

    

    # Complete model name if necessary
    if dataset in model_pretrained_checkpoint:
        model_name = '{}/models/open_book/{}'.format(os.environ['DATA_DIR'], model_pretrained_checkpoint)
        logging.info('Path to model: ' + model_name)

    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    logging.info('Path to model: ' + model_pretrained_checkpoint)
    word_embedding_model = models.Transformer('roberta-base')
    word_embedding_model.max_seq_length = 128

    if dataset not in model_pretrained_checkpoint:
        # Add special tokens
        special_tokens_dict = {'additional_special_tokens': ['[COL]', '[VAL]']}
        num_added_toks = word_embedding_model.tokenizer.add_special_tokens(special_tokens_dict)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        logging.info('Added special tokens [COL], [VAL]')

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                    pooling_mode_mean_tokens=True,
                                    pooling_mode_cls_token=False,
                                    pooling_mode_max_tokens=False)

    # Dense Layer on top
    #dense_model = models.Dense(word_embedding_model.get_word_embedding_dimension(), out_features=128)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Convert the dataset to a DataLoader ready for training
    logging.info("Read {} dataset".format(dataset))

    record_dict = {}
    source_field = '__source'
    left_source = 'table_a'
    right_source = 'table_b'

    path_to_data = f'./data/raw/{dataset}'
    entity_serializer = EntitySerializer(schema_org_class=dataset)
    # Load table A and table B
    with open(f'{path_to_data}/tableA.csv', newline='', encoding="utf-8") as f:
        print('asd')
        for record in csv.DictReader(f):
            record['id'] = f'{left_source}-{record["id"]}'
            if dataset in ['amazon-google', 'walmart-amazon']:
                record['name'] = record.pop('title')
            #record['title'] = record.pop('name')
            record[source_field] = left_source
            string_representation = entity_serializer.convert_to_str_representation(record)
            # del record['description']  # drop description, for benchmarking
            record_dict[record['id']] = string_representation
    print(f'Loaded {len(record_dict)} records')

    with open(f'{path_to_data}/tableB.csv', newline='', encoding="utf-8") as f:
        for record in csv.DictReader(f):
                record['id'] = f'{right_source}-{record["id"]}'
                #record['title'] = record.pop('name')
                record[source_field] = right_source
                string_representation = entity_serializer.convert_to_str_representation(record)
                #del record['description']  # drop description, for benchmarking
                record_dict[record['id']] = string_representation
    print(f'Loaded {len(record_dict)} records')


    train_samples = []
    valid_samples = []
    # test_samples = []
    valid_left = []
    valid_right = []
    valid_labels = []

    # Use both train+valid for training
    with open(f'{path_to_data}/train.csv', newline='', encoding="utf-8") as f:
        for record in csv.DictReader(f):
            left_record = record_dict['{}-{}'.format(left_source, record['ltable_id'])]
            right_record = record_dict['{}-{}'.format(right_source, record['rtable_id'])]

            score = float(record['label'])
            inp_example = InputExample(texts=[left_record, right_record], label=score)
            train_samples.append(inp_example)

    with open(f'{path_to_data}/valid.csv', newline='', encoding="utf-8") as f:
        for record in csv.DictReader(f):
            left_record = record_dict['{}-{}'.format(left_source, record['ltable_id'])]
            right_record = record_dict['{}-{}'.format(right_source, record['rtable_id'])]
            valid_left.append(left_record)
            valid_right.append(right_record)
            valid_labels.append(int(record['label']))

            score = float(record['label'])
            inp_example = InputExample(texts=[left_record, right_record], label=score)
            valid_samples.append(inp_example)

    print(f'Loaded {len(train_samples)} train samples')

    train_loader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)
    logger.warning('Loss is not defined!')
    logging.info("Read {} Training dev dataset".format(dataset))
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(valid_samples, name='{}-valid'.format(dataset))

    binary_acc_evaluator = BinaryClassificationEvaluator(
      sentences1=valid_left,
      sentences2=valid_right,
      labels=valid_labels,
      name="test-dev",
    )

    results = binary_acc_evaluator(model)
    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_loader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_loader, train_loss)],
              evaluator=evaluator,
              epochs=20,
              save_best_model=True,
              output_path=model_save_path,
              show_progress_bar=True)

    model.save(model_save_path)
    # print(binary_acc_evaluator(model))

    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################

    #model = SentenceTransformer(model_save_path)
    print(binary_acc_evaluator(model))
    # test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
    # test_evaluator(model, output_path=model_save_path)


if __name__ == "__main__":
    dataset_names = ['amazon-google', 'walmart-amazon', 'wdcproducts20pair', 'wdcproducts80pair']
    for dataset_name in dataset_names:
      sbert_finetuning(dataset=dataset_name, 
      model_pretrained_checkpoint='roberta-base', 
      epochs=20, 
      output_dir=f'./blockers/sbert/saved_blockers/{dataset_name}')