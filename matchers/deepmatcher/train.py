import deepmatcher as dm
import os
import pandas as pd

def convert_json_to_csv(data_directory, file_name):
  df = pd.read_json(f'{data_directory}/{file_name}.json.gz',  lines=True)
  df.to_csv(f'{data_directory}/{file_name}-deepmatcher.csv', index=False)

train_model = True

# Load directory
dataset_name = 'wdcproducts20pair'
data_directory = f'./data/processed'

# Convert JSON to CSV
convert_json_to_csv(data_directory, f'{dataset_name}-train-pairs')
convert_json_to_csv(data_directory, f'{dataset_name}-val-pairs')
convert_json_to_csv(data_directory, f'{dataset_name}-test-pairs')

 # Load data
train, validation, test = dm.data.process(
    path=data_directory,
    train=f'{dataset_name}-train-pairs-deepmatcher.csv',
    validation=f'{dataset_name}-val-pairs-deepmatcher.csv',
    test=f'{dataset_name}-test-pairs-deepmatcher.csv',
    ignore_columns=('left_id', 'right_id', 'left_cluster_id', 'right_cluster_id'),
    left_prefix='left_',
    right_prefix='right_',
    label_attr='label')


 # Create model
model = dm.MatchingModel(attr_comparator='abs-diff', 
                        attr_summarizer=dm.attr_summarizers.Hybrid(word_aggregator='max-pool'),
                        classifier='2-layer-highwaynet-softmax') #inv-freq-avg-pool

if train_model:
  # Train model
  model.run_train(
      train,
      validation,
      epochs=20,
      batch_size=8,
      best_save_path=f'./matchers/deepmatcher/saved_matchers/{dataset_name}/deepmatcher.pth')
else:
  # Load model
  model.load_state(f'./matchers/deepmatcher/saved_matchers/{dataset_name}/deepmatcher.pth')

# Evaluate on test set
model.run_eval(test)