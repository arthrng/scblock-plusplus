"""
This code is based on the code written by:
Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines.
In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer.

The code can be found at:
https://github.com/wbsg-uni-mannheim/SC-Block
"""

import gzip
import json
import logging
import os
import pandas as pd

def convert_table_to_table_format(dataset, table_name):
    """
    Convert Test set Table A of the DeepMatcher benchmark to a query table format.

    :param dataset: String representing the dataset name (e.g., 'amazon-google').
    :param table_name: String representing the name of the table to convert (e.g., 'tableB').
    """
    # Determine the parent directory and construct paths to the input and output files
    parent_directory = os.path.dirname(os.getcwd()).replace('\\', '/')
    path_to_table_b = f'{parent_directory}/src/data/raw/{dataset}/tableB.csv'
    path_to_corpus_table = f'{parent_directory}/src/data/corpus/{dataset}'

    # Read the CSV file and fill missing values with empty strings
    df_table_b = pd.read_csv(path_to_table_b).fillna('')

    # Initialize a list to hold formatted lines
    formatted_lines = []

    # Iterate over each row in the DataFrame
    for index, row in df_table_b.iterrows():
        # Convert row to a dictionary and normalize the keys to lowercase
        formatted_line = row.to_dict()
        formatted_line = {k.lower(): v for k, v in formatted_line.items()}

        # Rename 'id' to 'row_id' and handle specific dataset requirements
        formatted_line['row_id'] = formatted_line.pop('id')
        formatted_line['name'] = formatted_line.pop('title')

        # Add 'page_url' field
        formatted_line['page_url'] = f'{dataset}_{table_name}'

        # Append formatted line to the list
        formatted_lines.append(formatted_line)

    # Ensure the output directory exists
    if not os.path.exists(path_to_corpus_table):
        os.makedirs(path_to_corpus_table)

    # Define the path for the output file
    path_to_corpus_table_file = f'{path_to_corpus_table}/{dataset}_{table_name}.json.gz'

    # Write formatted lines to a compressed JSON file
    with gzip.open(path_to_corpus_table_file, 'wb') as output_file:
        for formatted_line in formatted_lines:
            new_line = json.dumps(formatted_line) + '\n'
            output_file.write(new_line.encode())

    logging.info(f'Converted {table_name} of {dataset} to table format')

def save_additional_records(collected_additional_records, path_to_corpus_table, dataset, additional_record_files):
    """
    Save additional records to a compressed JSON file.

    :param collected_additional_records: List of records to save.
    :param path_to_corpus_table: Path to the directory where the file will be saved.
    :param dataset: Name of the dataset.
    :param additional_record_files: Identifier for the additional records files.
    """
    path_to_corpus_table_file = f'{path_to_corpus_table}/{dataset}_additional_records_{additional_record_files}.json.gz'

    # Write additional records to a compressed JSON file
    with gzip.open(path_to_corpus_table_file, 'wb') as output_file:
        for collected_additional_record in collected_additional_records:
            new_line = json.dumps(collected_additional_record) + '\n'
            output_file.write(new_line.encode())

if __name__ == '__main__':
    # Configure logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Define the list of dataset names to process
    dataset_names = ['walmart-amazon', 'wdcproducts20pair', 'wdcproducts80pair']

    # Convert each dataset's table to the desired format
    for dataset_name in dataset_names:
        convert_table_to_table_format(dataset_name, 'tableB')
