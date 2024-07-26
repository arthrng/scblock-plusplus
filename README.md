# SC-Block++ #

This repository includes the code for the paper _SC-Block++: A Blocking Algorithm Based on Adaptive Flood Regularization_. **SC-Block++** is an extension of the state-of-the-art blocking method **SC-Block** proposed in the paper _Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines. In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer_. SC-Block++ incorporates **Adaptive Flood Regularization** to reduce the probability that SC-Block overfits. 

All of the code is written in PYTHON3 (https://www.python.org/) and makes use of the PyTorch framework (https://www.pytorch.org/).

## Running the code ##
In order to evaluate SC-Block++ against the other blocking methods, one must perform the following steps:
1. Prepare the data by running the `preprocess_data.py` script.
2. Run the `train.py` scripts of the blocking and matching algorithms.
3. Run `convert_to_query_table.py` to construct the query table for the datasets.
4. Run `convert_to_index_table.py` and `faiss_indexing.py` to construct the index table for the datasets.
5. Load the data into Elasticsearch by running `load_data_into_es.py`.
6. Run `run_pipeline.py` to execute the entity matching pipelines and to obtain the results.

## Explanation of the directories ##
* blockers - contains the scripts used to train the language models for the blockers.
* matchers - contains the scripts used to train the language models for the matchers.
* evaluation - contains the scripts used for the evaluation of the pipelines.
* query_and_index - contains the scripts used to construct the query and index tables.
* retrieval - contains the scripts used to construct the candidate/matching pairs.

## Credits ##
Some of the code we utilize is originally written by Brinkmann. et al (2024) and can be found at https://github.com/wbsg-uni-mannheim/SC-Block.
