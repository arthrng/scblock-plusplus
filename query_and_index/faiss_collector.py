"""
This code is based on the code written by:
Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines.
In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer.

The code can be found at:
https://github.com/wbsg-uni-mannheim/SC-Block
"""

import logging
import os
import faiss
import numpy as np

def determine_path_to_faiss_index(schema_org_class, model_name, pool, sim, clusters, switched=False):
    """
    Determine the path to save/load the FAISS index based on the provided parameters.
    
    :param schema_org_class: The schema organization class name.
    :param model_name: The name of the model used.
    :param pool: The pooling method used.
    :param sim: The similarity measure used.
    :param clusters: Whether clustering is used or not.
    :param switched: Whether the indices are switched or not.
    :return: The path to the FAISS index file.
    """
    # Define the directory to save the FAISS indices
    path_to_faiss_dir = './data/faiss'

    # Create the directory if it does not exist
    if not os.path.isdir(path_to_faiss_dir):
        os.mkdir(path_to_faiss_dir)

    # Determine the file name based on the parameters
    return '{}/{}/{}_faiss_{}_{}_{}.index'.format(path_to_faiss_dir, schema_org_class, schema_org_class, model_name.split('/')[-1], pool, sim)


class FaissIndexCollector:
    def __init__(self, schema_org_class, model_name, pooling, similarity_measure, final_representation,
                 dimensions, clusters, switched=False):
        """
        Initialize the FaissIndexCollector with the given parameters.

        :param schema_org_class: The schema organization class name.
        :param model_name: The name of the model used.
        :param pooling: The pooling method used.
        :param similarity_measure: The similarity measure used.
        :param final_representation: The final representation to be used.
        :param dimensions: The dimensions of the vectors.
        :param clusters: Whether clustering is used or not.
        :param switched: Whether the indices are switched or not.
        """
        self.schema_org_class = schema_org_class
        self.model_name = model_name
        self.pooling = pooling
        self.similarity_measure = similarity_measure
        self.dimensions = dimensions
        self.clusters = clusters
        self.switched = switched
        self.final_representation = final_representation

        # Initialize indices and entity representations
        self.indices = {}
        self.entity_representations = {}
        self.initialize_entity_representations()
        self.unsaved_representations = []
        self.next_representation = 0
        self.collected_entities = 0

        # Create FAISS index based on similarity measure
        index_type = faiss.IndexFlatL2 if self.similarity_measure == 'f2' else faiss.IndexFlatIP
        self.indices['index_{}_{}'.format(self.pooling, self.similarity_measure)] = index_type(self.dimensions)

    def initialize_entity_representations(self):
        """
        Initialize the entity representations dictionary.
        """
        self.entity_representations = {'{}_{}'.format(self.pooling, self.similarity_measure): []}

    def collect_entity_representation(self, entity):
        """
        Collect the representation of an entity and add it to the entity representations.

        :param entity: The entity from which to collect the representation.
        """
        identifier = 'entity_vector_{}'.format(self.pooling)
        if self.similarity_measure == 'cos':
            identifier += '_norm'
        
        if identifier not in entity:
            logging.getLogger().warning('Identifier: {} is not defined!'.format(identifier))
            return
        
        entity_rep = entity[identifier]
        self.entity_representations['{}_{}'.format(self.pooling, self.similarity_measure)].append(entity_rep)

    def add_entity_representations_to_indices(self):
        """
        Add collected entity representations to the FAISS index and persist the index.
        """
        index_identifier = 'index_{}_{}'.format(self.pooling, self.similarity_measure)
        representations = np.array(
            self.entity_representations['{}_{}'.format(self.pooling, self.similarity_measure)]
        ).astype('float32')
        self.indices[index_identifier].add(representations)

    def save_indices(self):
        """
        Save the FAISS index to the filesystem.
        """
        index_identifier = 'index_{}_{}'.format(self.pooling, self.similarity_measure)
        path_to_faiss_index = determine_path_to_faiss_index(
            self.schema_org_class, self.model_name, self.pooling, self.similarity_measure, self.clusters, switched=self.switched
        )
        faiss.write_index(self.indices[index_identifier], path_to_faiss_index)
        logging.info(
            'Saved Index - {} for model {} and schema org class {}'.format(
                index_identifier, self.model_name, self.schema_org_class
            )
        )

    def next_savable_entities(self):
        """
        Retrieve the next set of savable entities from unsaved representations.

        :return: The next set of savable entities.
        """
        entities = None
        removable_result = None
        for result in self.unsaved_representations:
            if result[0] == self.next_representation:
                entities = result[1]
                self.next_representation += 1
                removable_result = result
                break
        
        if removable_result is not None:
            self.unsaved_representations.remove(removable_result)
        
        return entities
