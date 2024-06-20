import os

import torch

import pandas as pd
import deepmatcher as dm
import sys
sys.path.append('.')
from matchers.matcher import Matcher


def determine_path_to_model(model_name, schema_org_class, context_attributes):
    context_attribute_string = '_'.join(context_attributes)
    path_to_model = '{}/models/open_book/finetuned_cross_encoder-{}-{}-{}'.format(os.environ['DATA_DIR'],
                                                                                  model_name, schema_org_class,
                                                                                  context_attribute_string)
    return path_to_model


class DeepMatcher(Matcher):

    def __init__(self, schema_org_class, model_path, base_model, context_attributes=None, matcher=False, max_length=128):
        super().__init__(schema_org_class, 'DeepMatcher {}'.format(model_path.split('/')[-2] + '/' + model_path.split('/')[-1]), context_attributes, matcher)

        self.context_attributes = context_attributes
        # Initialize tokenizer and model for BERT if necessary
        if model_path is not None:
            self.model = dm.MatchingModel(attr_comparator='abs-diff', 
                        attr_summarizer=dm.attr_summarizers.Hybrid(word_aggregator='max-pool'),
                        classifier='2-layer-highwaynet-softmax')
            
            # Load model
            self.model.load_state(model_path)

    def re_rank_evidences(self, query_table, evidences):
        """Re-rank evidences based on confidence of a cross encoder"""
        for row in query_table.table:
            rel_evidences = [evidence for evidence in evidences if evidence.entity_id == row['entityId']]

            if len(rel_evidences) > 0:

                # Create smaller batches of entities
                def batches(lst, chunk_size):
                    for i in range(0, len(lst), chunk_size):
                        yield lst[i:i + chunk_size]

                for evidence_chunk in batches(rel_evidences, min(8, len(rel_evidences))):
                    # Get unlabeled entities
                    left_entities = [row] * len(evidence_chunk)
                    left_entities = [{'left_' + str(key): val  for key, val in entity.items() if key in self.context_attributes} for entity in left_entities]
                    for entity in left_entities:
                        print(entity)
                        entity['left_title'] = entity.pop('left_name')
                    
                    right_entities = [rel_evidence.context for rel_evidence in evidence_chunk]
                    print(right_entities)
                    right_entities = [{'right_' + str(key): val for key, val in entity.items() if key in self.context_attributes} for entity in right_entities]
                    for entity in right_entities:
                        entity['right_title'] = entity.pop('right_name')

                    entities = [{**left_entity, **right_entity}for left_entity, right_entity in zip(left_entities, right_entities)]
                    entities = {k: [d[k] for d in entities] for k in entities[0]}
                    pd.DataFrame(entities).to_csv('temp_unlabeled.csv', index=False)
                    print(pd.DataFrame(entities))
                    print(entities)

                    unlabeled = dm.data.process_unlabeled(path='temp_unlabeled.csv', trained_model=self.model)
                    preds = self.model.run_prediction(unlabeled) 
                    print(preds)
                    
                    # logits = self.predict_matches(entities1=left_entities, entities2=right_entities)
                    # preds = [0 if pred.item() < 0.5 else 1 for pred in logits]

                    # for evidence, pred in zip(evidence_chunk, preds):
                    #     # Overwrite existing scores
                    #     #evidence.scores = {self.name: pred[1].item()}
                    #     evidence.scores[self.name] = pred
                    #     evidence.similarity_score = pred

        updated_evidences = sorted(evidences, key=lambda evidence: evidence.similarity_score, reverse=True)
        if self.matcher:
            updated_evidences = [evidence for evidence in updated_evidences if evidence.similarity_score > 0.5]

        return updated_evidences