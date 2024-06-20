import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import pandas as pd
import sys
sys.path.append('.')
from matchers.ditto.model import DittoModel
from matchers.ditto.dataset import DittoDataset
from matchers.matcher import Matcher
import torch
import torch.nn as nn
import os
import numpy as np
import random

from torch.utils import data
from tqdm import tqdm
from scipy.special import softmax



def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_str(ent1, ent2, summarizer=None, max_len=256, dk_injector=None):
    """Serialize a pair of data entries

    Args:
        ent1 (Dictionary): the 1st data entry
        ent2 (Dictionary): the 2nd data entry
        summarizer (Summarizer, optional): the summarization module
        max_len (int, optional): the max sequence length
        dk_injector (DKInjector, optional): the domain-knowledge injector

    Returns:
        string: the serialized version
    """
    content = ''
    for ent in [ent1, ent2]:
        if isinstance(ent, str):
            content += ent
        else:
            for attr in ent.keys():
                content += 'COL %s VAL %s ' % (attr, ent[attr])
        content += '\t'

    content += '0'
    new_ent1, new_ent2, _ = content.split('\t')

    return new_ent1 + '\t' + new_ent2 + '\t0'


def classify(sentence_pairs, device, model,
             lm='roberta-base',
             max_len=256,
             threshold=None):
    """Apply the MRPC model.

    Args:
        sentence_pairs (list of str): the sequence pairs
        model (MultiTaskNet): the model in pytorch
        max_len (int, optional): the max sequence length
        threshold (float, optional): the threshold of the 0's class

    Returns:
        list of float: the scores of the pairs
    """
    inputs = sentence_pairs
    dataset = DittoDataset(data=inputs,
                           max_length=max_len)
    # print(dataset[0])
    iterator = data.DataLoader(dataset=dataset,
                               batch_size=32,
                               shuffle=False,
                               num_workers=0,
                               collate_fn=DittoDataset.pad)

    # prediction
    all_probs = []
    all_logits = []
    with torch.no_grad():
        # print('Classification')
        for i, batch in enumerate(iterator):
            x, _ = batch
            x = x.to(device)
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_logits += logits.cpu().numpy().tolist()

    if threshold is None:
        threshold = 0.5

    pred = [1 if p > threshold else 0 for p in all_probs]
    return pred, all_logits


class DittoMatcher(Matcher):
    def __init__(self, schema_org_class, model_name, base_model, alpha, context_attributes=None, matcher=False, max_len=256):
        super().__init__(schema_org_class, 'Ditto Cross Encoder - {} {}'.format(base_model, max_len), context_attributes, matcher)

        self.model_name = model_name
        # Initialize tokenizer and model for ditto model
        if model_name is not None:
            model_path = model_name

            #self.device = torch.device('cpu') # Keep Ditto model on cpu for now.
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, additional_special_tokens=('[COL]', '[VAL]', '[SEP]', '[CLS]'))
            self.model = DittoModel(len_tokenizer=len(self.tokenizer), alpha=alpha)
            self.base_model = base_model
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)), strict=False)
            self.model = self.model.to(self.device)

            # threshold_path = '{}/ditto/{}/threshold.txt'.format(os.environ['DATA_DIR'], model_name)
            # with open(threshold_path) as f:
            #     self.threshold = float(f.readline().replace('threshold:', ''))

            self.max_length = max_len

    def predict_matches(self, entities1, entities2, excluded_attributes1=None, excluded_attributes2=None):

        records1_serial = [self.entity_serializer.convert_to_str_representation(entity1, excluded_attributes1)
                            for entity1 in entities1]
        records2_serial = [self.entity_serializer.convert_to_str_representation(entity2, excluded_attributes2)
                            for entity2 in entities2]
        df = pd.DataFrame({'features_left': records1_serial,
                          'features_right': records1_serial})

       
        set_seed(123)
        pred, _ = classify(df, self.device, self.model, lm=self.base_model, max_len=self.max_length) # threshold=self.threshold --> exclude Threshold for testing\
        set_seed(42)
        # ds_path = '{}/ditto/{}/prediction.csv'.format(os.environ['DATA_DIR'], self.model_name)
        # with open(ds_path, 'w') as file:
        #     for pair, prediction in zip(pairs, pred):
        #         file.write('{}\t{}\n'.format(pair, prediction))

        return pred

    def re_rank_evidences(self, query_table, evidences):
        """Re-rank evidences based on confidence of a cross encoder"""
        left_entities = []
        right_entities = []
        for row in query_table.table:
            rel_evidences = [evidence for evidence in evidences if evidence.entity_id == row['entityId']]
            if len(rel_evidences) > 0:
                    left_entities.extend([row] * len(rel_evidences))
                    right_entities.extend([rel_evidence.context for rel_evidence in rel_evidences])

        preds = self.predict_matches(entities1=left_entities, entities2=right_entities)

        i = 0
        # qt_path = '{}/ditto/{}/querytable.csv'.format(os.environ['DATA_DIR'], self.model_name)
        # with open(qt_path, 'w') as file:
        for row in query_table.table:
            rel_evidences = [evidence for evidence in evidences if evidence.entity_id == row['entityId']]
            if len(rel_evidences) > 0:
                for evidence in rel_evidences:
                    evidence.scores[self.name] = preds[i]
                    evidence.similarity_score = preds[i]
                    #file.write('{}\t{}\n'.format(self.entity_serializer.convert_to_str_representation(row), self.entity_serializer.convert_to_str_representation(evidence.context)))
                    i += 1

        updated_evidences = sorted(evidences, key=lambda evidence: evidence.similarity_score, reverse=True)
        if self.matcher:
            print('Number of evidences before matching: {}'.format(len(updated_evidences)))
            updated_evidences = [evidence for evidence in updated_evidences if evidence.similarity_score > 0.5]
            print('Number of evidences after matching: {}'.format(len(updated_evidences)))

        return updated_evidences