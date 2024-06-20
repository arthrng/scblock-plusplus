import logging

import sys
sys.path.append('.')
from matchers.ditto_matcher import DittoMatcher
from matchers.supcon import SupConMatcher
from matchers.deepmatcher import DeepMatcher
from matchers.matcher import Matcher


def select_matcher(matching_strategy, schema_org_class, context_attributes=None):
    """Return a re-ranker based on the defined re-ranking-strategy
        :return SimilarityReRanker
    """
    logger = logging.getLogger()

    if matching_strategy is None:
        matching_strategy_name = None
    else:
        matching_strategy_name = matching_strategy['name']

    logger.info('Select Similarity Re-ranking Strategy {}!'.format(matching_strategy_name))

    if matching_strategy_name == 'supconmatch':
        matching_strategy = SupConMatcher(schema_org_class, matching_strategy['model_name'], matching_strategy['base_model'], context_attributes, matching_strategy['matcher'])
    elif matching_strategy_name == 'ditto':
        matching_strategy = DittoMatcher(schema_org_class, matching_strategy['model_name'], matching_strategy['base_model'], context_attributes, matching_strategy['matcher'])
    elif matching_strategy_name == 'deepmatcher':
         matching_strategy = DeepMatcher(schema_org_class, matching_strategy['model_name'], matching_strategy['base_model'], context_attributes, matching_strategy['matcher'])
    else:
        # Fall back to default reranker, which does no re-ranking
        matching_strategy = Matcher(schema_org_class, 'Dummy Re-Ranker')

    return matching_strategy