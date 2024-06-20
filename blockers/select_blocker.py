
from blockers.barlowtwins_encoder import BarlowTwinsEncoder
from blockers.sbert_encoder import SBERTEncoder
from blockers.scblock_encoder import SCBlockEncoder

def select_blocker(model_name, model_path, dataset_name):

    if model_name== 'barlowtwins':
        model = BarlowTwinsEncoder(model_path, dataset_name)
    elif model_name == 'sbert':
        model = SBERTEncoder(model_path, dataset_name)
    elif model_name== 'scblock':
        model = SCBlockEncoder(model_path, dataset_name)
    else:
        raise Exception("No blocker was given.") 

    return model