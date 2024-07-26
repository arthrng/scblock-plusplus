import logging
from blockers.barlowtwins_encoder import BarlowTwinsEncoder
from blockers.sbert_encoder import SBERTEncoder
from blockers.scblock_encoder import SCBlockEncoder

def select_blocker(model_name, model_path, dataset_name):
    """
    Select and initialize the appropriate encoder based on the model name.

    Args:
        model_name (str): The name of the model ('barlowtwins', 'sbert', or 'scblock').
        model_path (str): Path to the pre-trained model.
        dataset_name (str): The name of the dataset.

    Returns:
        Encoder: An instance of the selected encoder class.

    Raises:
        ValueError: If an invalid model name is provided.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize the selected model
    if model_name == 'barlowtwins':
        logger.info(f"Initializing BarlowTwinsEncoder with model path: {model_path}")
        model = BarlowTwinsEncoder(model_path, dataset_name)
    elif model_name == 'sbert':
        logger.info(f"Initializing SBERTEncoder with model path: {model_path}")
        model = SBERTEncoder(model_path, dataset_name)
    elif model_name == 'scblock':
        logger.info(f"Initializing SCBlockEncoder with model path: {model_path}")
        model = SCBlockEncoder(model_path, dataset_name)
    else:
        raise ValueError(f"Invalid model name '{model_name}' provided. Valid options are 'barlowtwins', 'sbert', 'scblock'.")

    return model