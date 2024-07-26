"""
This code is based on the code written by:
Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines.
In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer.

The code can be found at:
https://github.com/wbsg-uni-mannheim/SC-Block
"""

import re
from value_normalizer import get_datatype, normalize_value, detect_not_none_value

def extract_entity(raw_entity, dataset):
    """Extract and normalize entity from raw JSON data.

    Args:
        raw_entity (dict): The raw JSON data of the entity.
        dataset (str): The dataset name which helps in determining the relevant keys.

    Returns:
        dict: A dictionary containing the normalized entity.
    """
    entity = {}

    for raw_key, raw_value in raw_entity.items():
        key = normalize_key(raw_key, dataset)

        if check_key_is_relevant(key, dataset):
            if isinstance(raw_value, str) and len(raw_value) > 0 and detect_not_none_value(raw_value):
                normalized_value = normalize_value(raw_value, get_datatype(key), raw_entity, entity)
                if len(normalized_value) > 0 and detect_not_none_value(normalized_value):
                    entity[key] = normalized_value
            elif isinstance(raw_value, dict):
                # Case 1: Property has 'name' sub-property
                if 'name' in raw_value and len(raw_value['name']) > 0 and detect_not_none_value(raw_value['name']):
                    normalized_value = normalize_value(raw_value['name'], get_datatype(key), raw_entity, entity)
                    if len(normalized_value) > 0 and detect_not_none_value(normalized_value):
                        entity[key] = normalized_value
                # Case 2: Lift all values by sub-property name
                else:
                    for raw_property_key, property_value in raw_value.items():
                        property_key = normalize_key(raw_property_key, dataset)
                        if check_key_is_relevant(property_key, dataset):
                            if len(property_value) > 0 and detect_not_none_value(property_value):
                                normalized_value = normalize_value(property_value, get_datatype(property_key), raw_entity, entity)
                                if len(normalized_value) > 0 and detect_not_none_value(normalized_value) and property_key not in entity:
                                    entity[property_key] = normalized_value
            elif isinstance(raw_value, list):
                # Case 1: List of strings
                if all(isinstance(element, str) for element in raw_value):
                    normalized_value = ', '.join([
                        normalize_value(element, get_datatype(key), raw_entity, entity)
                        for element in raw_value if len(element) > 0
                    ])
                    if len(normalized_value) > 0 and detect_not_none_value(normalized_value) and key not in entity:
                        entity[key] = normalized_value
                # Case 2: List of dicts with 'name' attribute
                elif all(isinstance(element, dict) for element in raw_value) and all('name' in element for element in raw_value):
                    normalized_value = ', '.join([
                        normalize_value(element['name'], get_datatype(key), raw_entity, entity)
                        for element in raw_value if len(element['name']) > 0 and isinstance(element['name'], str)
                    ])
                    if len(normalized_value) > 0 and detect_not_none_value(normalized_value) and key not in entity:
                        entity[key] = normalized_value
            else:
                entity[key] = str(raw_value)

    return entity

def normalize_key(key_value):
    """Normalize key value based on dataset schema.

    Args:
        key_value (str): The raw key value from JSON.

    Returns:
        str: The normalized key value.
    """
    # Replace unwanted characters
    key_value = re.sub(r'[\\u201d%20\\u201c]', '', key_value)
    key_value = re.sub(r"[^0-9a-zA-Z]+", '', key_value)

    return key_value

def check_key_is_relevant(key, dataset):
    """Check if a key is relevant for the given dataset schema.

    Args:
        key (str): The key to check.
        dataset (str): The dataset name.

    Returns:
        bool: True if the key is relevant, False otherwise.
    """
    if dataset == 'walmart-amazon':
        attributes = ['name', 'category', 'brand', 'modelno', 'price']
    elif 'wdcproducts' in dataset:
        attributes = ['brand', 'name', 'price', 'pricecurrency', 'description']
    else:
        raise ValueError(f'Dataset {dataset} is not known!')

    return key in attributes
