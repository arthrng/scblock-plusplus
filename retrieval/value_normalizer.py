"""
This code is based on the code written by:
Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines.
In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer.

The code can be found at:
https://github.com/wbsg-uni-mannheim/SC-Block
"""

import logging
import re
import phonenumbers
from dateutil.parser import parse
from datetime import datetime, timedelta
import country_converter as coco
from phonenumbers import NumberParseException

def get_datatype(attribute):
    """
    Determine the data type of the provided attribute based on a predefined mapping.

    Parameters:
    - attribute: The attribute name whose data type is to be determined.

    Returns:
    - str: The data type of the attribute. Possible values are 'date', 'duration', 'coordinate', 'telephone', 'country', or 'string'.
    """
    attr_2_datatype = {
        'datepublished': 'date',
        'duration': 'duration',
        'latitude': 'coordinate',
        'longitude': 'coordinate',
        'telephone': 'telephone',
        'addresscountry': 'country'
    }

    # Return the datatype corresponding to the attribute if found; otherwise, return 'string'
    return attr_2_datatype.get(attribute, 'string')


def detect_not_none_value(value):
    """
    Check if the provided value is not considered a 'none' value based on common placeholders.

    Parameters:
    - value: The value to be checked.

    Returns:
    - bool: True if the value is not a placeholder for 'none', otherwise False.
    """
    # Convert the value to lowercase string and check against common 'none' placeholders
    return str(value).lower() not in ['none', '-', '--', ' ', 'tbd', 'tba', 'n/a', 'na', '?', 'null', '#', '.', ',']


def normalize_value(value, datatype, raw_entity=None, entity=None):
    """
    Normalize the provided value based on its data type. This function handles various types including strings, dates, phone numbers, durations, coordinates, and country codes.

    Parameters:
    - value: The value to be normalized.
    - datatype: The data type of the value which dictates how it should be normalized.
    - raw_entity: Optional raw entity containing additional context for normalization (e.g., for phone number parsing).
    - entity: Optional entity containing additional context for normalization (e.g., for phone number parsing).

    Returns:
    - str: The normalized value as a string.
    """
    logger = logging.getLogger()
    final_value = value
    
    if datatype == 'string' or not isinstance(value, str):
        # For 'string' datatype or if value is not a string, no manipulation is needed
        logger.debug('No manipulation of data type string.')
    elif datatype == 'date':
        # Normalize date values
        try:
            d = parse(value, yearfirst=True, default=datetime(1900, 1, 1))
            final_value = d.strftime("%Y-%m-%d")
        except Exception as e:
            logger.debug(f'Error parsing date: {e}')
    
    elif datatype == 'telephone':
        # Normalize telephone numbers
        country = None
        if entity and 'addresscountry' in entity and isinstance(entity['addresscountry'], str):
            country = entity['addresscountry']
        elif raw_entity and 'address' in raw_entity and raw_entity['address'] and 'addresscountry' in raw_entity['address'] and isinstance(raw_entity['address']['addresscountry'], str):
            country = normalize_value(raw_entity['address']['addresscountry'], 'country')
        
        try:
            phone = phonenumbers.parse(value, country)
            final_value = phonenumbers.format_number(phone, phonenumbers.PhoneNumberFormat.E164)
        except NumberParseException as e:
            # If parsing fails, remove non-numeric characters and retry
            value = re.sub('[^0-9]', '', value)
            try:
                phone = phonenumbers.parse(value, None)
                final_value = phonenumbers.format_number(phone, phonenumbers.PhoneNumberFormat.E164)
            except NumberParseException as e:
                logger.debug(f'Error parsing phone number: {e}')
                # If still invalid, handle special cases where all characters might be '0'
                if len(value) > 0 and value[0] == '0' and value == len(value) * value[0]:
                    final_value = ''
                else:
                    final_value = value
    
    elif datatype == 'duration':
        # Normalize duration values
        try:
            d = parse_timedelta(value)
            time_dict = {
                'H': int(d.seconds / 3600),
                'M': int((d.seconds % 3600) / 60),
                'S': int((d.seconds % 3600) % 60)
            }
            strftduration = 'PT'
            for key, value in time_dict.items():
                if value > 0:
                    strftduration = f'{strftduration}{value}{key}'
            final_value = strftduration
        except Exception as e:
            logger.debug(f'Error parsing duration: {e}')
    
    elif datatype == 'coordinate':
        # Normalize coordinate values
        value = value.strip().replace('\"', '').replace('\\', '')
        try:
            final_value = parse_coordinate(value)
        except ValueError as e:
            logger.debug(f'Error parsing coordinate: {e}')
    
    elif datatype == 'country':
        # Normalize country codes
        try:
            coco_logger = coco.logging.getLogger()
            if coco_logger.level != logging.CRITICAL:
                coco_logger.setLevel(logging.CRITICAL)
            final_value = coco.convert(names=[value], to='ISO2', not_found=None)
        except Exception as e:
            logger.debug(f'Error converting country code: {e}')
            final_value = value
    
    else:
        raise ValueError(f'Normalization of datatype {datatype} is not implemented!')

    return str(final_value)


def parse_timedelta(value):
    """
    Parse a duration string into a timedelta object.

    Parameters:
    - value: The duration string to be parsed.

    Returns:
    - timedelta: A timedelta object representing the duration.
    
    Raises:
    - ValueError: If the string format is unknown or cannot be parsed.
    """
    value = str(value).replace(' ', '')
    regex_patterns = [
        r'P?T?((?P<hours>\d+?)(hr|h|H))?((?P<minutes>\d+?)(m|M|min|Min|phút|мин|分钟|perc|dakika))?((?P<seconds>\d+?)(s|S))?',
        r'P?T?((H)(?P<hours>\d+))?((M)(?P<minutes>\d+))?',
        r'(?P<hours>\d+):(?P<minutes>\d+)',
        r'^(?P<minutes>\d+)$'
    ]

    parts = None
    for pattern in regex_patterns:
        regex = re.compile(pattern)
        parts = regex.match(value)
        if parts:
            break

    if not parts or not parts.lastindex:
        raise ValueError(f'Unknown string format: {value}')

    parts = parts.groupdict()
    time_params = {name: int(param) for name, param in parts.items() if param}

    return timedelta(**time_params)


def parse_coordinate(original_value):
    """
    Parse a coordinate string into a float value. Supports various formats and scientific notation.

    Parameters:
    - original_value: The coordinate string to be parsed.

    Returns:
    - float: The normalized coordinate value rounded to 6 decimal places.

    Raises:
    - ValueError: If the string format is unknown or cannot be parsed.
    """
    value = original_value.replace(',', '.')

    regex = re.compile(r'((?P<coordinate>-?\d+(\.)\d+)(E)?)?((?P<exp>-?\d+?))?')
    parts = regex.match(value)
    if not parts or sum([1 for part in parts.groupdict().values() if part]) == 0:
        raise ValueError(f'Unknown string format: {value}')

    parts = parts.groupdict()
    if 'coordinate' in parts and parts['coordinate']:
        coordinate = float(parts['coordinate'])
        if 'exp' in parts and parts['exp']:
            coordinate *= 10 ** float(parts['exp'])
    else:
        raise ValueError(f'Unknown string format: {original_value}')

    return round(coordinate, 6)
