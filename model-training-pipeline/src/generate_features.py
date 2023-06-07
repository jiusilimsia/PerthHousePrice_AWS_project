import logging
import pandas as pd

logger = logging.getLogger(__name__)

def generate_features(all_data: pd.DataFrame, config: dict) -> (pd.DataFrame, list, list):
    """
    Generates additional features based on the given data and configuration.

    Args:
        all_data (pd.DataFrame): The input data.
        config (dict): Configuration parameters.

    Returns:
        pd.DataFrame: The modified data with additional features.
        list: The numerical columns in the modified data.
        list: The categorical columns in the modified data.

    """
    try:
        # Calculate additional features
        all_data['FREE_AREA'] = all_data['LAND_AREA'] - all_data['FLOOR_AREA']
        all_data['OTHERS_ROOMS_AREA'] = all_data['FLOOR_AREA'] * config.get('other_room_multiplier', 0.25)
        all_data['OTHERS_ROOMS_AREA'] = all_data['FLOOR_AREA'] * config.get('other_room_multiplier', 0.25)

        other_rooms_area = all_data['FLOOR_AREA'] - all_data['OTHERS_ROOMS_AREA']
        bed_bath_sum = all_data['BEDROOMS'] + all_data['BATHROOMS']
        all_data['GARAGE_AREA'] = other_rooms_area / bed_bath_sum
        bed_garage_sum = all_data['BEDROOMS'] + all_data['GARAGE']
        all_data['BATHROOMS_AREA'] = other_rooms_area / bed_garage_sum
        bath_garage_sum = all_data['BATHROOMS'] + all_data['GARAGE']
        all_data['BEDROOMS_AREA'] = other_rooms_area / bath_garage_sum
        # Drop unnecessary columns
        data = all_data.drop(['FREE_AREA', 'BUILD_YEAR', 'NEAREST_SCH_DIST',
                              'NEAREST_STN_DIST', 'POSTCODE', 'LONGITUDE', 'CBD_DIST'], axis=1)
        # Extract features for modeling
        features = data.drop(['ADDRESS', 'DATE_SOLD', 'PRICE'], axis=1)
        num_cols = list(features.select_dtypes(['int64', 'float64']))
        cat_cols = list(features.select_dtypes('object'))

        logger.info('Features were generated successfully.')
        return data, num_cols, cat_cols
    except Exception as e:
        logger.error('Error occurred while generating features: %s', str(e))
        raise e

def save_features(all_data: pd.DataFrame, data_path: str) -> None:
    """
    Saves the features to a CSV file.

    Args:
        all_data (pd.DataFrame): The features to be saved.
        data_path (str): The path where the features should be saved.

    Returns:
        None

    """
    try:
        all_data.to_csv(data_path, index=False)
        logging.info('Features were saved successfully to %s.', data_path)
    except Exception as e:
        logger.error('Error occurred while saving features: %s', str(e))
        raise e
