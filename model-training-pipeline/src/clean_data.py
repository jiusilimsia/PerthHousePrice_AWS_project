import logging
import pandas as pd

logger = logging.getLogger(__name__)

def clean_dataset(all_data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Cleans the dataset based on specified thresholds defined in the 'config' dictionary.

    Args:
        all_data (pd.DataFrame): The dataset to be cleaned.
        config (dict): A dictionary containing configuration parameters.

    Returns:
        pd.DataFrame: The cleaned dataset.

    """
    try:
        outlier_config = config['outlier']['upper_threshold']
        # Remove rows where price is above the upper threshold (default: $1.5 million)
        all_data.drop(
            all_data[all_data['PRICE'] > outlier_config.get('price', 1500000)].index,
            inplace=True
        )
        # Remove rows where LAND_AREA is above the upper threshold (default: 1500)
        all_data.drop(
            all_data[all_data['LAND_AREA'] > outlier_config.get('land_area', 1500)].index,
            inplace=True
        )
        # Remove rows where BUILD_YEAR is below the lower threshold (default: 1950)
        all_data.drop(
            all_data[all_data['BUILD_YEAR'] < config['outlier']['lower_threshold'].get('build_year', 1950)].index,
            inplace=True
        )
        # Remove rows where NEAREST_STN_DIST is above the upper threshold (default: 10000)
        all_data.drop(
            all_data[all_data['NEAREST_STN_DIST'] > outlier_config.get('nearest_stn_dist', 10000)].index,
            inplace=True
        )
        # Remove rows where NEAREST_SCH_DIST is above the upper threshold (default: 4)
        all_data.drop(
            all_data[all_data['NEAREST_SCH_DIST'] > outlier_config.get('nearest_sch_dist', 4)].index,
            inplace=True
        )
        # Remove rows where year is NaN
        all_data.dropna(inplace=True)
        logger.info('Dataset has been cleaned successfully.')

    except Exception as e:
        logger.error('Error occurred while cleaning dataset: %s', str(e))
        raise e

    return all_data
