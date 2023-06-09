from typing import List
import pickle
import logging
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


logger = logging.getLogger(__name__)

def generate_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Generates a preprocessor for numerical and categorical features.

    Args:
        num_cols (List[str]): List of numerical column names.
        cat_cols (List[str]): List of categorical column names.

    Returns:
        ColumnTransformer: Preprocessor for transforming numerical and categorical features.
    """
    # Define the transformer for numerical features
    numerical_transformer = MinMaxScaler(feature_range=(0, 1))

    # Define the transformer for categorical features
    # The best way to encode categorical data is to use CatBoostEncoder
    categorical_transformer = Pipeline(steps=[
        ('cat_encoder', ce.CatBoostEncoder())
    ])

    # Create the preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )

    logging.info('Preprocessor generated successfully.')
    return preprocessor

def save_preprocessor(preprocessor: ColumnTransformer, filename: str) -> None:
    """
    Save the preprocessor to a file.

    Args:
        preprocessor (ColumnTransformer): The preprocessor to save.
        filename (str): The file to save the preprocessor to.

    Returns:
        None
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(preprocessor, f)
        logging.info('Preprocessor saved successfully to %s.', filename)
    except (IOError, OSError, pickle.PicklingError) as e:
        logging.error('Failed to save preprocessor: %s', str(e))
