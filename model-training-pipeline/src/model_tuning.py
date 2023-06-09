import logging
from pathlib import Path
import pickle
from typing import Dict, Tuple, Union, List
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


def random_forest_tuning(train_data: pd.DataFrame, target: pd.Series,
                         config: Dict) -> Tuple[Union[RandomForestRegressor, None], Union[Dict, None]]:
    """
    Function to tune Random Forest hyperparameters

    :param train_data: Training data
    :param target: Training target
    :param config: Configuration dictionary for hyperparameters

    :return: Best model and best hyperparameters
    """
    try:
        rf_model = RandomForestRegressor()
        grid_search = GridSearchCV(estimator=rf_model, param_grid=config['random_forest_hyperparameters'],
                                   cv=config['cv'], n_jobs=-1, verbose=config['verbose'])
        grid_search.fit(train_data, target)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        logging.info('Random Forest tuning completed successfully')
        logging.debug('Result from Random Forest model: %s, %s', best_model, best_params)
        return best_model, best_params
    except (ValueError, TypeError) as e:
        logging.error('Error in Random Forest tuning: %s', e)
        return None, None

def xgboost_tuning(train_data: pd.DataFrame, target: pd.Series,
                   config: Dict) -> Tuple[Union[XGBRegressor, None], Union[Dict, None]]:
    """
    Function to tune XGBoost hyperparameters

    :param train_data: Training data
    :param target: Training target
    :param config: Configuration dictionary for hyperparameters

    :return: Best model and best hyperparameters
    """
    try:
        xgb = XGBRegressor()
        grid_search = GridSearchCV(estimator=xgb, param_grid=config['xgboost_hyperparameters'],
                                   cv=config['cv'], n_jobs=-1, verbose=config['verbose'])
        grid_search.fit(train_data, target)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        logging.info('XGBoost tuning completed successfully')
        logging.debug('Result from XGBoost model: %s, %s', best_model, best_params)
        return best_model, best_params
    except (ValueError, TypeError) as e:
        logging.error('Error in XGBoost tuning: %s', e)
        return None, None

def linear_ridge_tuning(train_data: pd.DataFrame, target: pd.Series,
                        config: Dict) -> Tuple[Union[Ridge, None], Union[Dict, None]]:
    """
    Function to tune Linear Ridge Regression hyperparameters

    :param train_data: Training data
    :param target: Training target
    :param config: Configuration dictionary for hyperparameters

    :return: Best model and best hyperparameters
    """
    try:
        ridge = Ridge()
        grid_search = GridSearchCV(estimator=ridge, param_grid=config['linear_ridge_hyperparameters'],
                                   cv=config['cv'], n_jobs=-1, verbose=config['verbose'])
        grid_search.fit(train_data, target)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        logging.info('Linear Ridge Regression tuning completed successfully')
        logging.debug('Result from Linear Ridge model: %s, %s', best_model, best_params)
        return best_model, best_params
    except (ValueError, TypeError) as e:
        logging.error('Error in Linear Ridge Regression tuning: %s', e)
        return None, None


def model_comparison(random_forest_model: RandomForestRegressor, xgboost_model: XGBRegressor, linear_ridge_model: Ridge,
                     test_data: pd.DataFrame, target_test: pd.Series,
                     config: Dict) -> Tuple[pd.DataFrame, Union[RandomForestRegressor, XGBRegressor, Ridge, None],
                                            Union[str, None], List[str]]:
    """
    Function to compare performance of tuned models

    :param random_forest_model: Tuned Random Forest model
    :param xgboost_model: Tuned XGBoost model
    :param linear_ridge_model: Tuned Linear Ridge model
    :param test_data: Test data
    :param target_test: Test target
    :param config: Configuration dictionary for metrics

    :return: Metrics results, best model, best model name, and other models' name
    """
    try:
        models = {'Random Forest': random_forest_model, 'XGBoost': xgboost_model, 'Linear Ridge': linear_ridge_model}
        model_names = ['Random Forest', 'XGBoost', 'Linear Ridge']
        metrics_results = {metric: [] for metric in config['metrics']}

        logging.debug('Begin building result dataframe')
        for _, model in models.items():
            predictions = model.predict(test_data)
            logging.debug('Check predictions')
            for metric in config['metrics']:
                logging.debug('metric: %s', metric)
                if metric == 'root_mean_squared_error':
                    logging.debug('rmse')
                    metrics_results[metric].append(np.sqrt(mean_squared_error(target_test, predictions)))
                else:
                    logging.debug('non-rmse')
                    # Mapping from metric name to function
                    metrics_map = {
                        'r2_score': r2_score,
                        'mean_absolute_error': mean_absolute_error,
                        'root_mean_squared_error': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                        # Add other metrics if needed
                    }

                    # Then replace eval(metric) with metrics_map[metric]
                    metrics_results[metric].append(metrics_map[metric](target_test, predictions))
                    logging.debug('Append to metrics_results: %s', metrics_map[metric](target_test, predictions))

        metrics_df = pd.DataFrame(metrics_results, index=model_names)
        logging.debug('metrics_df: %s', metrics_df)

        # If the best model metric is RMSE, we want the model with the smallest RMSE, so use idxmin instead of idxmax
        if config['best_model_metric'] == 'r2_score':
            best_model_name = metrics_df[config['best_model_metric']].idxmax()
        else:
            best_model_name = metrics_df[config['best_model_metric']].idxmin()

        best_model = models[best_model_name]
        logging.info('Model comparison completed successfully')
        other_models_name = set(model_names)
        other_models_name.remove(best_model_name)
        return metrics_df, best_model, best_model_name, list(other_models_name)
    except (ValueError, TypeError) as e:
        logging.error('Error in model comparison: %s', e)
        return None, None


def save_metrics(metrics_df: pd.DataFrame, artifacts: Path) -> None:
    """
    Function to save metrics dataframe to a csv file

    :param metrics_df: DataFrame containing metrics results
    :param filename: Name of the file to save
    """
    try:
        metrics_df.to_csv(artifacts / 'metrics_result')
        logging.info('Metrics result saved successfully to %s', artifacts)
    except FileNotFoundError as e:
        logger.error('Error while saving metrics data: %s', e)
        raise
    except PermissionError as e:
        logger.error('Permission denied when attempting to save metrics data: %s', e)
        raise
    except TypeError as e:
        logger.error('TypeError occurred when attempting to save metrics data: %s', e)
        raise

def save_model(model: Union[RandomForestRegressor, XGBRegressor, Ridge], file_path: Path) -> None:
    """
    Function to save model object to a pickle file

    :param model: Model object to save
    :param file_path: A Path object representing the path to the file where the model should be saved.
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info('Model saved successfully at %s', file_path)
    except FileNotFoundError as e:
        logger.error('Error while saving the model: %s', e)
        raise
    except pickle.PicklingError as e:
        logger.error('Error while pickling the model: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error while saving the model: %s', e)
        raise
