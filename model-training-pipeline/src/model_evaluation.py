import logging
from pathlib import Path
from typing import Dict, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from sklearn.base import BaseEstimator
import numpy as np

logger = logging.getLogger(__name__)


def evaluate_model(model: BaseEstimator, x_test: Union[np.ndarray, pd.DataFrame],
                   y_test: Union[np.ndarray, pd.Series]) -> pd.DataFrame:
    """
    Evaluates the model on the test data and returns a DataFrame
    with the actual and predicted values.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets

    Returns:
        DataFrame with actual and predicted values
    """
    y_pred = model.predict(x_test)
    results = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})

    return results


def plot_results(model_results: pd.DataFrame) -> Dict[str, Figure]:
    """
    Function to plot the results of the model

    Args:
        model_results: DataFrame containing actual and predicted prices

    Returns:
        dict: Dictionary containing plot figures
    """
    fig_dict = {}
    # Histogram
    fig1, ax1 = plt.subplots(figsize=(20, 8))
    model_results.hist(ax=ax1)
    fig_dict['histogram'] = fig1

    # Density plot
    fig2, ax2 = plt.subplots(figsize=(20, 8))
    model_results.plot(kind='kde', ax=ax2)
    fig_dict['density_plot'] = fig2

    # Scatterplot of actual price vs predicted price
    fig3, ax3 = plt.subplots(figsize=(20, 8))
    ax3.scatter(model_results['Actual Price'], model_results['Predicted Price'], color='green')
    ax3.set_xlabel('Actual Price')
    ax3.set_ylabel('Predicted Price')
    ax3.set_title('Actual Price vs Predicted Price')
    fig_dict['scatterplot'] = fig3

    logging.info('Plots created successfully.')
    return fig_dict


def save_graphs(fig_dict: Dict[str, Figure], plot_dir: Path) -> None:
    """
    Function to save the plotted graphs

    Args:
        fig_dict: Dictionary containing plot figures
        plot_dir: Directory to save the plots

    Returns:
        None
    """
    try:
        # Check if the directory exists, if not, create it
        plot_dir.mkdir(parents=True, exist_ok=True)

        for fig_name, fig in fig_dict.items():
            if isinstance(fig, plt.Figure):  # ensure the object is a Matplotlib figure
                fig.savefig(plot_dir / f'{fig_name}.png')
                plt.close(fig)
            else:
                logging.error("Object under key '%s' is not a Matplotlib figure.", fig_name)

        logging.info('Graphs saved successfully.')
    except FileNotFoundError as e:
        logging.error('Error occurred while saving graphs: %s', str(e))
