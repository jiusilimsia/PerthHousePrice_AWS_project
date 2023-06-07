import logging
from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

logger = logging.getLogger(__name__)

def save_summary_table(all_data: pd.DataFrame, save_dir: Path) -> Path:
    """Saves a summary statistics table for a pandas dataframe

    Args:
        data: The pandas dataframe to describe
        save_dir: The directory in which to save the summary table

    Returns:
        A Path object representing the file path of the saved summary table
    """
    summary_path = save_dir / 'summary_table.csv'
    try:
        logging.info('Creating and saving summary table')
        summary_table = all_data.describe()
        summary_table.to_csv(summary_path)
        logging.info('Summary table saved successfully')
    except Exception as err:
        logging.error('Failed to create and save summary table: %s', err)
        raise
    return summary_path


def save_figures(all_data: pd.DataFrame, config: dict, save_dir: Path) -> List[Path]:
    """Creates and saves figures for each feature in a pandas dataframe

    Args:
        data: The pandas dataframe containing the data to be plotted
        config: The configuration dictionary
        save_dir: The directory in which to save the figures

    Returns:
        A list of Path objects representing the file paths of the saved figures
    """

    fig_paths = []

    try:
        logging.info('Creating and saving figures')
        num_cols = list(all_data.select_dtypes(['int64', 'float64']))

        all_data = all_data.replace([np.inf, -np.inf], np.nan)

        figs = []
        fig_names = ['histogram', 'boxplot', 'correlation_heatmap', 'pairplot']

        # Histograms
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        for i, ax in zip(num_cols, axes.ravel()):
            ax.hist(all_data[i].dropna(), bins=30, density=True)
            ax.set_title(i)
        figs.append(fig)

        # Box plots
        fig, axes = plt.subplots(2, 7, figsize=(20, 12))
        for i, ax in zip(num_cols, axes.ravel()):
            sns.boxplot(y=all_data[i], ax=ax)
        plt.tight_layout()
        figs.append(fig)

        # Correlation heatmap
        heatmap = all_data[num_cols]
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        matrix = np.triu(heatmap.corr())
        fig, _ = plt.subplots(figsize=(20, 13))
        sns.heatmap(heatmap.corr(), annot=True, fmt='.1f', vmin=-0.4, center=0, cmap=cmap, mask=matrix)
        plt.title('Correlation matrix', fontsize=18)
        figs.append(fig)

        # Pair plots
        fig, axs = plt.subplots(len(config['pairplot_columns']), len(config['pairplot_columns']), figsize=(20, 20))
        for i in range(len(config['pairplot_columns'])):
            for j in range(len(config['pairplot_columns'])):
                axs[i, j].scatter(all_data[config['pairplot_columns'][i]], all_data[config['pairplot_columns'][j]])
                if i == len(config['pairplot_columns']) - 1:
                    axs[i, j].set_xlabel(config['pairplot_columns'][j])
                if j == 0:
                    axs[i, j].set_ylabel(config['pairplot_columns'][i])
        figs.append(fig)

        logging.info('Figures saved successfully')
    except Exception as err:
        logging.error('Failed to create and save figures: %s', err)
        raise

    return fig_paths
