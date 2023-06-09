# Library
import json
import os
import re
from time import sleep
import datetime
import logging.config
from pathlib import Path
import yaml
import argparse

import src.preprocess_data as pp
import src.clean_data as cd
import src.analysis as an
import src.generate_features as gf
import src.split_data as sd
import src.aws_utils as aws
import src.generate_preprocessor as gp



logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("pipeline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Acquire, clean, and create features from clouds data"
    )
    parser.add_argument(
        "--config",
        default="config/default-config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    # Load configuration file for parameters and run config
    with open(args.config, "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.error.YAMLError as e:
            logger.error("Error while loading configuration from %s", args.config)
        else:
            logger.info("Configuration file loaded from %s", args.config)

    run_config = config.get("run_config", {})

    # Set up output directory for saving artifacts
    now = int(datetime.datetime.now().timestamp())
    artifacts = Path(run_config.get("output", "runs")) / str(now)
    artifacts.mkdir(parents=True)

    # build folder for artifacts without model
    other_dir = artifacts / Path("other_artifacts")
    other_dir.mkdir(parents=True)

    data_dir = other_dir / Path(run_config.get("output_data", "data"))
    data_dir.mkdir(parents=True)

    # Save config file to artifacts directory for traceability
    config_dir = other_dir / Path(run_config.get("output_config", "config"))
    config_dir.mkdir(parents=True)

    with (config_dir / "config.yaml").open("w") as f:
        yaml.dump(config, f)

    data_root = Path("data")
    data_root.mkdir(parents=True, exist_ok=True)
    input_data = run_config["input_data"]

    # Download data from S3 raw bucket
    pp.download_s3(config["aws"]["bucket_raw"], input_data, data_root / input_data)

    # Get the file path within the data directory
    file_path = os.path.join(data_root, run_config["input_data"])

    # Create structured dataset from raw data; save to disk
    all_data,num_cols,cat_cols = pp.preprocess_dataset(file_path, config["preprocess_dataset"])
    pp.save_dataset(all_data, data_dir / "house_preprocessed.csv")
    logger.info("Finished preprocessing the dataset.")

    # EDA
    figures = other_dir / "figures&tables"
    figures.mkdir()  # Replace with your desired directory
    an.save_summary_table(all_data, figures)
    an.save_figures(all_data, config["analysis"], figures)
    # clean the data
    cleaned_data = cd.clean_dataset(all_data, config["clean_data"])
    logger.info("Finished cleaning the dataset.")
    # generated features
    features,updated_num_cols,updated_cat_cols = gf.generate_features(cleaned_data, config["generate_features"])
    # Save the features dataset in the disk
    gf.save_features(features, data_dir / "house_features.csv")
    logger.info("Finished generating features.")
    preprocessor = gp.generate_preprocessor(updated_num_cols,updated_cat_cols)
    logger.info("Finished generating preprocessor.")
    X_train_transformed, X_test_transformed, y_train, y_test, fitted_preprocessor = sd.split_data(features,
                                                                                                  preprocessor,
                                                                                                  config["split_data"])
    gp.save_preprocessor(fitted_preprocessor, artifacts / "fitted_preprocessor.pkl")
    sd.save_splited_data(X_train_transformed, X_test_transformed, y_train, y_test, data_dir)
    logger.info("Finished splitting the data.")