# Library
import json
import os
import re
from time import sleep
import botocore
import argparse
import datetime
import logging.config
from pathlib import Path
import yaml
import typer

import src.preprocess_data as pp
import src.clean_data as cd
import src.analysis as an
import src.generate_features as gf
import src.model_tuning as mt
import src.generate_preprocessor as gp
import src.split_data as sd
import src.model_evaluation as me
import src.aws_utils as aws



logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("pipeline")



def load_config(config_ref: str) -> dict:
    """
    Load configuration data either from an S3 bucket or from a local path.

    Parameters:
        config_ref (str): Reference to configuration data. If it starts with "s3://", it is 
            considered as an S3 URI and configuration data is downloaded from the corresponding 
            S3 bucket. Otherwise, it is considered as a local path.

    Returns:
        dict: Configuration data.

    Raises:
        EnvironmentError: If the config file does not exist at the specified path.
    """
    if config_ref.startswith("s3://"):
        # Get config file from S3
        config_file = Path("config/downloaded-config.yaml")
        logger.debug("Start downloading configuration file from AWS S3")
        try:
            bucket, key = re.match(r"s3://([^/]+)/(.+)", config_ref).groups()
            aws.download_s3(bucket, key, config_file)
            logger.info("Download configuration file from S3 succesfully")
        except AttributeError:  # If re.match() does not return groups
            logger.error("Could not parse S3 URI: %s", config_ref)
            config_file = Path("config/default.yaml")
        except botocore.exceptions.ClientError as e:  # If there is an error downloading
            logger.error("Unable to download config file from S3: %s", config_ref)
            logger.error(e)
    else:
        # Load config from local path
        config_file = Path(config_ref)
        logger.info("Configuration file read successfully")
    if not config_file.exists():
        raise EnvironmentError(f"Config file at {config_file.absolute()} does not exist")

    with config_file.open() as f:
        return yaml.load(f, Loader=yaml.SafeLoader)



def run_pipeline(config):
    """
    Run a data processing pipeline based on a given configuration.

    The pipeline involves the following steps:
        Setting up output directories.
        Downloading data from S3 bucket.
        Preprocessing, cleaning, and generating features from the dataset.
        Splitting the dataset.
        Tuning and comparing different models.
        Evaluating the best model.
        Saving results and uploading all artifacts to S3.

    Parameters:
        config (dict): Configuration for the pipeline.
    """
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

    # Tune the models
    rf_model, rf_par = mt.random_forest_tuning(X_train_transformed, y_train,config["model_tuning"])
    xgb_model, xgb_par = mt.xgboost_tuning(X_train_transformed, y_train,config["model_tuning"])
    lr_model, lr_par = mt.linear_ridge_tuning(X_train_transformed, y_train,config["model_tuning"])
    metrics_df, best_model, best_model_name, other_models_name = mt.model_comparison(rf_model,
                                                                                     xgb_model,
                                                                                     lr_model,
                                                                                     X_test_transformed,
                                                                                     y_test, config["model_tuning"])
    # Save the metrics and models
    mt.save_metrics(metrics_df, other_dir)
    best_model_file_name = "best_model_object_"+best_model_name+".pkl"
    mt.save_model(best_model, artifacts / best_model_file_name)
    for model_name in other_models_name:
        if model_name == "Random Forest":
            mt.save_model(rf_model, artifacts / "rf_model_object.pkl")
        elif model_name == "XGBoost":
            mt.save_model(xgb_model, artifacts / "xgb_model_object.pkl")
        elif model_name == "Linear Ridge":
            mt.save_model(lr_model, artifacts / "lr_model_object.pkl")

    # Model evaluation
    model_results = me.evaluate_model(best_model, X_test_transformed, y_test)
    fig_dict = me.plot_results(model_results)
    me.save_graphs(fig_dict, other_dir / config["model_evaluation"]["plot_results"]["output_dir"])

    # Upload all artifacts to S3
    aws_config = config.get("aws")
    aws.upload_artifacts(artifacts, aws_config)



def process_message(msg: aws.Message):
    """
    Process a message received from an SQS queue.

    The message body is expected to contain the bucket name and object key for an 
        S3 object that contains a pipeline configuration. This configuration is then used 
        to run a data processing pipeline.

    Parameters:
        msg (aws.Message): Message received from the SQS queue.
    """
    message_body = json.loads(msg.body)
    bucket_name = message_body["detail"]["bucket"]["name"]
    object_key = message_body["detail"]["object"]["key"]
    config_uri = f"s3://{bucket_name}/{object_key}"
    logger.info("Running pipeline with config from: %s", config_uri)
    config = load_config(config_uri)
    run_pipeline(config)



def main(
    sqs_queue_url: str,
    max_empty_receives: int = 3,
    delay_seconds: int = 10,
    wait_time_seconds: int = 10,
):
    """
    Continually poll an SQS queue for messages and process each one.

    This function will stop processing after a certain number of consecutive empty receives 
    (default is 3). After each message is processed, it is deleted from the queue.

    Parameters:
        sqs_queue_url (str): The URL of the SQS queue to poll for messages.
        max_empty_receives (int, optional): The number of consecutive empty receives 
            after which to stop processing. Default is 3.
        delay_seconds (int, optional): The number of seconds to sleep between each poll. 
            Default is 10.
        wait_time_seconds (int, optional): The duration (in seconds) for which the 
            call will wait for a message to arrive in the queue before returning. 
            Default is 10.
    """
    # Keep track of the number of times we ask queue for messages but receive none
    empty_receives = 0
    # After so many empty receives, we will stop processing and await the next trigger
    while empty_receives < max_empty_receives:
        logger.info("Polling queue for messages...")
        messages = aws.get_messages(
            sqs_queue_url,
            max_messages=2,
            wait_time_seconds=wait_time_seconds,
        )
        logger.info("Received %d messages from queue", len(messages))

        if len(messages) == 0:
            # Increment our empty receive count by one if no messages come back
            empty_receives += 1
            sleep(delay_seconds)
            continue

        # Reset empty receive count if we get messages back
        empty_receives = 0
        for m in messages:
            # Perform work based on message content
            try:
                process_message(m)
                logger.info("Pipeline finished.")
            # We want to suppress all errors so that we can continue processing next message
            except Exception as e:
                logger.error("Unable to process message, continuing...")
                logger.error(e)
                continue
            # We must explicitly delete the message after processing it
            aws.delete_message(sqs_queue_url, m.handle)
        # Pause before asking the queue for more messages
        sleep(delay_seconds)



if __name__ == "__main__":
    typer.run(main)