import os
import logging
import boto3
import botocore
from dataclasses import dataclass


logger = logging.getLogger(__name__)


def upload_artifacts(artifacts, config):
    """Upload all the artifacts in the specified directory to S3

    Args:
        artifacts: Directory containing all the artifacts from a given experiment
        config: Config required to upload artifacts to S3; see example config file for structure

    Returns:
        List of S3 uri's for each file that was uploaded
    """
    # Retrieve the S3 configuration from the config dictionary
    upload = config["upload"]
    bucket_name = config["bucket_model_artifacts"]

    # If the upload flag is set to False, skip the upload process
    if not upload:
        logger.info("Upload is disabled in the configuration.")
        return []

    # Create a boto3 session and S3 client
    session = boto3.Session()
    s3_client = session.client("s3")

    # Initialize a list to store the S3 URIs of the uploaded files
    uploaded_uris = []

    try:
        # Iterate through all files in the artifacts directory
        for root, _, files in os.walk(artifacts):
            for file in files:
                file_path = os.path.join(root, file)
                # Create the S3 key by replacing the local artifacts path with the prefix
                s3_key = file_path.replace(str(artifacts), "", 1).lstrip(
                    os.sep
                )

                # Upload the file to S3
                s3_client.upload_file(Filename=file_path, Bucket=bucket_name, Key=s3_key)

                # Add the uploaded file's S3 URI to the list
                uploaded_uri = f"s3://{bucket_name}/{s3_key}"  # use f-string
                uploaded_uris.append(uploaded_uri)

                # logger.debug("Successfully uploaded %s to %s", file_path, uploaded_uri)

    except botocore.exceptions.BotoCoreError as e:  # catch specific exception
        logger.error("Failed to upload files: %s", str(e))
        return []

    logger.info("All files have been uploaded successfully.")

    return uploaded_uris

def download_s3(bucket_name: str, s3_key: str, local_file: str) -> None:
    """
    Download a file from an AWS S3 bucket.

    Parameters:
        bucket_name (str): The name of the S3 bucket.
        s3_key (str): The key of the object to download.
        local_file (str): The local path where the file will be saved.

    Returns:
        None
    """
    s3_client = boto3.client("s3")
    try:
        s3_client.download_file(bucket_name, s3_key, str(local_file))
        logger.info(
        "Download successful. File downloaded from bucket '%s' with key '%s' to '%s'.",
        bucket_name,
        s3_key,
        local_file
        )
    except botocore.exceptions.BotoCoreError as e:  # catch specific exception
        logger.error("Download failed. Exception: %s", e)


@dataclass
class Message:
    handle: str
    body: str


def get_messages(
    queue_url: str,
    max_messages: int = 1,
    wait_time_seconds: int = 1,
) -> list[Message]:
    """
    Retrieve a list of messages from an Amazon SQS queue.

    This function uses boto3, the AWS SDK for Python, to retrieve messages
    from a specified SQS queue. It returns a list of Message objects.

    Parameters:
        queue_url (str): The URL of the SQS queue from which to retrieve messages.
        max_messages (int, optional): The maximum number of messages to retrieve. Defaults to 1.
        wait_time_seconds (int, optional): The duration (in seconds) for which the 
            call will wait for a message to arrive in the queue before returning. Defaults to 1.

    Returns:
        list[Message]: A list of Message objects. Each Message object contains 
        the 'ReceiptHandle' and the 'Body' of the retrieved message. If no message 
        is available, returns an empty list.

    Exceptions:
        If a boto3.client("sqs") operation error occurs, it logs the error and returns 
        an empty list.
    """
    sqs = boto3.client("sqs")
    try:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_time_seconds,
        )
    except botocore.exceptions.ClientError as e:
        logger.error(e)
        return []
    if "Messages" not in response:
        return []
    return [Message(m["ReceiptHandle"], m["Body"]) for m in response["Messages"]]


def delete_message(queue_url: str, receipt_handle: str):
    """
    Deletes a specified message from an Amazon SQS queue.

    This function uses boto3, the AWS SDK for Python, to delete a message 
    from a specified SQS queue.

    Parameters:
        queue_url (str): The URL of the SQS queue from which to delete the message.
        receipt_handle (str): The receipt handle associated with the message to delete. 

    Returns:
        None

    Note:
        The receipt_handle is the identifier you must provide to delete the message.
    """
    sqs = boto3.client("sqs")
    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
