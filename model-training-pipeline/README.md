# Pipeline for Training Random Forest, XGBoost, and Logistic Regression Models

This pipeline is designed to facilitate the training and evaluation of three popular machine learning models: Random Forest, XGBoost, and Logistic Regression. The goal is to provide a comprehensive framework that enables users to easily experiment with and compare the performance of these models on their datasets.

## Introduction

The Random Forest, XGBoost, and Logistic Regression models are widely used for various machine learning tasks, including classification and regression. This pipeline streamlines the process of training, optimizing, and evaluating these models, making it easier for researchers and practitioners to leverage their strengths and compare their performance.

## Model Pipeline Detailed Breakdown

Our model pipeline includes various stages from data preprocessing to model evaluation. Here is a step-by-step explanation:

1. **Load Configuration**: The pipeline starts by loading the YAML configuration file specified in the command line argument. This file contains all the parameters and paths necessary to run the pipeline.

2. **Setup Directories**: After loading the configuration file, directories for saving output artifacts are created based on the current timestamp. The configuration file itself is also saved in these directories for traceability.

3. **Preprocessing**: The raw dataset is preprocessed by handling missing values, outlier detection, and removal, etc. The preprocessed data is then saved back to disk.

4. **Exploratory Data Analysis (EDA)**: The pipeline performs exploratory data analysis (EDA) on the preprocessed data. It generates summary statistics and visualizations that are saved for further analysis.

5. **Data Cleaning**: Any necessary cleaning operations are performed on the dataset, such as imputing missing values, scaling, or encoding categorical variables.

6. **Feature Generation**: New features are generated based on the clean data, improving the performance of the model.

7. **Preprocessing & Data Splitting**: A preprocessing pipeline is generated and fitted. The dataset is then split into training and testing subsets.

8. **Model Tuning & Selection**: Three different models, namely Random Forest, XGBoost, and Linear Ridge, are tuned and compared on the training data. The model with the best performance on the test data is selected as the best model.

9. **Model Evaluation**: The best model is evaluated on the test data. The evaluation metrics and plots are saved for review.

10. **Artifact Saving**: All generated artifacts, including preprocessed data, models, evaluation results, and figures are saved to a designated directory.

11. **Upload to S3**: Lastly, all artifacts are uploaded to an AWS S3 bucket for storage and easy accessibility.

The pipeline is built in a modular way so that each function in the pipeline can be modified or replaced as necessary.

## Requirements

To use this pipeline, you need to have the following dependencies installed:
- Python 3.x
- boto3==1.26.142
- botocore==1.29.142
- category-encoders==2.6.1
- contourpy==1.0.7
- cycler==0.11.0
- fonttools==4.39.4
- jmespath==1.0.1
- joblib==1.2.0
- kiwisolver==1.4.4
- matplotlib==3.7.1
- numpy==1.24.3
- packaging==23.1
- pandas==2.0.1
- patsy==0.5.3
- Pillow==9.5.0
- pyparsing==3.0.9
- python-dateutil==2.8.2
- pytz==2023.3
- PyYAML==6.0
- s3transfer==0.6.1
- scikit-learn==1.2.2
- scipy==1.10.1
- seaborn==0.12.2
- six==1.16.0
- statsmodels==0.14.0
- threadpoolctl==3.1.0
- tzdata==2023.3
- urllib3==1.26.16
- xgboost==1.7.5

## Installation

1. Clone this repository to your local machine.
2. Install the required dependencies using pip:
   ```
   pip install -r requirements.txt
   ```

Note: It is recommended to set up a virtual environment before installing the dependencies to avoid any conflicts with existing packages.

Now you are ready to use the pipeline and train the Random Forest, XGBoost, and Logistic Regression models on your dataset.

## Run

To run the pipeline, follow these steps:

1. Ensure that you have installed all the required dependencies as mentioned in the "Requirements" section.
2. Open a terminal or command prompt.
3. Navigate to the directory where the pipeline files are located.
4. Run the pipeline script using Python:
   ```
   python pipeline.py
   ```
5. The pipeline will execute, loading your dataset, training the Random Forest, XGBoost, and Logistic Regression models, and evaluating their performance.
6. The results will be displayed in the console or saved to a file, depending on how you have configured the pipeline script.

## How to Run in Docker

We've also provided a Dockerfile that sets up everything, including installing packages and running the pipeline. To run the Docker container, follow the steps below:

### Connect to AWS through SSO

Ensure that the AWS CLI is installed and available

```bash
aws --version
```

Be sure that you have logged in to your account using SSO

```bash
aws --profile personal-sso-admin sso login
```

Set your default AWS profile for the remainder of your shell session:

```bash
export AWS_DEFAULT_PROFILE=personal-sso-admin
```

Confirm that you are authenticated as the proper identity:

```bash
aws sts get-caller-identity
```

Follow the steps below to build and run the Docker image for the app and tests:

### Build the Docker Image for the App

```bash
docker build -t pipelineimg -f Dockerfile . 
```

### Run the entire model pipeline

```bash
docker run -v ~/.aws:/root/.aws -e AWS_PROFILE=personal-sso-admin pipelineimg
```

## Cloud Service Used

### AWS ECR

AWS Elastic Container Registry (ECR) is where we store Docker images of our application. Each update to the application generates a new Docker image which gets pushed into ECR. This enables version control and seamless deployments.

### AWS ECS

Amazon Elastic Container Service (ECS) is where our application is hosted and run. It pulls the latest Docker image from ECR and manages the execution, scaling, and load balancing of our application.

