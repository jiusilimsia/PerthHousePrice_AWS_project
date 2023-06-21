# House Price Prediction - Deployment with Docker and AWS (ECS & ECR)

This README illustrates the process of deploying a House Price Prediction application using Docker, AWS Elastic Container Service (ECS) and Elastic Container Registry (ECR). This Streamlit-based application provides a user-friendly interface for predicting house prices based on different parameters, such as location, number of bathrooms, and number of living rooms.

## Prerequisites

- Docker installed on your machine
- AWS CLI installed and configured with your AWS credentials
- All project files available locally

## Overview of the Deployment Process

1. **Build the Docker Image:** Using Docker, we package the application and its dependencies into a Docker image based on the instructions provided in the Dockerfile.

2. **Test the Docker Image Locally:** The Docker image is run locally to ensure everything is working as expected.

3. **Push the Docker Image to AWS ECR:** We then push the Docker image to AWS's Elastic Container Registry (ECR). ECR is a fully-managed Docker container registry that makes it easy for developers to store, manage, and deploy Docker container images.

4. **Deploy the Docker Image on AWS ECS:** Finally, the Docker image is deployed to the Elastic Container Service (ECS), AWS's highly scalable, high performance container orchestration service. ECS allows you to easily run and scale containerized applications on AWS.

This process enables the application to run on a scalable, managed service in AWS, without the need for you to handle the underlying infrastructure. Users can access the application via a web interface and adjust various parameters to get custom house price predictions.
