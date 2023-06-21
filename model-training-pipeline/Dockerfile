FROM --platform=linux/x86_64 python:3.9-slim

# set working directory
WORKDIR /app

# copy project files to the container
COPY src /app/src
COPY requirements.txt /app/requirements.txt
COPY pipeline.py /app/pipeline.py
COPY config /app/config
COPY data /app/data

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# set the entrypoint
ENTRYPOINT ["python", "pipeline.py"]