#FROM python:3.9-slim
FROM --platform=linux/x86_64 python:3.9-slim
# set working directory
WORKDIR /app

# copy project files to the container
COPY src /app/src
COPY requirements.txt /app/requirements.txt
COPY interface.py /app/interface.py
COPY config /app/config
COPY data /app/data
COPY .streamlit /app/.streamlit

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 80 for http traffic
EXPOSE 80

# Set the command to run the Streamlit application
CMD ["streamlit", "run", "--server.port=80", "--server.fileWatcherType=none", "interface.py"]
#CMD ["streamlit","run","interface.py"]
