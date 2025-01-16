# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in container
WORKDIR /app

# Install system dependencies required for h5py and other packages
RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file into container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest into container
COPY . .

RUN export $(grep -v '^#' docker.env | xargs)
ENV EXP_DIR=/app/Experiments
ENV DB_DIR=/app/Droplet_db
ENV MODEL_DIR=/app/CNN_models
ENV expID_pattern='[A-Z]{2,4}_[A-Z0-9]{2,4}_[0-9]{3}'

# Start a bash shell
CMD ["python", "main.py"]


