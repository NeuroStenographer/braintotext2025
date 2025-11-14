# Base Dockerfile for braintotext2025 (b2txt25 environment)
# Ubuntu-based image with Python, Kedro, and PySpark for data processing
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    openjdk-11-jdk-headless \
    curl \
    git \
    build-essential \
    libhdf5-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment for PySpark
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the project in editable mode (will be done after source is copied)
# Copy the entire project
COPY . /app/

# Install the project
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data/01_raw \
    /app/data/02_intermediate \
    /app/data/03_primary \
    /app/data/04_feature \
    /app/data/05_model_input \
    /app/data/06_models \
    /app/data/07_model_output \
    /app/data/08_reporting

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV KEDRO_LOGGING_CONFIG=/app/conf/logging.yml

# Expose port for Kedro Viz
EXPOSE 4141

# Default command
CMD ["kedro", "run"]
