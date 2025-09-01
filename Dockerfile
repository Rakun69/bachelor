# Dockerfile for IoT ZK-SNARK Evaluation System
FROM ubuntu:22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install ZoKrates
RUN curl -LSfs get.zokrat.es | sh
ENV PATH="/root/.zokrates/bin:${PATH}"

# Create working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY src/ ./src/
COPY circuits/ ./circuits/
COPY configs/ ./configs/
COPY run_evaluation.sh .

# Setup Python virtual environment
RUN python3 -m venv iot_zk_env
RUN /bin/bash -c "source iot_zk_env/bin/activate && pip install --upgrade pip"
RUN /bin/bash -c "source iot_zk_env/bin/activate && pip install -r requirements.txt"

# Create data directories
RUN mkdir -p data/{raw,benchmarks,proofs,visualizations,comparison,iot_analysis}

# Make run script executable
RUN chmod +x run_evaluation.sh

# Set entrypoint
ENTRYPOINT ["/bin/bash"]
