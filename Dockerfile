FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

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

# Install ZoKrates 0.8.8
RUN curl -L https://github.com/Zokrates/ZoKrates/releases/download/0.8.8/zokrates-0.8.8-x86_64-unknown-linux-gnu.tar.gz \
    | tar xz -C /usr/local/bin \
    && chmod +x /usr/local/bin/zokrates

WORKDIR /app

COPY requirements.txt ./
COPY src/ ./src/
COPY circuits/ ./circuits/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Erstelle venv in /opt (außerhalb von /app, damit sie nicht überschrieben wird)
RUN python3 -m venv --copies /opt/iot_zk_env

# Install dependencies using the venv's pip
RUN /opt/iot_zk_env/bin/pip install --upgrade pip
RUN /opt/iot_zk_env/bin/pip install -r requirements.txt

RUN chmod +x scripts/run_in_docker.sh
RUN mkdir -p data/{raw,benchmarks,proofs,visualizations,comparison,iot_analysis}

ENV PYTHONPATH=/app
ENV VIRTUAL_ENV=/opt/iot_zk_env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

CMD ["bash"]
