#!/usr/bin/env bash
set -euo pipefail

READING_COUNTS=${1:-"10,20,30"}
IMAGE_NAME=iot-zk-snark-eval
HOST_PROJECT_DIR="/home/ramon/bachelor"
CONTAINER_PROJECT_DIR="/app"

echo "[INFO] Building Docker image ${IMAGE_NAME}..."
docker build --no-cache -t ${IMAGE_NAME} .

echo "[INFO] Running container.."
docker run --rm \
  --cpus=6 \
  --memory=2g \
  -v "${HOST_PROJECT_DIR}:${CONTAINER_PROJECT_DIR}" \
  -w "${CONTAINER_PROJECT_DIR}" \
  ${IMAGE_NAME} \
  bash -c " \
    find . -name '*.pyc' -delete && \
    find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true && \
    python3 -c \"import numpy; print('NumPy version:', numpy.__version__)\" && \
    python3 scripts/measure_crossover_real.py \
      --reading-counts ${READING_COUNTS} \
      --warmup-runs 0 \
      --repetitions 0 \
      --mode warm \
  "

echo "[INFO] Done."
