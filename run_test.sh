#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=iot-zk-zkminimal

echo "[INFO] Build minimal image..."
docker build -f Dockerfile.minimal -t ${IMAGE_NAME} .

echo "[INFO] Run test..."
docker run --rm -v "$(pwd):/app" -w /app ${IMAGE_NAME} \
    python3 scripts/test_numpy.py

echo "[INFO] Done."
