#!/bin/bash
set -euo pipefail

echo "🔧 Rebuilding Nova circuit with batch_size=10..."

# Build Docker image
echo "[INFO] Building Docker image..."
docker build -t iot-zk-snark-eval .

# Run container to recompile Nova circuit
echo "[INFO] Recompiling Nova circuit in Docker..."
docker run --rm \
  --cpus=1.0 \
  --memory=2g \
  --memory-swap=2g \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/circuits:/app/circuits" \
  iot-zk-snark-eval -c "
    cd /app/circuits/nova
    echo 'Compiling Nova circuit with batch_size=10...'
    zokrates compile -i iot_recursive.zok
    echo 'Setup Nova circuit...'
    zokrates nova setup
    echo 'Nova circuit rebuilt successfully!'
    ls -la
  "

echo "✅ Nova circuit rebuilt with batch_size=10"
