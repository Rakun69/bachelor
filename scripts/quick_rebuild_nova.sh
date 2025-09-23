#!/bin/bash
set -euo pipefail

echo "🔧 Quick Nova circuit rebuild..."

# Use existing Docker image and recompile Nova circuit
docker run --rm \
  --cpus=1.0 \
  --memory=2g \
  --memory-swap=2g \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/circuits:/app/circuits" \
  iot-zk-snark-eval -c "
    cd /app/circuits/nova
    echo 'Recompiling Nova circuit with batch_size=10...'
    zokrates compile -i iot_recursive.zok
    echo 'Setup Nova circuit...'
    zokrates nova setup
    echo 'Nova circuit rebuilt successfully!'
    echo 'Files:'
    ls -la *.out *.r1cs *.params 2>/dev/null || echo 'Some files may not exist yet'
  "

echo "✅ Nova circuit rebuilt!"
