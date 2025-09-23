#!/bin/bash
set -euo pipefail

echo "🚀 Running optimized Nova test directly..."

# Run the measurement directly in existing Docker container
docker run --rm \
  --cpus=1.0 \
  --memory=2g \
  --memory-swap=2g \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/circuits:/app/circuits" \
  -v "$(pwd)/scripts:/app/scripts" \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/configs:/app/configs" \
  iot-zk-snark-eval -c "
    source iot_zk_env/bin/activate
    echo 'Running optimized Nova test with batch_size=10...'
    python scripts/measure_crossover_real.py --reading-counts '10,25,50,75,100'
    echo 'Test completed!'
  "

echo "✅ Optimized test completed!"
