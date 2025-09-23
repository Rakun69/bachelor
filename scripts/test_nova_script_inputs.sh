#!/bin/bash
set -euo pipefail

echo "🔍 Testing Nova script input generation..."

# Test the script's input generation with a small dataset
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
    echo 'Testing with just 10 readings...'
    python scripts/measure_crossover_real.py --reading-counts '10' --repetitions 1
    echo 'Test completed!'
  "

echo "✅ Script test completed!"
