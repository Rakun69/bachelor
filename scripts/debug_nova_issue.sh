#!/bin/bash
set -euo pipefail

echo "🔍 Debugging Nova issue..."

# Debug Nova circuit compilation and input format
docker run --rm \
  --cpus=1.0 \
  --memory=2g \
  --memory-swap=2g \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/circuits:/app/circuits" \
  -v "$(pwd)/scripts:/app/scripts" \
  iot-zk-snark-eval -c "
    cd /app/circuits/nova
    echo '=== Nova Circuit Debug ==='
    echo 'Circuit file:'
    cat iot_recursive.zok
    echo ''
    echo 'Trying to compile...'
    zokrates compile -i iot_recursive.zok
    echo 'Compilation successful!'
    echo ''
    echo 'Trying Nova setup...'
    zokrates nova setup
    echo 'Setup successful!'
    echo ''
    echo 'Testing with simple input...'
    echo '{\"sum\": \"0\", \"count\": \"0\"}' > init.json
    echo '[{\"values\": [\"1\", \"2\", \"3\", \"4\", \"5\", \"6\", \"7\", \"8\", \"9\", \"10\"], \"batch_id\": \"1\"}]' > steps.json
    echo 'Input files created:'
    cat init.json
    echo ''
    cat steps.json
    echo ''
    echo 'Trying Nova prove...'
    zokrates nova prove
    echo 'Nova prove successful!'
  "

echo "✅ Debug completed!"
