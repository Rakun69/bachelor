#!/bin/bash
set -euo pipefail

echo "🔍 Debugging script-generated inputs..."

# Debug what inputs the script actually generates
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
    cd /app
    python3 -c \"
import sys
sys.path.append('/app')
from scripts.measure_crossover_real import load_iot_data, prepare_nova_inputs
from pathlib import Path
import json

# Load data
project_root = Path('/app')
data = load_iot_data(project_root)
print(f'Loaded {len(data)} IoT readings')

# Test with 10 readings
subset = data[:10]
print(f'Testing with {len(subset)} readings')

# Prepare Nova inputs
nova_dir, steps = prepare_nova_inputs(10, subset)
print(f'Generated {steps} Nova steps')

# Check generated files
print('Generated init.json:')
with open(nova_dir / 'init.json') as f:
    print(f.read())

print('Generated steps.json:')
with open(nova_dir / 'steps.json') as f:
    steps_data = json.load(f)
    print(json.dumps(steps_data, indent=2))
\"
  "

echo "✅ Debug completed!"
