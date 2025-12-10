# Verifiable Data Transformations in IoT Environments using Recursive zk-SNARKs

This repository contains the implementation and evaluation pipeline for a smart-home IoT use case that compares standard zk-SNARKs with recursive zk-SNARKs based on Nova. The focus is on privacy-preserving and verifiable data transformations for smart-meter-like readings.

IoT readings are processed by an edge orchestrator that applies policies such as range checks and aggregation. Instead of sharing raw data, the system produces zero-knowledge proofs that allow a verifier to confirm correctness without learning private values.

## Main components

- ZoKrates circuits for policy checks and aggregation
- Python evaluation and analysis scripts for benchmarking and plotting
- A Dockerized execution environment for reproducible results

## Repository structure overview

The structure may vary slightly depending on your local setup.

- `circuits/`  
  ZoKrates circuits such as `filter_range.zok`
- `scripts/`  
  Evaluation and helper scripts including `run_in_docker.sh`
- `data/`  
  Input readings and intermediate artifacts
- `results/`  
  Exported JSON or CSV outputs
- `plots/`  
  Generated figures
- `Dockerfile`  
  Reproducible environment definition

## Quick start

1. Create and activate the environment

```bash
python3 -m venv iot_zk_env
source iot_zk_env/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Start Run with Docker

```bash
./scripts/run_in_docker.sh
```

You should run the script exactly like this. Adjust experiment settings directly inside the `run_in_docker.sh` file.

## Configuration inside run_in_docker.sh

The script exposes the most important experiment knobs at the top of the file.

### Reading counts

`READING_COUNTS` defines the input sizes used in the crossover measurements. For each value in the list, the system is evaluated three times, and the reported proving and verification times for that reading count are computed as the average of these runs to reduce variability in the measured results.

Edit this line in the script to change the evaluated input sizes.

```bash
READING_COUNTS=${1:-"200,400,600,800,1000"}
```

### Batch size

`BATCH_SIZE` controls how readings are grouped into batches for the evaluation and is passed into the measurement pipeline. The standard configuration uses a batch size of 1. To evaluate the efficiency impact of batching, we increased it to 20.

Edit this line in the script to change the batch size.

```bash
BATCH_SIZE=${2:-20}
```

### CPU and memory limits

To simulate edge-like devices, we use Docker to run the evaluation in a controlled environment with explicit resource limits. This allows us to approximate constrained compute and memory conditions and to compare standard and recursive proving performance under reproducible, edge-relevant settings.

Edit these flags in the script to change the resource profile.

```bash
  --cpus=4 \
  --memory=2g \
```
