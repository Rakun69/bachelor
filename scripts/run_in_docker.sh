set -euo pipefail

READING_COUNTS=${1:-"100,200,300,400,500,600,700,800,900,1000"}
BATCH_SIZE=${2:-20}

NOVA_COMPRESS=${3:-true}

MAX_COUNT=$(echo $READING_COUNTS | tr ',' '\n' | sort -n | tail -1)

IMAGE_NAME=iot-zk-snark-eval

HOST_PROJECT_DIR="${HOST_PROJECT_DIR:-$(pwd)}"
CONTAINER_PROJECT_DIR="/app"

echo "[INFO] Building Docker image ${IMAGE_NAME}..."
docker build --no-cache -t ${IMAGE_NAME} .

echo "[INFO] Running container.."
docker run --rm \
  --cpus=4 \
  --memory=2g \
  -v "${HOST_PROJECT_DIR}:${CONTAINER_PROJECT_DIR}" \
  -w "${CONTAINER_PROJECT_DIR}" \
  -e NUM_READINGS=${MAX_COUNT} \
  ${IMAGE_NAME} \
  bash -c " \
    # Clean old caches
    find . -name '*.pyc' -delete && \
    find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true && \

    # Zeige NumPy Version
    python3 -c \"import numpy; print('NumPy version:', numpy.__version__)\" && \

    # Device Keys + Dummy Readings generieren
    python3 scripts/setup_device_and_dummy_data.py && \

    # Proof-Messungen starten
    python3 scripts/measure_crossover_real.py \
      --reading-counts ${READING_COUNTS} \
      --warmup-runs 1 \
      --repetitions 3 \
      --mode warm \
      --batch-size ${BATCH_SIZE} \
      $( [ "${NOVA_COMPRESS}" = "true" ] && echo "--nova-compress" ) \
      \
  "

echo "[INFO] Done."