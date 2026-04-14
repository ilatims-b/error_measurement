#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
DATA_DIR="${DATA_DIR:-./data}"
SCRIPT_DIR="${SCRIPT_DIR:-./src}"
DATASET_JSON="${DATASET_JSON:-${DATA_DIR}/processed_dataset.json}"
GENERATIONS_JSON="${GENERATIONS_JSON:-${DATA_DIR}/generations.json}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
NUM_CONTINUATIONS="${NUM_CONTINUATIONS:-3}"

echo "=========================================="
echo " Step 2: Generating Text Continuations"
echo "=========================================="

python "${SCRIPT_DIR}/generate.py" \
    --input-file "${DATASET_JSON}" \
    --output-file "${GENERATIONS_JSON}" \
    --model-name "${MODEL_NAME}" \
    --num-continuations "${NUM_CONTINUATIONS}"
