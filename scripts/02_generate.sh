#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
DATA_DIR="${DATA_DIR:-./data}"
SCRIPT_DIR="${SCRIPT_DIR:-./src}"
DATASET_JSONL="${DATASET_JSONL:-${DATA_DIR}/processed_dataset.jsonl}"
GENERATIONS_JSONL="${GENERATIONS_JSONL:-${DATA_DIR}/generations.jsonl}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
NUM_CONTINUATIONS="${NUM_CONTINUATIONS:-3}"

echo "=========================================="
echo " Step 2: Generating Text Continuations"
echo "=========================================="

python "${SCRIPT_DIR}/generate.py" \
    --input-file "${DATASET_JSONL}" \
    --output-file "${GENERATIONS_JSONL}" \
    --model-name "${MODEL_NAME}" \
    --num-continuations "${NUM_CONTINUATIONS}"
