#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
DATA_DIR="${DATA_DIR:-./data}"
SCRIPT_DIR="${SCRIPT_DIR:-./src}"
DATASET_JSONL="${DATASET_JSONL:-${DATA_DIR}/processed_dataset.jsonl}"
GENERATIONS_JSONL="${GENERATIONS_JSONL:-${DATA_DIR}/generations.jsonl}"
EVALUATED_JSONL="${EVALUATED_JSONL:-${DATA_DIR}/evaluated_generations.jsonl}"

# API settings
API_KEY="${API_KEY}"
BASE_URL="${BASE_URL:-https://api.groq.com/openai/v1}"
MODEL_NAME="${MODEL_NAME:-llama-3.3-70b-versatile}"

# Evaluation settings
CHUNK_SIZE="${CHUNK_SIZE:-128}"
EXTRACTION_PROMPT="${EXTRACTION_PROMPT}"
VERIFICATION_PROMPT="${VERIFICATION_PROMPT}"

echo "=========================================="
echo " Step 3: Fact Extraction and Verification"
echo "=========================================="

python "${SCRIPT_DIR}/fact_pipeline.py" \
    --gen-file "${GENERATIONS_JSONL}" \
    --source-file "${DATASET_JSONL}" \
    --output-file "${EVALUATED_JSONL}" \
    --api-key "${API_KEY}" \
    --base-url "${BASE_URL}" \
    --model-name "${MODEL_NAME}" \
    --chunk-size "${CHUNK_SIZE}" \
    ${EXTRACTION_PROMPT:+--extraction-prompt "$EXTRACTION_PROMPT"} \
    ${VERIFICATION_PROMPT:+--verification-prompt "$VERIFICATION_PROMPT"}
