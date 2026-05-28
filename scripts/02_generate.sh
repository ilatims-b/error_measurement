#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# ---------------------------------------------------------------------------
# Configuration — local HuggingFace path
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-./data}"
SCRIPT_DIR="${SCRIPT_DIR:-./src}"
DATASET_JSON="${DATASET_JSON:-${DATA_DIR}/processed_dataset.json}"
GENERATIONS_JSON="${GENERATIONS_JSON:-${DATA_DIR}/generations.json}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
NUM_CONTINUATIONS="${NUM_CONTINUATIONS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"

# ---------------------------------------------------------------------------
# Configuration — Grok / OpenAI-compatible API path
# Set GROK_API_KEY to activate the API backend instead of local inference.
# ---------------------------------------------------------------------------
GROK_API_KEY="${GROK_API_KEY:-}"
GROK_BASE_URL="${GROK_BASE_URL:-https://api.x.ai/v1}"
GROK_MODEL_NAME="${GROK_MODEL_NAME:-grok-3-mini}"
GROK_RPM="${GROK_RPM:-60}"      # Requests per minute limit
GROK_TPM="${GROK_TPM:-100000}"  # Tokens per minute limit

echo "=========================================="
echo " Step 2: Generating Text Continuations"
echo "=========================================="

if [ -n "${GROK_API_KEY}" ]; then
    echo " Backend : Grok API (${GROK_BASE_URL})"
    echo " Model   : ${GROK_MODEL_NAME}"
    echo " RPM     : ${GROK_RPM}   TPM: ${GROK_TPM}"
    echo "=========================================="

    python "${SCRIPT_DIR}/generate.py" \
        --input-file    "${DATASET_JSON}" \
        --output-file   "${GENERATIONS_JSON}" \
        --model-name    "${GROK_MODEL_NAME}" \
        --num-continuations "${NUM_CONTINUATIONS}" \
        --max-new-tokens    "${MAX_NEW_TOKENS}" \
        --temperature       "${TEMPERATURE}" \
        --top-p             "${TOP_P}" \
        --api-key       "${GROK_API_KEY}" \
        --base-url      "${GROK_BASE_URL}" \
        --rpm           "${GROK_RPM}" \
        --tpm           "${GROK_TPM}"
else
    echo " Backend : Local HuggingFace model"
    echo " Model   : ${MODEL_NAME}"
    echo "=========================================="

    python "${SCRIPT_DIR}/generate.py" \
        --input-file        "${DATASET_JSON}" \
        --output-file       "${GENERATIONS_JSON}" \
        --model-name        "${MODEL_NAME}" \
        --num-continuations "${NUM_CONTINUATIONS}" \
        --max-new-tokens    "${MAX_NEW_TOKENS}" \
        --temperature       "${TEMPERATURE}" \
        --top-p             "${TOP_P}"
fi
