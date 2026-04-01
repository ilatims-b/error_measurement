#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Prompt user for SEC EDGAR email
# echo "Please enter your email for the SEC EDGAR API:"
# read SEC_EMAIL

# # Prompt user for Grok/OpenAI API Key
# echo "Please enter your API Key for Fact Extraction (Grok/OpenAI compliant):"
# read -s API_KEY
# echo ""


# # Base API URL
# echo "Please enter the Model API Base URL (default: https://api.x.ai/v1):"
# read BASE_URL
# BASE_URL=${BASE_URL:-"https://api.x.ai/v1"}

# Extraction Model Name
# echo "Please enter the Extraction Model Name (default: grok-2-latest):"
# read MODEL_NAME
# MODEL_NAME=${MODEL_NAME:-"grok-2-latest"}

# Configuration
DATA_DIR="./data"
SCRIPT_DIR="./src"
DATASET_JSONL="${DATA_DIR}/processed_dataset.jsonl"
GENERATIONS_JSONL="${DATA_DIR}/generations.jsonl"
EVALUATED_JSONL="${DATA_DIR}/evaluated_generations.jsonl"
API_KEY=""
SEC_EMAIL=""
BASE_URL=""
MODEL_NAME="llama-3.3-70b-versatile"
mkdir -p "${DATA_DIR}"

# echo ""
# echo "=========================================="
# echo " Step 1: Downloading & Processing SEC Data"
# echo "=========================================="
# python "${SCRIPT_DIR}/dataset.py" \
#     --email "${SEC_EMAIL}" \
#     --download-dir "${DATA_DIR}/sec_filings" \
#     --output-file "${DATASET_JSONL}"

echo ""
echo "=========================================="
echo " Step 2: Generating Text Continuations"
echo "=========================================="
python "${SCRIPT_DIR}/generate.py" \
    --input-file "${DATASET_JSONL}" \
    --output-file "${GENERATIONS_JSONL}" \
    --model-name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --num-continuations 3

# echo ""
# echo "=========================================="
# echo " Step 3: Fact Extraction and Verification"
# echo "=========================================="
# python "${SCRIPT_DIR}/fact_pipeline.py" \
#     --gen-file "${GENERATIONS_JSONL}" \
#     --source-file "${DATASET_JSONL}" \
#     --output-file "${EVALUATED_JSONL}" \
#     --api-key "${API_KEY}" \
#     --base-url "${BASE_URL}" \
#     --model-name "${MODEL_NAME}"

# echo ""
# echo "=========================================="
# echo " Pipeline Execution Complete!"
# echo " Check ${EVALUATED_JSONL} for final results."
# echo "=========================================="
