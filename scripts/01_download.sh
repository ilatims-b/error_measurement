#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
SEC_EMAIL="${SEC_EMAIL:-ce24b119@smail.iitm.ac.in}"
DATA_DIR="${DATA_DIR:-./data}"
SCRIPT_DIR="${SCRIPT_DIR:-./src}"
DATASET_JSONL="${DATASET_JSONL:-${DATA_DIR}/processed_dataset.jsonl}"

echo "=========================================="
echo " Step 1: Downloading & Processing SEC Data"
echo "=========================================="

python "${SCRIPT_DIR}/dataset.py" \
    --email "${SEC_EMAIL}" \
    --download-dir "${DATA_DIR}/sec_filings" \
    --output-file "${DATASET_JSONL}" \
    --seed-tokens 200
