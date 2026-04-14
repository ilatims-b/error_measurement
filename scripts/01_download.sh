#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
SEC_EMAIL="${SEC_EMAIL:-ce24b119@smail.iitm.ac.in}"
TICKERS="${TICKERS:-AAPL}"
DOC_TYPES="${DOC_TYPES:-10-Q}"
DATA_DIR="${DATA_DIR:-./data}"
SCRIPT_DIR="${SCRIPT_DIR:-./src}"
DATASET_JSON="${DATASET_JSON:-${DATA_DIR}/processed_dataset.json}"

echo "=========================================="
echo " Step 1: Downloading & Processing SEC Data"
echo " Tickers:   ${TICKERS}"
echo " Doc Types: ${DOC_TYPES}"
echo "=========================================="

python "${SCRIPT_DIR}/dataset.py" \
    --email "${SEC_EMAIL}" \
    --tickers ${TICKERS} \
    --doc-types ${DOC_TYPES} \
    --download-dir "${DATA_DIR}/sec_filings" \
    --output-file "${DATASET_JSON}" \
    --seed-tokens 200
