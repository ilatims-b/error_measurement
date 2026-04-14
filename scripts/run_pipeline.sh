#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
export DATA_DIR="./data"
export SCRIPT_DIR="./src"
export DATASET_JSON="${DATA_DIR}/processed_dataset.json"
export GENERATIONS_JSON="${DATA_DIR}/generations.json"
export EVALUATED_JSON="${DATA_DIR}/evaluated_generations.json"

# User-specific configuration (override these as needed)
export SEC_EMAIL="ce24b119@smail.iitm.ac.in"
# export API_KEY=""
# export BASE_URL="https://api.groq.com/openai/v1"
# export MODEL_NAME="llama-3.3-70b-versatile"

# Ensure directories exist
mkdir -p "${DATA_DIR}"

# Step 1: Downloading & Processing SEC Data
./scripts/01_download.sh

# Step 2: Generating Text Continuations
./scripts/02_generate.sh

# Step 3: Fact Extraction and Verification
./scripts/03_evaluate.sh

echo ""
echo "=========================================="
echo " Pipeline Execution Complete!"
echo " Check ${EVALUATED_JSON} for final results."
echo "=========================================="
