#!/bin/bash
# P5-CID: Contextual Item Description based ID generation
# Uses contextual item descriptions for ID assignment

DATASET=${1:-"Instruments"}
DATA_PATH=${2:-""}
OUTPUT_DIR=${3:-""}

python baseline/text_gr/generate_text_ids.py \
    --method p5cid \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --token_length 4 \
    --vocab_size 256

echo "P5-CID ID generation completed for ${DATASET}"
