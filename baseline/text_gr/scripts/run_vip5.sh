#!/bin/bash
# VIP5: Vision-language-personalization based ID generation
# Uses random ID assignment (no semantic tokenizer)

DATASET=${1:-"Instruments"}
DATA_PATH=${2:-""}
OUTPUT_DIR=${3:-""}

python baseline/text_gr/generate_text_ids.py \
    --method vip5 \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --token_length 4 \
    --vocab_size 256

echo "VIP5 ID generation completed for ${DATASET}"
