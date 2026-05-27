#!/bin/bash
# BERT4Rec: Bidirectional Encoder Representations from Transformers for Sequential Recommendation
# Uses bidirectional attention with Cloze-style masking

DATASET=${1:-"Instruments"}
DATA_PATH=${2:-""}
DEVICE=${3:-"cuda:0"}
CKPT_DIR="checkpoints/bert4rec/${DATASET}"

mkdir -p ${CKPT_DIR}

python baseline/sr_baselines/train_sr.py \
    --model bert4rec \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --device ${DEVICE} \
    --hidden_size 64 \
    --max_seq_len 50 \
    --num_heads 2 \
    --num_blocks 2 \
    --dropout 0.5 \
    --lr 1e-3 \
    --epochs 200 \
    --batch_size 256 \
    --ckpt_dir ${CKPT_DIR}

echo "BERT4Rec training completed for ${DATASET}"
