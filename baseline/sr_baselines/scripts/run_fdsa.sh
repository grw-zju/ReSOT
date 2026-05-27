#!/bin/bash
# FDSA: Feature-level Deep Sequential Attention
# Models item-level and feature-level transitions with separate self-attention blocks

DATASET=${1:-"Instruments"}
DATA_PATH=${2:-""}
DEVICE=${3:-"cuda:0"}
CKPT_DIR="checkpoints/fdsa/${DATASET}"

mkdir -p ${CKPT_DIR}

python baseline/sr_baselines/train_sr.py \
    --model fdsa \
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
    --feature_dim 64 \
    --ckpt_dir ${CKPT_DIR}

echo "FDSA training completed for ${DATASET}"
