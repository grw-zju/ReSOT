#!/bin/bash
# S3-Rec: Self-Supervised Sequential Recommendation
# Uses mutual information maximization for pre-training

DATASET=${1:-"Instruments"}
DATA_PATH=${2:-""}
DEVICE=${3:-"cuda:0"}
CKPT_DIR="checkpoints/s3rec/${DATASET}"

mkdir -p ${CKPT_DIR}

python baseline/sr_baselines/train_sr.py \
    --model s3rec \
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

echo "S3-Rec training completed for ${DATASET}"
