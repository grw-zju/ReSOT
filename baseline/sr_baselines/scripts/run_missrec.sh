#!/bin/bash
# MISSRec: Multimodal Interest-aware Sequential Recommendation
# Pre-trains interest-aware Transformer on multimodal content

DATASET=${1:-"Instruments"}
DATA_PATH=${2:-""}
DEVICE=${3:-"cuda:0"}
CKPT_DIR="checkpoints/missrec/${DATASET}"

mkdir -p ${CKPT_DIR}

python baseline/sr_baselines/train_sr.py \
    --model missrec \
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
    --text_dim 4096 \
    --image_dim 768 \
    --ckpt_dir ${CKPT_DIR}

echo "MISSRec training completed for ${DATASET}"
