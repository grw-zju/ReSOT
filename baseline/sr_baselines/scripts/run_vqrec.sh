#!/bin/bash
# VQ-Rec: Vector Quantized Recommendation
# Maps item text into VQ discrete codes for cross-domain generalization

DATASET=${1:-"Instruments"}
DATA_PATH=${2:-""}
DEVICE=${3:-"cuda:0"}
CKPT_DIR="checkpoints/vqrec/${DATASET}"

mkdir -p ${CKPT_DIR}

python baseline/sr_baselines/train_sr.py \
    --model vqrec \
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
    --codebook_size 256 \
    --num_codebooks 4 \
    --code_dim 64 \
    --ckpt_dir ${CKPT_DIR}

echo "VQ-Rec training completed for ${DATASET}"
