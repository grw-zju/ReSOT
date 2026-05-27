#!/bin/bash
# TIGER: standard RQ-VAE without distance-aware codebook update
# Uses same RQ-VAE architecture as ReSOT but disables OT-based components

DATASET=${1:-"Instruments"}
DATA_ROOT=${2:-""}
DEVICE=${3:-"cuda:0"}
CKPT_DIR="checkpoints/tiger/${DATASET}"

mkdir -p ${CKPT_DIR}

python baseline/gr_index/train_index.py \
    --baseline tiger \
    --datasets ${DATASET} \
    --data_root ${DATA_ROOT} \
    --device ${DEVICE} \
    --lr 1e-3 \
    --epochs 5000 \
    --batch_size 1024 \
    --eval_step 50 \
    --num_emb_list 256 256 256 256 \
    --e_dim 64 \
    --layers 2048 1024 512 256 128 64 \
    --quant_loss_weight 1.0 \
    --weight_decay 1e-4 \
    --kmeans_init 1 \
    --kmeans_iters 100 \
    --sk_iters 50 \
    --ckpt_dir ${CKPT_DIR}

echo "TIGER index training completed for ${DATASET}"
