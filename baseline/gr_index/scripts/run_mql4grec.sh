#!/bin/bash
# MQL4GRec: multi-quantization layer for generative recommendation
# Translates multimodal content into unified quantitative language

DATASET=${1:-"Instruments"}
DATA_ROOT=${2:-""}
DEVICE=${3:-"cuda:0"}
CKPT_DIR="checkpoints/mql4grec/${DATASET}"

mkdir -p ${CKPT_DIR}

python baseline/gr_index/train_index.py \
    --baseline mql4grec \
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

echo "MQL4GRec index training completed for ${DATASET}"
