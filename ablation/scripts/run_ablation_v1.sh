#!/bin/bash
# ReSOT Ablation V1: No GWOT, No UOT, No SQ (等同TIGER)
# 对应论文Table 2第一行: 纯RQ-VAE，无任何ReSOT组件
DATASET=${1:-"Instruments"}
DATA_ROOT=${2:-""}
DEVICE=${3:-"cuda:0"}
CKPT_DIR="checkpoints/resot_ablation/v1_${DATASET}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p ${CKPT_DIR}

python train_ablation.py \
    --data_root ${DATA_ROOT} \
    --datasets ${DATASET} \
    --embedding_file .emb-llama-td.npy \
    --use_gwot 0 \
    --use_uot 0 \
    --use_sq 0 \
    --sk_epsilons 0 0 0 0 \
    --ckpt_dir ${CKPT_DIR} \
    --epochs 5000 \
    --batch_size 1024 \
    --e_dim 64 \
    --num_emb_list 256 256 256 256 \
    --lr 1e-3 \
    --device ${DEVICE}

python generate_ablation_index.py \
    --dataset ${DATASET} \
    --ckpt_path ${CKPT_DIR}/best_collision_model.pth \
    --output_dir ${DATA_ROOT}/${DATASET} \
    --output_file "${DATASET}.ablation_v1_indices.json" \
    --content text \
    --embedding_file .emb-llama-td.npy \
    --data_root ${DATA_ROOT} \
    --use_gwot 0 \
    --use_uot 0 \
    --use_sq 0 \
    --device ${DEVICE}
