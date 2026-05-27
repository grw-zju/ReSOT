#!/bin/bash
# ReSOT Ablation V4: GWOT+Sinkhorn SQ (No UOT)
# 对应论文Table 2第四行: GWOT结构感知重建+Sinkhorn软量化，无UOT
DATASET=${1:-"Instruments"}
DATA_ROOT=${2:-""}
DEVICE=${3:-"cuda:0"}
CKPT_DIR="checkpoints/resot_ablation/v4_${DATASET}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p ${CKPT_DIR}

python train_ablation.py \
    --data_root ${DATA_ROOT} \
    --datasets ${DATASET} \
    --embedding_file .emb-llama-td.npy \
    --use_gwot 1 \
    --use_uot 0 \
    --use_sq 1 \
    --sk_epsilons 0 0 0 0.01 \
    --ckpt_dir ${CKPT_DIR} \
    --epochs 5000 \
    --batch_size 1024 \
    --e_dim 64 \
    --num_emb_list 256 256 256 256 \
    --lr 1e-3 \
    --badmm_iters 5 \
    --badmm_kl_penalty 0.1 \
    --fgw_alpha 0.25 \
    --device ${DEVICE}

python generate_ablation_index.py \
    --dataset ${DATASET} \
    --ckpt_path ${CKPT_DIR}/best_collision_model.pth \
    --output_dir ${DATA_ROOT}/${DATASET} \
    --output_file "${DATASET}.ablation_v4_indices.json" \
    --content text \
    --embedding_file .emb-llama-td.npy \
    --data_root ${DATA_ROOT} \
    --use_gwot 1 \
    --use_uot 0 \
    --use_sq 1 \
    --device ${DEVICE}
