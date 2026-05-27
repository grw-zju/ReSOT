#!/bin/bash
# ReSOT Full Pipeline
# (1) ontology distance preprocessing
# (2) RQ-VAE index training
# (3) index generation
# (4) T5 fine-tuning
# (5) evaluation

set -e

DATASET=${1:-"Instruments"}
DATA_ROOT=${2:-"./data"}
DEVICE=${3:-"cuda:0"}
RUN_BASELINES=${4:-"0"}

CKPT_BASE="checkpoints"

echo "========================================="
echo "  ReSOT Full Pipeline"
echo "  Dataset: ${DATASET}"
echo "========================================="

# ==========================================
# Step 1: Train ReSOT RQ-VAE Index
# ==========================================
echo "[Step 1] Training ReSOT RQ-VAE..."
mkdir -p ${CKPT_BASE}/resot/${DATASET}

python -u index/main_mul.py \
    --datasets ${DATASET} \
    --data_root ${DATA_ROOT} \
    --device ${DEVICE} \
    --lr 1e-3 \
    --epochs 5000 \
    --batch_size 1024 \
    --eval_step 50 \
    --learner AdamW \
    --weight_decay 1e-4 \
    --num_emb_list 256 256 256 256 \
    --e_dim 64 \
    --layers 2048 1024 512 256 128 64 \
    --quant_loss_weight 1.0 \
    --kmeans_init 1 \
    --kmeans_iters 100 \
    --sk_epsilons 0.0 0.0 0.0 0.01 \
    --sk_iters 50 \
    --embedding_file .emb-llama-td.npy \
    --ckpt_dir ${CKPT_BASE}/resot/${DATASET} \
    --use_ot_loss 1 \
    --ot_loss_weight 0.25 \
    --ot_eps 0.05 \
    --badmm_iters 5 \
    --badmm_kl_penalty 0.1 \
    --fgw_alpha 0.25 \
    --use_mirror_grad 1 \
    --mirror_grad_iters 30 \
    --mirror_grad_step 0.01

echo "[Step 1] ReSOT RQ-VAE training completed."

# ==========================================
# Step 2: Generate ReSOT Indices
# ==========================================
echo "[Step 2] Generating ReSOT indices..."
RESOT_CKPT="${CKPT_BASE}/resot/${DATASET}/best_collision_model.pth"
OUTPUT_DIR="${DATA_ROOT}/${DATASET}"

python index/generate_indices_distance_mul.py \
    --datasets ${DATASET} \
    --ckpt_path ${RESOT_CKPT} \
    --data_root ${DATA_ROOT} \
    --embedding_file .emb-llama-td.npy \
    --output_file .index_lemb.json \
    --content text \
    --device ${DEVICE}

python index/generate_indices_distance_mul.py \
    --datasets ${DATASET} \
    --ckpt_path ${RESOT_CKPT} \
    --data_root ${DATA_ROOT} \
    --embedding_file .emb-ViT-L-14.npy \
    --output_file .index_vitemb.json \
    --content image \
    --device ${DEVICE}

echo "[Step 2] ReSOT index generation completed."

# ==========================================
# Step 3: T5 Fine-tuning + Evaluation
# ==========================================
echo "[Step 3] Fine-tuning T5 model..."
T5_OUTPUT="${CKPT_BASE}/resot_t5/${DATASET}"
mkdir -p ${T5_OUTPUT}

python finetune.py \
    --dataset ${DATASET} \
    --data_path ${DATA_ROOT} \
    --base_model t5-base \
    --tasks seqrec \
    --index_file .index_lemb.json \
    --image_index_file .index_vitemb.json \
    --ckpt_path ${T5_OUTPUT} \
    --output_dir ${T5_OUTPUT} \
    --device ${DEVICE} \
    --epochs 10 \
    --per_device_batch_size 64 \
    --learning_rate 1e-3 \
    --num_beams 20

echo "[Step 3] T5 fine-tuning completed."

echo "[Step 4] Evaluating..."
python test.py \
    --dataset ${DATASET} \
    --data_path ${DATA_ROOT} \
    --ckpt_path ${T5_OUTPUT} \
    --tasks seqrec \
    --index_file .index_lemb.json \
    --image_index_file .index_vitemb.json \
    --metrics hit@1,hit@5,hit@10,ndcg@5,ndcg@10 \
    --num_beams 20 \
    --device ${DEVICE}

echo "[Step 4] Evaluation completed."

# ==========================================
# Step 5: Run Baselines (optional)
# ==========================================
if [ "${RUN_BASELINES}" == "1" ]; then
    echo "[Step 5] Running all baselines..."

    for BL in tiger lcrec letter mmesid mql4grec macrec; do
        echo "  Training ${BL} index..."
        bash baseline/gr_index/scripts/run_${BL}.sh ${DATASET} ${DATA_ROOT} ${DEVICE}
    done

    echo "  Generating P5-CID / VIP5 IDs..."
    bash baseline/text_gr/scripts/run_p5cid.sh ${DATASET} ${DATA_ROOT} ${OUTPUT_DIR}
    bash baseline/text_gr/scripts/run_vip5.sh ${DATASET} ${DATA_ROOT} ${OUTPUT_DIR}

    for BL in sasrec bert4rec fdsa s3rec vqrec missrec; do
        echo "  Training ${BL}..."
        bash baseline/sr_baselines/scripts/run_${BL}.sh ${DATASET} ${DATA_ROOT} ${DEVICE}
    done

    for V in v1 v2 v3 v4 v5 v6; do
        echo "  Running ablation ${V}..."
        bash ablation/scripts/run_ablation_${V}.sh ${DATASET} ${DATA_ROOT} ${DEVICE}
    done

    echo "[Step 5] All baselines completed."
fi

echo "========================================="
echo "  Pipeline completed for ${DATASET}!"
echo "========================================="
