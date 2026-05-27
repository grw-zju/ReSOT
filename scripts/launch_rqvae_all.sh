#!/bin/bash
# Launch all 6 RQ-VAE training jobs (3 datasets x 2 modalities) on 4 GPUs
# GPU allocation:
#   cuda:0: Instruments/llama, Games/llama (after Instruments completes)
#   cuda:1: Instruments/ViT-L-14
#   cuda:2: Arts/llama, Games/ViT-L-14 (after Arts completes)
#   cuda:3: Arts/ViT-L-14

cd /share/gengrenwu/ReSOT/index

COMMON_ARGS="--num_emb_list 256 256 256 256 \
  --sk_epsilons 0.0 0.0 0.0 0.01 \
  --eval_step 1 \
  --batch_size 1024 \
  --epochs 1000 \
  --early_stop_patience 50 \
  --diversity_loss_weight 0.1 \
  --e_dim 64 \
  --use_ot_loss 1 \
  --ot_loss_weight 0.25 \
  --badmm_iters 5 \
  --badmm_kl_penalty 0.1 \
  --fgw_alpha 0.25 \
  --use_uot 1 \
  --uot_tau_row 1.0 \
  --uot_tau_col 0.1 \
  --uot_eta 0.01 \
  --uot_iters 30 \
  --data_root ../data"

mkdir -p log/Instruments/llama_256 log/Instruments/ViT-L-14_256
mkdir -p log/Arts/llama_256 log/Arts/ViT-L-14_256
mkdir -p log/Games/llama_256 log/Games/ViT-L-14_256

echo "=== Launching 6 RQ-VAE training jobs ==="

# GPU 0: Instruments llama + Games llama (sequential)
echo "Starting Instruments/llama on cuda:0"
nohup python -u main_mul.py ${COMMON_ARGS} \
  --device cuda:0 \
  --embedding_file .emb-llama-td.npy \
  --datasets Instruments \
  --ckpt_dir log/Instruments/llama_256 \
  > log/Instruments/llama_256/train.log 2>&1 &
PID_INST_LLAMA=$!
echo "Instruments llama PID: ${PID_INST_LLAMA}"

# GPU 1: Instruments ViT-L-14
echo "Starting Instruments/ViT-L-14 on cuda:1"
nohup python -u main_mul.py ${COMMON_ARGS} \
  --device cuda:1 \
  --embedding_file .emb-ViT-L-14.npy \
  --datasets Instruments \
  --ckpt_dir log/Instruments/ViT-L-14_256 \
  > log/Instruments/ViT-L-14_256/train.log 2>&1 &
PID_INST_VIT=$!
echo "Instruments ViT PID: ${PID_INST_VIT}"

# GPU 2: Arts llama + Games ViT-L-14 (sequential)
echo "Starting Arts/llama on cuda:2"
nohup python -u main_mul.py ${COMMON_ARGS} \
  --device cuda:2 \
  --embedding_file .emb-llama-td.npy \
  --datasets Arts \
  --ckpt_dir log/Arts/llama_256 \
  > log/Arts/llama_256/train.log 2>&1 &
PID_ARTS_LLAMA=$!
echo "Arts llama PID: ${PID_ARTS_LLAMA}"

# GPU 3: Arts ViT-L-14
echo "Starting Arts/ViT-L-14 on cuda:3"
nohup python -u main_mul.py ${COMMON_ARGS} \
  --device cuda:3 \
  --embedding_file .emb-ViT-L-14.npy \
  --datasets Arts \
  --ckpt_dir log/Arts/ViT-L-14_256 \
  > log/Arts/ViT-L-14_256/train.log 2>&1 &
PID_ARTS_VIT=$!
echo "Arts ViT PID: ${PID_ARTS_VIT}"

echo ""
echo "=== First batch launched (4 jobs on 4 GPUs) ==="
echo "Games jobs will be launched after Instruments/Arts complete on shared GPUs."
echo ""
echo "Monitor progress with: tail -f index/log/*/train.log"
echo ""
echo "To launch Games after first batch, run: scripts/launch_rqvae_games.sh"
