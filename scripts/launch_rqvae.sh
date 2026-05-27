#!/bin/bash
# Launch all 6 RQ-VAE training jobs (3 datasets x 2 modalities) on 4 GPUs

cd /share/gengrenwu/ReSOT/index

COMMON_ARGS="--num_emb_list 256 256 256 256 \
  --sk_epsilons 0.0 0.0 0.0 0.01 \
  --eval_step 1 \
  --batch_size 1024 \
  --epochs 1000 \
  --early_stop_patience 50 \
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

echo "Starting Instruments/llama on cuda:0"
nohup python -u main_mul.py ${COMMON_ARGS} \
  --device cuda:0 \
  --embedding_file .emb-llama-td.npy \
  --datasets Instruments \
  --ckpt_dir log/Instruments/llama_256 \
  > log/Instruments/llama_256/train.log 2>&1 &
echo "PID: $!"

echo "Starting Instruments/ViT on cuda:1"
nohup python -u main_mul.py ${COMMON_ARGS} \
  --device cuda:1 \
  --embedding_file .emb-ViT-L-14.npy \
  --datasets Instruments \
  --ckpt_dir log/Instruments/ViT-L-14_256 \
  > log/Instruments/ViT-L-14_256/train.log 2>&1 &
echo "PID: $!"

echo "Starting Arts/llama on cuda:2"
nohup python -u main_mul.py ${COMMON_ARGS} \
  --device cuda:2 \
  --embedding_file .emb-llama-td.npy \
  --datasets Arts \
  --ckpt_dir log/Arts/llama_256 \
  > log/Arts/llama_256/train.log 2>&1 &
echo "PID: $!"

echo "Starting Arts/ViT on cuda:3"
nohup python -u main_mul.py ${COMMON_ARGS} \
  --device cuda:3 \
  --embedding_file .emb-ViT-L-14.npy \
  --datasets Arts \
  --ckpt_dir log/Arts/ViT-L-14_256 \
  > log/Arts/ViT-L-14_256/train.log 2>&1 &
echo "PID: $!"

echo "All 4 initial RQ-VAE jobs launched. Games will be queued after completion."
echo "Monitor with: tail -f index/log/*/train.log"
