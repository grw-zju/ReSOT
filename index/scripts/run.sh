Model=llama
Code_num=256
Datasets=Instruments

OUTPUT_DIR=log/$Datasets/${Model}_${Code_num}
mkdir -p $OUTPUT_DIR

python -u main_mul.py \
  --num_emb_list $Code_num $Code_num $Code_num $Code_num \
  --sk_epsilons 0.0 0.0 0.0 0.01 \
  --device cuda:0 \
  --data_root ../data \
  --embedding_file .emb-llama-td.npy \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR \
  --eval_step 50 \
  --batch_size 1024 \
  --epochs 5000 \
  --e_dim 64 \
  --use_ot_loss 1 \
  --ot_loss_weight 0.25 \
  --badmm_iters 5 \
  --badmm_kl_penalty 0.1 \
  --fgw_alpha 0.25 \
  --use_mirror_grad 1 \
  --mirror_grad_iters 30 \
  --mirror_grad_step 0.01 \
  > $OUTPUT_DIR/train.log

Model=ViT-L-14

OUTPUT_DIR=log/$Datasets/${Model}_${Code_num}
mkdir -p $OUTPUT_DIR

python -u main_mul.py \
  --num_emb_list $Code_num $Code_num $Code_num $Code_num \
  --sk_epsilons 0.0 0.0 0.0 0.01 \
  --device cuda:0 \
  --data_root ../data \
  --embedding_file .emb-ViT-L-14.npy \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR \
  --eval_step 50 \
  --batch_size 1024 \
  --epochs 5000 \
  --e_dim 64 \
  --use_ot_loss 1 \
  --ot_loss_weight 0.25 \
  --badmm_iters 5 \
  --badmm_kl_penalty 0.1 \
  --fgw_alpha 0.25 \
  --use_mirror_grad 1 \
  --mirror_grad_iters 30 \
  --mirror_grad_step 0.01 \
  > $OUTPUT_DIR/train.log
