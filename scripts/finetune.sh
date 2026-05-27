
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1

Index_file=.index_lemb.json
Image_index_file=.index_vitemb.json

Tasks='seqrec,seqimage,item2image,image2item,fusionseqrec'
Valid_task=seqrec

Datasets='Instruments'

load_model_name=pretrain_ckpt_path

OUTPUT_DIR=./log/$Datasets
mkdir -p $OUTPUT_DIR
log_file=$OUTPUT_DIR/train.log

torchrun --nproc_per_node=2 --master_port=2309 finetune.py \
    --data_path ./data/ \
    --dataset $Datasets \
    --output_dir $OUTPUT_DIR \
    --load_model_name $load_model_name \
    --per_device_batch_size 256 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --logging_step 50 \
    --max_his_len 20 \
    --prompt_num 4 \
    --patient 10 \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --tasks $Tasks \
    --valid_task $Valid_task > $log_file

results_file=$OUTPUT_DIR/results_${Valid_task}_${20}.json
save_file=$OUTPUT_DIR/save_${Valid_task}_${20}.json

torchrun --nproc_per_node=2 --master_port=2309 test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path ./data/ \
    --dataset $Datasets \
    --test_batch_size 64 \
    --num_beams 20 \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --test_task $Valid_task \
    --results_file $results_file \
    --save_file $save_file \
    --filter_items > $log_file

python ensemble.py \
    --output_dir $OUTPUT_DIR\
    --dataset $Datasets\
    --data_path ./data/\
    --index_file $Index_file\
    --image_index_file $Image_index_file\
    --num_beams 20

