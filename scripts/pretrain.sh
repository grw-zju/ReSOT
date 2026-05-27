
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1

Base_model=ckpt
Per_device_batch_size=1024
Learning_rate=1e-3
Epoch=50

Index_file=.index_lemb.json
Image_index_file=.index_vitemb.json

Tasks=seqrec,seqimage
Valid_task=seqrec

Datasets='Pet,Cell,Automotive,Tools,Toys,Sports'

OUTPUT_DIR=./log/$Datasets/${Base_model}_b${Per_device_batch_size}_lr${Learning_rate}_${Tasks}/pretrain
mkdir -p $OUTPUT_DIR
log_file=$OUTPUT_DIR/pretrain.log

torchrun --nproc_per_node=4 --master_port=2309 pretrain.py \
    --data_path ./data/ \
    --pretrain_datasets $Datasets \
    --output_dir $OUTPUT_DIR \
    --base_model ./config/$Base_model \
    --per_device_batch_size $Per_device_batch_size \
    --learning_rate $Learning_rate \
    --epochs $Epoch \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --logging_step 50 \
    --train_data_mode 0 \
    --max_his_len 20 \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --tasks $Tasks \
    --valid_task $Valid_task > $log_file

