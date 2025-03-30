#!/bin/bash
# export CUDA_DEVICE_MAX_CONNECTIONS=1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_IFNAME=ens12f0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=0
#export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
#export NCCL_IB_SL=3
#export NCCL_CHECKS_DISABLE=1
export NCCL_P2P_DISABLE=0
#export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_DEBUG=INFO

INDEX=0 # 表示节点索引（node_rank）为 0，因为只有一个节点
MASTER_ADDR="127.0.0.1" # 使用本地回环地址作为主节点地址，不需要外部 IP。
# communication on taiji platform
DISTRIBUTED_ARGS="
    --nproc_per_node 4 \
    --nnodes 1 \
    --node_rank $INDEX \
    --master_addr $MASTER_ADDR \
    --master_port 9999
"
export NCCL_TIMEOUT=25200
MODEL_TYPE=mixtral-8x7b
OUTPUT_DIR=$1
OUTPUT_DIR_FT=${OUTPUT_DIR}/llava-s3-finetune_task_lora_wo_complete_20_0.001_0.005_wo_CG
mkdir -p ${OUTPUT_DIR_FT}

torchrun $DISTRIBUTED_ARGS vita/train/train.py \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path /nfsdat/home/jmaslm/modelhub/VITA_ckpt \
    --model_type $MODEL_TYPE \
    --version mixtral_two \
    --dataset_use Pretrain_video \
    --vision_tower /nfsdat/home/jmaslm/modelhub/InternViT-300M-448px \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True\
    --audio_encoder /nfsdat/home/jmaslm/modelhub/audio-encoder-2wh_zh_en_audioset_Mixtral-8x7B_New-base-tunning \
    --freeze_audio_encoder True \
    --freeze_audio_encoder_adapter False \
    --image_aspect_ratio square \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR_FT} \
    --num_train_epochs 2 \
    --max_steps 600 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 9100 \
    --ddp_timeout 25200 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee -a ${OUTPUT_DIR_FT}/log_node_$INDEX.txt && echo "Done."



