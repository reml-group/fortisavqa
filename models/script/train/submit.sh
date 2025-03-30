#!/bin/bash
#SBATCH --job-name=train-nn-gpu                  # 作业名
#SBATCH --time=50:00:00                          # 预估时间限制
#SBATCH --partition=gpu                          # 使用GPU分区
#SBATCH --gres=gpu:A800:4                        # 请求4个GPU（根据需求调整）
#SBATCH --mem=120G                               # 请求64GB内存（根据需求调整）
#SBATCH --cpus-per-task=128                      # 每个任务8个CPU核心
#SBATCH --nodes=1                                # 使用1个节点
#SBATCH --ntasks=1                               # 1个任务
#SBATCH --output=./outputs/%x-%j.out             # 输出日志路径
#SBATCH --error=./outputs/%x-%j.err              # 错误日志路径
#SBATCH --nodelist=GPU136

# 打印一些信息
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# actiavte the virtual environment

# export PYTHONPATH=./
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# OUTPUT_DIR=/nfsdat/home/jmaslm/code/VITA
# bash script/train/finetuneTask_single_node.sh ${OUTPUT_DIR}		

# mv /nfsdat/home/jmaslm/code/VITA/llava-s3-finetune_task_lora_wo_complete_20_0.001_0.005_wo_CG_A100/checkpoint-600/tokenizer* /nfsdat/home/jmaslm/code/VITA/llava-s3-finetune_task_lora_wo_complete_20_0.001_0.005_wo_CG/checkpoint-600/

# CUDA_VISIBLE_DEVICES=0,1,2,3 python avqa_test.py \
#        --model_base	/nfsdat/home/jmaslm/modelhub/VITA_ckpt \
#        --test_json	/nfsdat/home/jmaslm/datahub/MUSIC_AVQA/avqa-test-10%.json  \
#        --output_path /nfsdat/home/jmaslm/code/VITA/test_output/avqa_wo_complete_20_0.001_0.005_wo_CG_0326.json \
#        --model_path /nfsdat/home/jmaslm/code/VITA/llava-s3-finetune_task_lora_wo_complete_20_0.001_0.005_wo_CG/checkpoint-400 \
#        --video_dir /nfsdat/home/jmaslm/datahub/MUSIC_AVQA_R/video \
#        --audio_dir /nfsdat/home/jmaslm/datahub/MUSIC_AVQA_R/audio

# CUDA_VISIBLE_DEVICES=0,1,2,3 python avqa_test.py \
#        --model_base	/nfsdat/home/jmaslm/modelhub/VITA_ckpt \
#        --test_json	/nfsdat/home/jmaslm/datahub/testset_split_new_v2/avqa-test-headtail-newsplit-1%.json  \
#        --output_path /nfsdat/home/jmaslm/code/VITA/test_output/avqar_wo_complete_20_0.001_0.005_wo_CG_0326.json \
#        --model_path /nfsdat/home/jmaslm/code/VITA/llava-s3-finetune_task_lora_wo_complete_20_0.001_0.005_wo_CG/checkpoint-400 \
#        --video_dir /nfsdat/home/jmaslm/datahub/MUSIC_AVQA_R/video \
#        --audio_dir /nfsdat/home/jmaslm/datahub/MUSIC_AVQA_R/audio

CUDA_VISIBLE_DEVICES=0,1,2,3 python avqa_test.py \
       --model_base	/nfsdat/home/jmaslm/modelhub/VITA_ckpt \
       --test_json	/nfsdat/home/jmaslm/datahub/test_case.json  \
       --output_path /nfsdat/home/jmaslm/code/VITA/test_output/test_case_output.json \
       --model_path /nfsdat/home/jmaslm/code/VITA/llava-s3-finetune_task_lora_wo_complete_20_0.001_0.005/checkpoint-600 \
       --video_dir /nfsdat/home/jmaslm/datahub/MUSIC_AVQA_R/video \
       --audio_dir /nfsdat/home/jmaslm/datahub/MUSIC_AVQA_R/audio