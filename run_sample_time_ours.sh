#!/bin/bash
#SBATCH -J trl_train
#SBATCH -o logs_box/%j.log
#SBATCH -e logs_box/%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 48:00:00
# SBATCH -w gpu08
#SBATCH -c 4
#SBATCH --gres=gpu:nvidia_a800_80gb_pcie:1
#SBATCH --mem=50G

module use --append /share/public/apps/modulefiles
module load cuda/12.4.1

. "$HOME"/miniconda3/etc/profile.d/conda.sh
conda activate bon10

# vllm serve /share/home/syji/model/Qwen/Qwen3-4B --generation-config vllm --chat_template /share/home/syji/code/BoX/chat_template/origin.jinja --port 8001 --tensor_parallel_size 1
# export CUDA_VISIBLE_DEVICES=1
export MODEL_PATH='/home/syji/code/cot/train/model/Qwen2.5-7B-Instruct'
# export MODEL_PATH='/home/syji/model/Qwen/Qwen2.5-0.5B-Instruct'
# export MODEL_PATH='/home/syji/model/Qwen/Qwen2.5-1.5B-Instruct'
# export MODEL_PATH='/home/syji/model/Qwen/Qwen3-1.7B'

export MODEL_PATH='/share/home/syji/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
# export MODEL_PATH='/share/home/syji/model/deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
# export MODEL_PATH='/share/home/syji/model/Qwen/Qwen3-4B'
# export MODEL_PATH='/home/syji/model/Qwen/QwQ-32B'

# vllm serve /share/home/syji/model/Qwen/Qwen3-4B --generation-config vllm --tensor_parallel_size 1 --gpu_memory_utilization 0.9
#         --tensor-parallel-size 2 \
#         --port 8000

# export ADAPTER_PATH='/home/syji/code/cot/train/output/Qwen2.5-7B-Instruct_romdom'
# export ADAPTER_PATH='/home/syji/code/cot/train/output/Qwen2.5-7B-Instruct_merged_gsm8k_aime_1200_V2/checkpoint-10'
ADAPTER_PATH=''
export len_ratio=1.0
export sample_num=16
temperature=1.0
method='BOX'
# export MODEL_NAME=$(basename $(dirname $ADAPTER_PATH))-$(basename $ADAPTER_PATH)_$len_ratio

export MODEL_NAME=$(basename $MODEL_PATH)_sample_"${sample_num}"_temperature_"${temperature}"_nothink_method_"${method}"
port=8003


# export DATASET_PATH='/home/jishiyu/O1/omni-math-rule-main/aime.jsonl'
# export DATASET_TYPE="aime"

# export DATASET_PATH='/home/jishiyu/O1/omni-math-rule-main/omni_math_rule.jsonl'
# export DATASET_TYPE="omni-math"

export DATASET_PATH='/share/home/syji/code/BoX/omni-math-rule-main/aime2024.jsonl'
export DATASET_TYPE="aime2024"
max_len=16384

export DATASET_PATH='/share/home/syji/code/BoX/omni-math-rule-main/aime2025.jsonl'
export DATASET_TYPE="aime2025"
max_len=8192


# export DATASET_PATH='/share/home/syji/code/BoX/omni-math-rule-main/math500.jsonl'
# # export DATASET_PATH='/home/syji/code/cot/train/eval/omni-math-rule-main/math500_5.jsonl'
# export DATASET_TYPE="math500"
# max_len=8192

# export DATASET_PATH='/home/syji/code/cot/train/eval/omni-math-rule-main/gsm8k_test.jsonl'
# export DATASET_TYPE="gsm8k"
# max_len=10240

# export DATASET_PATH='/home/syji/code/cot/train/eval/omni-math-rule-main/gsm8k_train_data2.jsonl'
# export DATASET_TYPE="gsm8k_train_data2"

# export DATASET_PATH='/home/syji/code/cot/train/eval/omni-math-rule-main/gsm8k_train_sampled2000.jsonl'
# export DATASET_TYPE="gsm8k_train"

# export DATASET_PATH='/home/jishiyu/O1/Omni-MATH-main/Omni-Math.jsonl'
# export DATASET_TYPE="omni_math_all"

export DATASET_PATH='/share/home/syji/code/BoX/omni-math-rule-main/GPQA_Diamond_shuffle.jsonl'
export DATASET_TYPE="gpqa"
max_len=8192
# # # 8192
# export DATASET_PATH='/share/home/syji/code/BoX/omni-math-rule-main/amc23.jsonl'
# export DATASET_TYPE="amc23"
# max_len=8192


export SAVE_PATH="inference/results_time_serve/${MODEL_NAME}_${DATASET_TYPE}_infile.jsonl"
# export MASTER_ADDR="localhost"
# export MASTER_PORT="1231"
# export GLOO_SOCKET_IFNAME="lo"
# export NCCL_SOCKET_IFNAME="lo"
# export WANDB_DISABLED=true
# unset VLLM_USE_MODELSCOPE

# export CUDA_VISIBLE_DEVICES=0


# python inference/inference_vllm.py --model $MODEL_PATH \
python inference/inference_vllm_sample_time_ours.py --model "${MODEL_PATH}" \
    --data_file "${DATASET_PATH}" \
    --tensor_parallel_size 1 \
    --adapter_path "${ADAPTER_PATH}" \
    --len_ratio "${len_ratio}" \
    --save_path "${SAVE_PATH}" \
    --sample_num "${sample_num}" \
    --temperature "${temperature}" \
    --max_len "${max_len}" \
    --dataset_type "${DATASET_TYPE}" \
    --generation_batch_size 1 \
    --method "${method}" \
    --port "${port}"


EXP_NAME="${MODEL_NAME}"
# DATASET_TYPE="omni-math"

INPUT_PATH="${SAVE_PATH}"

set -ex
SPLIT="test"
python3 -u evaluation/math_eval.py \
    --data_name "${DATASET_TYPE}" \
    --exp_name "${EXP_NAME}" \
    --split "${SPLIT}" \
    --output_dir "./output_time_serve" \
    --input_path "${INPUT_PATH}"
