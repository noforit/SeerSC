

. "$HOME"/miniconda3/etc/profile.d/conda.sh
conda activate SeerSC

MODEL_NAME=DeepSeek-R1-Qwen-7B
DATASET_NAME=math500
METHOD=SC # ESC, ASC, SeerSC

echo "MODEL=$MODEL_NAME  DATASET=$DATASET_NAME  METHOD=$METHOD"

SAVE_PATH="inference/results/${MODEL_NAME}_${DATASET_NAME}_${METHOD}_infile.jsonl"
port=8000


python inference/inference_vllm.py \
    --config configs/config.yaml \
    --model_name "${MODEL_NAME}" \
    --dataset_type "${DATASET_NAME}" \
    --method "${METHOD}" \
    --save_path "${SAVE_PATH}" \
    --port "${port}"

export MODEL_NAME=$(basename $MODEL_PATH)_sample_"${sample_num}"_temperature_"${temperature}"_method_"${method}"




# export DATASET_PATH='/home/jishiyu/O1/omni-math-rule-main/aime.jsonl'
# export DATASET_TYPE="aime"

# export DATASET_PATH='/home/jishiyu/O1/omni-math-rule-main/omni_math_rule.jsonl'
# export DATASET_TYPE="omni-math"

export DATASET_PATH='/share/home/syji/code/BoX/omni-math-rule-main/aime2024.jsonl'
export DATASET_TYPE="aime2024"
max_len=16384

python inference/inference_vllm_sample_time_serve.py --model "${MODEL_PATH}" \
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
    --method "${METHOD}" \
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
