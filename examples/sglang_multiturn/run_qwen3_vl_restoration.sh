#!/bin/bash
# run_qwen3_vl_restoration.sh
#
# Launch Qwen3-VL multi-turn GRPO training for image restoration with verl.
#
# Prerequisites:
#   1. Install verl and its dependencies (see README).
#   2. Build the training dataset:
#        python examples/data_preprocess/convert_restoration_dataset.py \
#          --input_parquet /path/to/raw_dataset.parquet \
#          --output_dir data/restoration \
#          --train_ratio 0.9
#   3. Update MODEL_PATH below (or export QWEN3_VL_MODEL_PATH before running).
#   4. Verify GPU counts / memory utilisation settings match your hardware.
#
# Run from the project root:
#   bash examples/sglang_multiturn/run_qwen3_vl_restoration.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Explicit Environment Variables 
# ---------------------------------------------------------------------------
export QWEN3_VL_MODEL_PATH="/home/LXJ/Python_Projects/ViGoRL/Qwen_Model/Qwen3-VL-8B-Instruct_for_sft"
# 使用用逗号分隔的方式，自动读取data目录下所有parquet文件交给verl
export TRAIN_FILES=$(ls -d /home/LXJ/Python_Projects/verl/data/train/*.parquet | paste -sd "," -)
export VAL_FILES=$(ls -d /home/LXJ/Python_Projects/verl/data/test/*.parquet | paste -sd "," -)
export N_GPUS=4
export EXPERIMENT_NAME="多轮强化学习_$(date +%m%d)"

MODEL_PATH="${QWEN3_VL_MODEL_PATH}"
TRAIN_FILES="${TRAIN_FILES}"
VAL_FILES="${VAL_FILES}"
N_GPUS="${N_GPUS}"
EXPERIMENT_NAME="${EXPERIMENT_NAME}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=== verl Qwen3-VL Image Restoration GRPO Training ==="
echo "  Project root : ${PROJECT_ROOT}"
echo "  Model path   : ${MODEL_PATH}"
echo "  Train files  : ${TRAIN_FILES}"
echo "  Val files    : ${VAL_FILES}"
echo "  GPUs/node    : ${N_GPUS}"
echo "======================================================="

cd "${PROJECT_ROOT}"

python3 -m verl.trainer.main_ppo \
  --config-path="${PROJECT_ROOT}/examples/sglang_multiturn/config" \
  --config-name='restoration_multiturn_grpo' \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  "data.train_files=[${TRAIN_FILES}]" \
  "data.val_files=[${VAL_FILES}]" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger="['console','swanlab']" \
  "$@"
