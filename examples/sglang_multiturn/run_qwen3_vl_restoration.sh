#!/usr/bin/env bash
# Image Restoration Multi-turn GRPO Training
# Model : Qwen3-VL-7B-Instruct
# GPUs  : 4x GPU (GPU 0-2 → SGLang rollout, GPU 3 → restoration models + IQA)
#
# Prerequisites:
#   1. Convert dataset first:
#      python examples/data_preprocess/convert_restoration_dataset.py \
#        --input_parquet /path/to/raw.parquet \
#        --output_dir data/restoration
#
#   2. Run from project root:
#      export QWEN3_VL_MODEL_PATH="/path/to/Qwen3-VL-7B-Instruct"
#      bash examples/sglang_multiturn/run_qwen3_vl_restoration.sh
#
# Key parameters (override via env or command line, see YAML for full config):
#   MODEL_PATH, TRAIN_FILES, VAL_FILES, N_GPUS, TOTAL_EPOCHS

set -x
export HYDRA_FULL_ERROR=1
ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# ---------------------------------------------------------------------------
# Key parameters (most config lives in restoration_multiturn_grpo.yaml)
# ---------------------------------------------------------------------------
MODEL_PATH="${QWEN3_VL_MODEL_PATH:-Qwen/Qwen3-VL-7B-Instruct}"
TRAIN_FILES="${TRAIN_FILES:-$PROJECT_DIR/data/restoration/train.parquet}"
VAL_FILES="${VAL_FILES:-$PROJECT_DIR/data/restoration/test.parquet}"
N_GPUS="${N_GPUS:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='restoration_multiturn_grpo' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.total_epochs=$TOTAL_EPOCHS \
    "$@"
