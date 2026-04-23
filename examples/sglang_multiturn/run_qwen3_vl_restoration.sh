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
#   MODEL_PATH, TRAIN_FILES, VAL_FILES, ROLLOUT_GPUS, VISIBLE_GPUS, TOTAL_EPOCHS

set -x
export HYDRA_FULL_ERROR=1
export SGLANG_DISABLE_CUDA_GRAPH=1
export HF_HUB_DISABLE_SSL_VERIFICATION=1
ulimit -n 65535

# Fix CUDA library path - PyTorch's CUDA 12.8 runtime must take priority over system CUDA 12.4
export LD_LIBRARY_PATH=/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/torch/lib:/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Preload PyTorch's CUDA 12.8 runtime to avoid cudaGetDriverEntryPointByVersion errors
# This must be set BEFORE ray.init() for it to propagate to spawned subprocesses
export LD_PRELOAD=/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12

# Bypass xformers flash-attn version check (xformers requires 2.7.1-2.8.2, we have 2.8.3)
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1

# Bypass SGLang CuDNN version check (PyTorch 2.9.1 needs CuDNN 9.15+, we have 9.16)
export SGLANG_DISABLE_CUDNN_CHECK=1

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
LOG_DIR="/home/LXJ/Python_Projects/verl/log"

# ---------------------------------------------------------------------------
# Key parameters (most config lives in restoration_multiturn_grpo.yaml)
# ---------------------------------------------------------------------------
MODEL_PATH="${QWEN3_VL_MODEL_PATH:-/home/LXJ/Python_Projects/ViGoRL/Qwen_Model/Qwen3-VL-8B-Instruct_for_sft}"
TRAIN_FILES="${TRAIN_FILES:-$PROJECT_DIR/data/restoration/train.parquet}"
VAL_FILES="${VAL_FILES:-$PROJECT_DIR/data/restoration/test.parquet}"
# Optional: override rollout GPU count. If empty, use YAML value as-is.
# Example: ROLLOUT_GPUS=2
ROLLOUT_GPUS="${ROLLOUT_GPUS:-}"
# Optional: constrain visible GPUs for the whole run.
# Example: VISIBLE_GPUS=0,1,2 (then set ROLLOUT_GPUS=2 to leave one GPU for agent worker)
VISIBLE_GPUS="${VISIBLE_GPUS:-}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"

if [[ -n "$VISIBLE_GPUS" ]]; then
    export CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
fi

ROLLOUT_GPU_ARGS=()
if [[ -n "$ROLLOUT_GPUS" ]]; then
    ROLLOUT_GPU_ARGS+=("trainer.n_gpus_per_node=$ROLLOUT_GPUS")
fi

# ---------------------------------------------------------------------------
# Create log directory and log file
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_restoration.log"

# ---------------------------------------------------------------------------
# Unbuffered Python output - critical for seeing errors in real time
# ---------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# ---------------------------------------------------------------------------
# Launch training (use -u for unbuffered output)
# ---------------------------------------------------------------------------
python3 -u -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='restoration_multiturn_grpo' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    "${ROLLOUT_GPU_ARGS[@]}" \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.experiment_name="multiturn_grpo_$(date +%m%d)" \
    "$@" 2>&1 | tee "$LOG_FILE"
