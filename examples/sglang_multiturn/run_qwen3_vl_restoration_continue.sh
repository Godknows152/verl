#!/usr/bin/env bash
# Image Restoration Multi-turn GRPO Training - Continue from checkpoint
# Model / GPUs: see restoration_multiturn_grpo.yaml
#
# Usage:
#   bash examples/sglang_multiturn/run_qwen3_vl_restoration_continue.sh
#
# Continues training from global_step_108 of multiturn_grpo_0503_repeat_penalty_from75
# Total 3 epochs (333 steps)

set -x
export HYDRA_FULL_ERROR=1
export HF_HUB_DISABLE_SSL_VERIFICATION=1
ulimit -n 65535

# Keep tool logging at warning level unless explicitly overridden
export VERL_LOGGING_LEVEL=WARN

# Fix CUDA library path - PyTorch's CUDA 12.8 runtime must take priority over system CUDA 12.4
export LD_LIBRARY_PATH=/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/torch/lib:/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Preload PyTorch's CUDA 12.8 runtime to avoid cudaGetDriverEntryPointByVersion errors
export LD_PRELOAD=/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12

# Bypass xformers flash-attn version check
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1

# Bypass SGLang CuDNN version check
export SGLANG_DISABLE_CUDNN_CHECK=1

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
LOG_DIR="/home/LXJ/Python_Projects/verl/log"

export VERL_LOG_DIR="$LOG_DIR"

TRAIN_FILES="${TRAIN_FILES:-$PROJECT_DIR/data/restoration/train.parquet}"
VAL_FILES="${VAL_FILES:-$PROJECT_DIR/data/restoration/test.parquet}"

mkdir -p "$LOG_DIR"

# Checkpoint to resume from
CKPT_PATH="/home/LXJ/Python_Projects/verl/checkpoints/verl/multiturn_grpo_0503_repeat_penalty_from75/global_step_108"
LOG_FILE="$LOG_DIR/continue_0504_3epoch_$(date +%Y%m%d_%H%M%S).nohup.log"

export PYTHONUNBUFFERED=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

echo "Starting continue training at $(date)" > "$LOG_FILE"

python3 -u -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='restoration_multiturn_grpo' \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    trainer.experiment_name="multiturn_grpo_0504_continue" \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$CKPT_PATH" \
    trainer.default_local_dir="checkpoints/verl/multiturn_grpo_0504_continue" \
    trainer.total_epochs=3 \
    trainer.save_freq=3 \
    trainer.test_freq=10 \
    "$@" 2>&1 | tee -a "$LOG_FILE"
