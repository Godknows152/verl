#!/usr/bin/env bash
# Image Restoration Multi-turn GRPO Training — 3-GPU variant
#
# 与 run_qwen3_vl_restoration.sh 的区别：
#   - 使用 restoration_multiturn_grpo_3gpu.yaml（3 卡配置）
#   - 模型路径指向 Step102 合并后的 HF 模型（checkpoints/merged/Train4/Step102）
#   - resume_mode=disable，从 Step102 全新开始，不续训旧 VERL checkpoint
#   - tool config 中 auto_unload=false（显存充足，无需每次采样后自卸载）
#   - GPU 0-2: SGLang rollout + 全部图像修复模型 (replica) + IQA (replica)
#   - 对称池模式：每张卡都有完整的修复模型和 IQA，与 4 卡相同的复制模式
#
# 使用方式：
#   从项目根目录执行：
#     bash examples/sglang_multiturn/run_qwen3_vl_restoration_3gpu.sh
#
# step120 已合并至 checkpoints/merged/step120，无需重新合并。

set -x
export HYDRA_FULL_ERROR=1
export HF_HUB_DISABLE_SSL_VERIFICATION=1
ulimit -n 65535

export VERL_LOGGING_LEVEL=WARN

# Fix CUDA library path
export LD_LIBRARY_PATH=/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/torch/lib:/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Preload PyTorch's CUDA 12.8 runtime
export LD_PRELOAD=/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12

export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export SGLANG_DISABLE_CUDNN_CHECK=1

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
LOG_DIR="/home/LXJ/Python_Projects/verl/log"
export VERL_LOG_DIR="$LOG_DIR"

TRAIN_FILES="${TRAIN_FILES:-$PROJECT_DIR/data/restoration/train.parquet}"
VAL_FILES="${VAL_FILES:-$PROJECT_DIR/data/restoration/test.parquet}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_restoration_3gpu.log"

# 清空上次训练残留的 restoration tool 日志
> "$LOG_DIR/restoration_tool_info.log"
> "$LOG_DIR/restoration_tools.log"

# 清空上次训练残留的临时修复图片
rm -rf /home/LXJ/tmp/verl_restoration
mkdir -p /home/LXJ/tmp/verl_restoration

export PYTHONUNBUFFERED=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# 显式使用 verl conda 环境的 Python，避免调用到系统 python3
PYTHON_BIN=/home/LXJ/anaconda3/envs/verl/bin/python

$PYTHON_BIN -u -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='restoration_multiturn_grpo_3gpu' \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    trainer.experiment_name="multiturn_grpo_3gpu_$(date +%m%d)" \
    trainer.default_local_dir="checkpoints/verl/multiturn_grpo_3gpu/$(date +%m%d%H%M)" \
    "$@" 2>&1 | tee "$LOG_FILE"
