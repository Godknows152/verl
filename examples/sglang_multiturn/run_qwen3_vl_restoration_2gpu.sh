#!/usr/bin/env bash
# Image Restoration Multi-turn GRPO Training — 2-GPU variant
#
# 与 run_qwen3_vl_restoration.sh 的区别：
#   - 使用 restoration_multiturn_grpo_2gpu.yaml（2 卡配置）
#   - GPU 0: SGLang rollout + 全部图像修复模型 (replica 0) + IQA (replica 0)
#   - GPU 1: SGLang rollout + 全部图像修复模型 (replica 1) + IQA (replica 1)
#   - 对称池模式：每张卡都有完整的修复模型和 IQA，与 4 卡相同的复制模式
#   - checkpoint 保存在 checkpoints/verl/multiturn_grpo_2gpu/
#
# 使用方式：
#   从项目根目录执行：
#      bash examples/sglang_multiturn/run_qwen3_vl_restoration_2gpu.sh

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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
LOG_DIR="/home/LXJ/Python_Projects/verl/log"
export VERL_LOG_DIR="$LOG_DIR"
export RAY_TMPDIR="${RAY_TMPDIR:-/home/LXJ/tmp/ray}"

TRAIN_FILES="${TRAIN_FILES:-$PROJECT_DIR/data/restoration/train.parquet}"
VAL_FILES="${VAL_FILES:-$PROJECT_DIR/data/restoration/test.parquet}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_restoration_2gpu.log"

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
RAY_BIN=/home/LXJ/anaconda3/envs/verl/bin/ray

# 清理上次训练残留的 Ray 会话，避免旧 object store/session 占满临时目录。
"$RAY_BIN" stop --force || true
mkdir -p "$RAY_TMPDIR"
find "$RAY_TMPDIR" -maxdepth 1 -mindepth 1 -name 'session_*' -exec rm -rf {} +

$PYTHON_BIN -u -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='restoration_multiturn_grpo_2gpu' \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    trainer.experiment_name="multiturn_grpo_2gpu_$(date +%m%d)" \
    trainer.default_local_dir="checkpoints/verl/multiturn_grpo_2gpu/$(date +%m%d%H%M)" \
    "$@" 2>&1 | tee "$LOG_FILE"
