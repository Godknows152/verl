#!/usr/bin/env bash
# Image Restoration Multi-turn GRPO Training — 3-GPU variant
#
# 与 run_qwen3_vl_restoration.sh 的区别：
#   - 使用 restoration_multiturn_grpo_3gpu.yaml（3 卡配置）
#   - 模型路径指向 step_108 合并后的 HF 模型
#   - resume_mode=resume_path，从 checkpoints/verl/multiturn_grpo_3gpu/global_step_12 续训
#   - 本次调整：kl_loss_coef 0.1→0.05，entropy_coeff 0.01→0.02
#   - checkpoint 保存在 checkpoints/verl/multiturn_grpo_3gpu_kl05_ent02/
#
# 使用方式：
#   1. 先合并 step_108 checkpoint（见文件顶部注释）
#   2. 从项目根目录执行：
#      bash examples/sglang_multiturn/run_qwen3_vl_restoration_3gpu.sh
#
# 合并 checkpoint 命令（首次使用前运行一次）：
#   cd /home/LXJ/Python_Projects/verl
#   python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir checkpoints/verl/multiturn_grpo_0503_repeat_penalty_from75/global_step_108/actor \
#     --target_dir checkpoints/merged

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
    "$@" 2>&1 | tee "$LOG_FILE"
