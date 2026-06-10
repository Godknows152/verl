#!/usr/bin/env bash
# Image Restoration Multi-turn GRPO Training
# Model / GPUs: see restoration_multiturn_grpo.yaml
#
# Prerequisites:
#   1. Convert dataset first:
#      python examples/data_preprocess/convert_restoration_dataset.py \
#        --input_parquet /path/to/raw.parquet \
#        --output_dir data/restoration
#
#   2. Run from project root:
#      bash examples/sglang_multiturn/run_qwen3_vl_restoration.sh
#
# All config lives in examples/sglang_multiturn/config/restoration_multiturn_grpo.yaml

set -x
export HYDRA_FULL_ERROR=1
export HF_HUB_DISABLE_SSL_VERIFICATION=1
ulimit -n 65535

# Keep tool logging at warning level unless explicitly overridden
export VERL_LOGGING_LEVEL=WARN

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
PYTHON_BIN="${PYTHON:-/home/LXJ/anaconda3/envs/verl/bin/python}"

# Expose LOG_DIR to restoration_tool.py file handler so INFO logs land next to training logs
export VERL_LOG_DIR="$LOG_DIR"

# ---------------------------------------------------------------------------
# Key parameters - all live in restoration_multiturn_grpo.yaml
# Override via env if needed: e.g. TRAIN_FILES=xxx bash run_xxx.sh
# ---------------------------------------------------------------------------
TRAIN_FILES="${TRAIN_FILES:-$PROJECT_DIR/data/restoration/train.parquet}"
VAL_FILES="${VAL_FILES:-$PROJECT_DIR/data/restoration/test.parquet}"

# ---------------------------------------------------------------------------
# Create log directory and log file
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_restoration.log"

# 清空上次训练残留的 restoration tool 日志
> "$LOG_DIR/restoration_tool_info.log"
> "$LOG_DIR/restoration_tools.log"

# 清空上次训练残留的临时修复图片
rm -rf /home/LXJ/tmp/verl_restoration
mkdir -p /home/LXJ/tmp/verl_restoration

# ---------------------------------------------------------------------------
# Unbuffered Python output - critical for seeing errors in real time
# ---------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
# All restoration checkpoints used by this experiment are expected to be
# available locally. Avoid runtime Hugging Face HEAD/download attempts inside
# tool workers, which can leave diffusers models half-initialized on meta tensors
# when the network is unstable.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1

# 显式使用 verl conda 环境的 Python，避免调用到系统 python3
PYTHON_BIN=/home/LXJ/anaconda3/envs/verl/bin/python
RAY_BIN=/home/LXJ/anaconda3/envs/verl/bin/ray

# Ray 临时目录（session/shared-memory 等），避免撑满 /tmp
export RAY_TMPDIR="${RAY_TMPDIR:-/home/LXJ/tmp/ray}"

# 清理上次训练残留的 Ray 会话，避免旧 object store/session 占满临时目录。
"$RAY_BIN" stop --force || true
mkdir -p "$RAY_TMPDIR"
find "$RAY_TMPDIR" -maxdepth 1 -mindepth 1 -name 'session_*' -exec rm -rf {} +


# ---------------------------------------------------------------------------
# Launch training (use -u for unbuffered output)
# ---------------------------------------------------------------------------
"$PYTHON_BIN" -u -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='restoration_multiturn_grpo' \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    trainer.experiment_name="multiturn_grpo_$(date +%m%d)" \
    trainer.default_local_dir="checkpoints/multiturn_grpo_$(date +%m%d)" \
    "$@" 2>&1 | tee "$LOG_FILE"
