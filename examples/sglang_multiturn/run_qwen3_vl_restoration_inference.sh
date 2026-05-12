#!/usr/bin/env bash

set -x
export HYDRA_FULL_ERROR=1
export HF_HUB_DISABLE_SSL_VERIFICATION=1
ulimit -n 65535

export VERL_LOGGING_LEVEL=WARN

# Keep the inference launcher aligned with the current restoration training environment.
export LD_LIBRARY_PATH=/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/torch/lib:/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/home/LXJ/anaconda3/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export PYTHONUNBUFFERED=1

PROJECT_DIR="$(pwd)"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/log}"
PYTHON_BIN="${PYTHON:-/home/LXJ/anaconda3/envs/verl/bin/python}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$PROJECT_DIR/examples/sglang_multiturn/config/restoration_multiturn_grpo.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_restoration_inference.log"

EXTRA_ARGS=()
if [[ -n "$MODEL_PATH" ]]; then
    EXTRA_ARGS+=(--model-path "$MODEL_PATH")
fi
if [[ -n "$DATA_PATH" ]]; then
    EXTRA_ARGS+=(--data-path "$DATA_PATH")
fi
if [[ -n "$IMAGE_PATH" ]]; then
    EXTRA_ARGS+=(--image-path "$IMAGE_PATH")
fi
if [[ -n "$IMAGE_DIR" ]]; then
    EXTRA_ARGS+=(--image-dir "$IMAGE_DIR")
fi
if [[ -n "$DEGRADATION_TYPE" ]]; then
    EXTRA_ARGS+=(--degradation-type "$DEGRADATION_TYPE")
fi
if [[ -n "$USER_PROMPT" ]]; then
    EXTRA_ARGS+=(--user-prompt "$USER_PROMPT")
fi
if [[ -n "$TOOL_CONFIG" ]]; then
    EXTRA_ARGS+=(--tool-config "$TOOL_CONFIG")
fi
if [[ -n "$TOOL_OUTPUT_DIR" ]]; then
    EXTRA_ARGS+=(--tool-output-dir "$TOOL_OUTPUT_DIR")
fi

"$PYTHON_BIN" -u scripts/eval_restoration_inference.py \
    --train-config "$TRAIN_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}" \
    "$@" 2>&1 | tee "$LOG_FILE"