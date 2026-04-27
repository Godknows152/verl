#!/usr/bin/env python3
"""
Pre-download IQA models to local cache.

Run this before starting training to avoid SSL/HuggingFace download issues
during the first run.

Usage:
    cd /home/LXJ/Python_Projects/verl
    python restoration_tools/preload_iqa_models.py
"""

import os
import sys
import torch

# Add restoration_tools/agent_tools to path
AGENT_TOOLS_PATH = os.path.join(os.path.dirname(__file__), "agent_tools")
sys.path.insert(0, AGENT_TOOLS_PATH)

QALIGN_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "q_align")

# Disable SSL verification for HuggingFace downloads (if behind proxy)
# Set to "1" if you encounter SSL errors behind a corporate proxy
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = os.environ.get("HF_HUB_DISABLE_SSL_VERIFICATION", "1")


def preload_qalign():
    """Pre-download QAlign model from local path (validates local cache)."""
    print("=" * 60)
    print("Pre-loading QAlign model...")
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            QALIGN_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": "cpu"},  # Load on CPU to save GPU memory
        )
        print(f"  QAlign loaded: {QALIGN_PATH}")
        del model
        torch.cuda.empty_cache()
        print("  OK")
    except Exception as e:
        print(f"  WARNING: {e}")


def preload_pyiqa_models():
    """Pre-download pyiqa models (MANIQA, MUSIQ, CLIPIQA, NIQE)."""
    print("=" * 60)
    print("Pre-loading pyiqa models (MANIQA, MUSIQ, CLIPIQA, NIQE)...")

    import pyiqa

    models = ["maniqa", "musiq", "clipiqa", "niqe"]

    for name in models:
        print(f"  Downloading {name}...", end=" ", flush=True)
        try:
            # Download only (without running inference) by creating the metric on CPU
            # This will cache the model files to ~/.cache/huggingface/ or ~/.cache/timm/
            metric = pyiqa.create_metric(name, device="cpu")
            print("OK")
            del metric
        except Exception as e:
            print(f"FAILED: {e}")
        torch.cuda.empty_cache()


def main():
    print("=" * 60)
    print("IQA Model Pre-loader")
    print("=" * 60)
    print(f"QAlign path: {QALIGN_PATH}")
    print(f"Device: cuda" if torch.cuda.is_available() else "Device: cpu")
    print()

    preload_qalign()
    preload_pyiqa_models()

    print()
    print("=" * 60)
    print("All IQA models pre-loaded. You can now start training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
