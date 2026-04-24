#!/usr/bin/env python3
"""
Compute IQA normalization statistics (mean/std) from restoration dataset images.

Usage:
    python scripts/compute_iqa_stats.py \
        --parquet data/restoration/train.parquet \
        --max_samples 500 \
        --output_json scripts/iqa_stats.json

The output JSON can be used to update the mean_std dict in iqa_reward.py.
"""
import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add restoration_tools to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_AGENT_TOOLS = _PROJECT_ROOT / 'restoration_tools' / 'agent_tools'
sys.path.insert(0, str(_AGENT_TOOLS))

os.environ['TRANSFORMERS_TORCH_LOAD_IS_SAFE'] = '1'
warnings.filterwarnings('ignore')


def collect_image_paths(parquet_path, max_samples):
    """Collect unique image paths from parquet, sampling across degradation types."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)

    # Group by degradation type to ensure coverage
    paths_by_type = {}
    for _, row in df.iterrows():
        ei = row['extra_info']
        deg_type = ei.get('degradation_type', 'unknown')
        img_path = ei.get('image_path', None)
        if img_path and os.path.exists(img_path):
            paths_by_type.setdefault(deg_type, []).append(img_path)

    # Sample evenly across types
    per_type = max(1, max_samples // len(paths_by_type))
    sampled = []
    for deg_type, paths in paths_by_type.items():
        n = min(per_type, len(paths))
        sampled.extend(np.random.choice(paths, n, replace=False))
        print(f"  {deg_type}: {n}/{len(paths)} images")

    return sampled


def main():
    parser = argparse.ArgumentParser(description='Compute IQA normalization statistics')
    parser.add_argument('--parquet', required=True, help='Path to training parquet file')
    parser.add_argument('--max_samples', type=int, default=500,
                        help='Max images to process (default: 500)')
    parser.add_argument('--device', default='cuda', help='Device for IQA models')
    parser.add_argument('--output_json', help='Save stats to JSON file')
    parser.add_argument('--qalign_path', default=None,
                        help='Path to QAlign checkpoint (default: restoration_tools/checkpoints/q_align)')
    args = parser.parse_args()

    # Collect image paths
    print("Collecting image paths...")
    image_paths = collect_image_paths(args.parquet, args.max_samples)
    print(f"Total images to process: {len(image_paths)}")

    # Load IQA scorer (only once)
    print("\nLoading IQA models...")
    from iqa_reward import IQAScore
    scorer = IQAScore(device=args.device, qalign_path=args.qalign_path)

    # Compute raw scores for all images
    all_scores = {
        'qalign': [],
        'maniqa': [],
        'musiq': [],
        'clipiqa': [],
        'niqe_raw': [],       # raw NIQE (lower is better)
        'niqe_transformed': [],  # exp(-niqe/10) after transform
    }

    print("\nComputing IQA scores...")
    for img_path in tqdm(image_paths):
        try:
            with torch.no_grad():
                img = scorer.preprocess_image(img_path)

                qalign = scorer.qalign_metric.score([img], task_="quality", input_="image").item()
                maniqa = scorer.maniqa_metric(img).item()
                musiq = scorer.musiq_metric(img).item()
                clipiqa = scorer.clipiqa_metric(img).item()
                niqe_raw = scorer.niqe_metric(img).item()
                niqe_t = math.exp(-niqe_raw / 10.0)

                all_scores['qalign'].append(qalign)
                all_scores['maniqa'].append(maniqa)
                all_scores['musiq'].append(musiq)
                all_scores['clipiqa'].append(clipiqa)
                all_scores['niqe_raw'].append(niqe_raw)
                all_scores['niqe_transformed'].append(niqe_t)
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")

    # Compute statistics
    print("\n=== IQA Normalization Statistics ===")
    stats = {}
    for metric_name, values in all_scores.items():
        if not values:
            continue
        arr = np.array(values)
        mean = float(arr.mean())
        std = float(arr.std())
        stats[metric_name] = {'mean': mean, 'std': std, 'n': len(values)}
        p5, p95 = np.percentile(arr, [5, 95])
        print(f"  {metric_name:20s}: mean={mean:.6f}, std={std:.6f}, "
              f"min={arr.min():.4f}, max={arr.max():.4f}, p5={p5:.4f}, p95={p95:.4f}")

    # Print the code-ready dict
    print("\n=== Copy-paste ready ===")
    # For the current code, only qalign/maniqa/musiq are normalized;
    # clipiqa and niqe_transformed also need entries
    ready = {}
    for key in ['qalign', 'maniqa', 'musiq']:
        if key in stats:
            ready[key] = {'mean': stats[key]['mean'], 'std': stats[key]['std']}
    # these two currently aren't normalized but should be
    for key in ['clipiqa', 'niqe_transformed']:
        if key in stats:
            ready[key] = {'mean': stats[key]['mean'], 'std': stats[key]['std']}

    print(json.dumps(ready, indent=2))

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved full stats to {args.output_json}")


if __name__ == '__main__':
    main()
