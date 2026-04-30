#!/usr/bin/env python3
"""
Compute frozen IQA normalization statistics from local restoration training data.

For multi-turn RL, keep these statistics fixed during training. By default this
script uses the original training images referenced by the parquet dataset. If
you also have offline-generated restored images from previous runs, you can add
them via --include_image_dir to better approximate the rollout distribution
without introducing online reward drift.

Usage:
    python scripts/compute_iqa_stats.py \
        --parquet data/restoration/train.parquet \
        --output_json data/restoration/iqa_stats.json
"""
import argparse
import json
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

_RUNTIME_METRICS = ('qalign', 'maniqa', 'musiq', 'clipiqa', 'niqe')
_IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def _parse_extra_info(extra_info):
    if isinstance(extra_info, dict):
        return extra_info
    if isinstance(extra_info, str):
        try:
            return json.loads(extra_info)
        except json.JSONDecodeError:
            return {}
    return {}


def _sample_paths(paths, max_samples, rng):
    if max_samples <= 0 or len(paths) <= max_samples:
        return list(paths)
    return rng.choice(paths, size=max_samples, replace=False).tolist()


def collect_image_paths(parquet_paths, max_samples, rng):
    """Collect unique image paths from parquet, sampling across degradation types."""
    import pandas as pd

    # Group by degradation type to ensure coverage
    paths_by_type = {}
    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            ei = _parse_extra_info(row.get('extra_info', {}))
            deg_type = ei.get('degradation_type', 'unknown')
            img_path = ei.get('image_path', None)
            if img_path and os.path.exists(img_path):
                paths_by_type.setdefault(deg_type, set()).add(img_path)

    if not paths_by_type:
        return []

    normalized_groups = {
        deg_type: sorted(paths)
        for deg_type, paths in sorted(paths_by_type.items())
    }
    if max_samples <= 0:
        sampled = []
        for deg_type, paths in normalized_groups.items():
            print(f"  {deg_type}: {len(paths)} images")
            sampled.extend(paths)
        return sampled

    # Sample evenly across types
    per_type = max(1, max_samples // len(normalized_groups))
    sampled = []
    for deg_type, paths in normalized_groups.items():
        n = min(per_type, len(paths))
        sampled.extend(_sample_paths(paths, n, rng))
        print(f"  {deg_type}: {n}/{len(paths)} images")

    if len(sampled) < max_samples:
        chosen = set(sampled)
        remaining = []
        for paths in normalized_groups.values():
            remaining.extend([path for path in paths if path not in chosen])
        extra_budget = min(max_samples - len(sampled), len(remaining))
        if extra_budget > 0:
            sampled.extend(_sample_paths(remaining, extra_budget, rng))

    return sampled


def collect_extra_image_paths(image_dirs, max_samples, rng):
    """Collect optional offline-generated restored images from local directories."""
    collected = []
    for image_dir in image_dirs:
        root = Path(image_dir).expanduser().resolve()
        if not root.exists():
            print(f"  Skipping missing extra image dir: {root}")
            continue
        for path in root.rglob('*'):
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                collected.append(str(path))

    unique_paths = sorted(set(collected))
    if max_samples > 0:
        unique_paths = _sample_paths(unique_paths, max_samples, rng)
    return unique_paths


def main():
    parser = argparse.ArgumentParser(description='Compute IQA normalization statistics')
    parser.add_argument('--parquet', nargs='+', required=True,
                        help='One or more training parquet files')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Max original training images to process; 0 means all')
    parser.add_argument('--include_image_dir', action='append', default=[],
                        help='Optional directory containing offline-generated restored images')
    parser.add_argument('--max_extra_images', type=int, default=0,
                        help='Max extra restored images to process; 0 means all')
    parser.add_argument('--device', default='cuda', help='Device for IQA models')
    parser.add_argument('--output_json', default='data/restoration/iqa_stats.json',
                        help='Save runtime stats JSON file')
    parser.add_argument('--qalign_path', default=None,
                        help='Path to QAlign checkpoint (default: restoration_tools/checkpoints/q_align)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed used when sampling local training images')
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # Collect image paths
    print("Collecting image paths...")
    original_image_paths = collect_image_paths(args.parquet, args.max_samples, rng)
    print(f"Original images to process: {len(original_image_paths)}")

    extra_image_paths = collect_extra_image_paths(args.include_image_dir, args.max_extra_images, rng)
    if extra_image_paths:
        print(f"Extra offline restored images to process: {len(extra_image_paths)}")

    image_paths = list(dict.fromkeys(original_image_paths + extra_image_paths))
    print(f"Total images to process: {len(image_paths)}")
    if not image_paths:
        raise ValueError('No valid image paths found in the provided training data sources')

    # Load IQA scorer (only once)
    print("\nLoading IQA models...")
    from iqa_reward import IQAScore
    scorer = IQAScore(device=args.device, qalign_path=args.qalign_path, normalize_scores=False)

    # Compute raw scores for all images
    all_scores = {
        'qalign': [],
        'maniqa': [],
        'musiq': [],
        'clipiqa': [],
        'niqe': [],
        'niqe_raw': [],
    }

    print("\nComputing IQA scores...")
    for img_path in tqdm(image_paths):
        try:
            with torch.no_grad():
                score_dict = scorer.get_raw_iqa_scores(img_path)

                all_scores['qalign'].append(score_dict['qalign'])
                all_scores['maniqa'].append(score_dict['maniqa'])
                all_scores['musiq'].append(score_dict['musiq'])
                all_scores['clipiqa'].append(score_dict['clipiqa'])
                all_scores['niqe'].append(score_dict['niqe'])
                all_scores['niqe_raw'].append(score_dict['niqe_raw'])
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

    runtime_stats = {
        metric_name: {
            'mean': stats[metric_name]['mean'],
            'std': stats[metric_name]['std'],
            'n': stats[metric_name]['n'],
        }
        for metric_name in _RUNTIME_METRICS
        if metric_name in stats
    }

    payload = {
        'version': 1,
        'normalization': 'zscore',
        'frozen': True,
        'score_space': 'qalign_maniqa_musiq_clipiqa_exp_niqe',
        'source': {
            'parquet': args.parquet,
            'include_image_dir': args.include_image_dir,
            'num_original_images': len(original_image_paths),
            'num_extra_images': len(extra_image_paths),
        },
        'metrics': runtime_stats,
        'details': stats,
    }

    print("\n=== Runtime normalization payload ===")
    print(json.dumps(payload['metrics'], indent=2))

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved runtime stats to {output_path}")


if __name__ == '__main__':
    main()
