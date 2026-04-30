#!/usr/bin/env python3
"""
Generate a data-driven IQA metric weight map from local restoration training data.

Method:
1. Score degraded training images in normalized IQA space (z-scores).
2. For each degradation type, measure each metric's positive deficit:
     deficit = max(0, -mean_z)
3. Down-weight unstable metrics using:
     reliability = 1 / (1 + std_z)
4. Convert to a weight vector and shrink toward uniform weights to avoid overfitting.

This yields a degradation-aware prior without keeping hand-written metric weights.
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_AGENT_TOOLS = _PROJECT_ROOT / 'restoration_tools' / 'agent_tools'
sys.path.insert(0, str(_AGENT_TOOLS))

os.environ['TRANSFORMERS_TORCH_LOAD_IS_SAFE'] = '1'
warnings.filterwarnings('ignore')

_METRIC_NAMES = ('qalign', 'maniqa', 'musiq', 'clipiqa', 'niqe')
_UNIFORM_WEIGHT = np.array([0.2] * 5, dtype=np.float64)


def _parse_extra_info(extra_info):
    if isinstance(extra_info, dict):
        return extra_info
    if isinstance(extra_info, str):
        try:
            return json.loads(extra_info)
        except json.JSONDecodeError:
            return {}
    return {}


def _sample_paths(paths, sample_size, rng):
    if sample_size <= 0 or len(paths) <= sample_size:
        return list(paths)
    return rng.choice(paths, size=sample_size, replace=False).tolist()


def collect_image_paths_by_type(parquet_paths, max_samples, rng):
    paths_by_type = defaultdict(set)
    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            extra_info = _parse_extra_info(row.get('extra_info', {}))
            degradation_type = extra_info.get('degradation_type', 'unknown')
            image_path = extra_info.get('image_path', None)
            if image_path and os.path.exists(image_path):
                paths_by_type[degradation_type].add(image_path)

    normalized_groups = {
        degradation_type: sorted(paths)
        for degradation_type, paths in sorted(paths_by_type.items())
    }
    if max_samples <= 0:
        return normalized_groups

    sampled_groups = {}
    per_type = max(1, max_samples // max(1, len(normalized_groups)))
    for degradation_type, paths in normalized_groups.items():
        sampled_groups[degradation_type] = _sample_paths(paths, per_type, rng)
    return sampled_groups


def compute_weight_vector(score_matrix, uniform_mix):
    mean_z = score_matrix.mean(axis=0)
    std_z = score_matrix.std(axis=0)
    deficit = np.maximum(0.0, -mean_z)
    reliability = 1.0 / (1.0 + std_z)
    signal = deficit * reliability

    if signal.sum() <= 1e-8:
        data_weight = _UNIFORM_WEIGHT.copy()
    else:
        data_weight = signal / signal.sum()

    final_weight = (1.0 - uniform_mix) * data_weight + uniform_mix * _UNIFORM_WEIGHT
    final_weight = final_weight / final_weight.sum()
    return {
        'mean_z': mean_z.tolist(),
        'std_z': std_z.tolist(),
        'deficit': deficit.tolist(),
        'reliability': reliability.tolist(),
        'signal': signal.tolist(),
        'data_weight': data_weight.tolist(),
        'final_weight': final_weight.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description='Compute a data-driven IQA weight map')
    parser.add_argument('--parquet', nargs='+', required=True, help='One or more training parquet files')
    parser.add_argument('--stats_json', required=True, help='Path to frozen IQA normalization stats JSON')
    parser.add_argument('--output_json', default='data/restoration/iqa_weight_map.json', help='Output weight map JSON')
    parser.add_argument('--device', default='cuda', help='Device for IQA models')
    parser.add_argument('--max_samples', type=int, default=0, help='Max total original images to process; 0 means all')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for stratified sampling')
    parser.add_argument('--uniform_mix', type=float, default=0.2, help='Shrinkage toward uniform weights in [0,1]')
    parser.add_argument('--qalign_path', default=None, help='Optional QAlign checkpoint path')
    args = parser.parse_args()

    if not (0.0 <= args.uniform_mix <= 1.0):
        raise ValueError('--uniform_mix must be in [0, 1]')

    rng = np.random.default_rng(args.seed)

    print('Collecting image paths by degradation type...')
    paths_by_type = collect_image_paths_by_type(args.parquet, args.max_samples, rng)
    total_images = 0
    for degradation_type, paths in paths_by_type.items():
        print(f'  {degradation_type}: {len(paths)} images')
        total_images += len(paths)
    if total_images == 0:
        raise ValueError('No valid image paths found in the provided training parquet files')
    print(f'Total images to process: {total_images}')

    print('\nLoading normalized IQA scorer...')
    from iqa_reward import IQAScore

    scorer = IQAScore(
        device=args.device,
        qalign_path=args.qalign_path,
        normalize_scores=True,
        normalization_stats_path=args.stats_json,
    )

    scores_by_type = {degradation_type: [] for degradation_type in paths_by_type}
    print('\nComputing normalized IQA scores...')
    for degradation_type, image_paths in paths_by_type.items():
        for image_path in tqdm(image_paths, desc=degradation_type, leave=False):
            score_vector = scorer.get_iqa_score(image_path)
            scores_by_type[degradation_type].append(score_vector)

    weight_map = {}
    details = {}
    for degradation_type, score_rows in scores_by_type.items():
        score_matrix = np.array(score_rows, dtype=np.float64)
        result = compute_weight_vector(score_matrix, args.uniform_mix)
        weight_map[degradation_type] = result['final_weight']
        details[degradation_type] = {
            'sample_count': int(score_matrix.shape[0]),
            'metrics': {
                metric_name: {
                    'mean_z': float(result['mean_z'][idx]),
                    'std_z': float(result['std_z'][idx]),
                    'deficit': float(result['deficit'][idx]),
                    'reliability': float(result['reliability'][idx]),
                    'signal': float(result['signal'][idx]),
                    'data_weight': float(result['data_weight'][idx]),
                    'final_weight': float(result['final_weight'][idx]),
                }
                for idx, metric_name in enumerate(_METRIC_NAMES)
            },
        }

    payload = {
        'version': 1,
        'method': 'positive_deficit_x_reliability_with_uniform_shrinkage',
        'normalization_stats_path': str(Path(args.stats_json).expanduser().resolve()),
        'metric_order': list(_METRIC_NAMES),
        'uniform_mix': float(args.uniform_mix),
        'default_weight': _UNIFORM_WEIGHT.tolist(),
        'weights': weight_map,
        'details': details,
        'source': {
            'parquet': args.parquet,
            'max_samples': args.max_samples,
            'device': args.device,
            'total_images': total_images,
        },
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print('\n=== IQA weight map ===')
    print(json.dumps(weight_map, indent=2))
    print(f'\nSaved weight map to {output_path}')


if __name__ == '__main__':
    main()