#!/usr/bin/env python3
"""
Compute degradation-action-aware IQA weight map (v2).

Unlike v1 (which only measures z-score deficit on degraded images),
v2 runs each restoration tool on sampled images and measures per-metric
IQA improvements.  The weight for each metric is determined by how well
that metric distinguishes targeted (degradation-specific) tools from
generic (general-purpose) tools:

  discriminability[metric] = targeted_delta[metric] - generic_delta[metric]

Metrics where targeted tools outperform generic ones receive higher
weights, so that the IQA reward naturally favors degradation-specific
tools under the new weighting scheme.

Usage:
    python scripts/compute_iqa_weight_map_v2.py \
        --parquet data/restoration/train.parquet \
        --stats_json data/restoration/iqa_stats.json \
        --output_json data/restoration/iqa_weight_map_v2.json \
        --device cuda:0 \
        --iqa_device cuda:3 \
        --samples_per_type 32
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
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
sys.path.insert(0, str(_PROJECT_ROOT))

os.environ['TRANSFORMERS_TORCH_LOAD_IS_SAFE'] = '1'
warnings.filterwarnings('ignore')

_METRIC_NAMES = ('qalign', 'maniqa', 'musiq', 'clipiqa', 'niqe')
_UNIFORM_WEIGHT = np.array([0.2] * 5, dtype=np.float64)

# All restoration tools (excluding stop)
ALL_RESTORATION_ACTIONS = [
    'real_esrgan', 'scunet', 'retinexformer_fivek', 'hvicidnet', 'lightdiff',
    'turbo_rain', 's2former', 'idt', 'ridcp', 'kanet', 'turbo_snow', 'snowmaster',
]

# Degradation type → targeted (specialized) tools
DEGRADATION_TARGETED_ACTIONS = {
    'night':       ['retinexformer_fivek', 'hvicidnet', 'lightdiff'],
    'rain_streak': ['s2former', 'turbo_rain', 'idt'],
    'rain_drop':   ['idt', 'turbo_rain', 's2former'],
    'rain_drive':  ['turbo_rain', 'idt', 's2former'],
    'fog':         ['ridcp', 'kanet'],
    'snow':        ['turbo_snow', 'snowmaster'],
}

# Generic tools that work across all degradation types
GENERIC_ACTIONS = ['real_esrgan', 'scunet']


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


def collect_image_paths_by_type(parquet_paths, samples_per_type, rng):
    """Collect and sample image paths grouped by degradation type."""
    paths_by_type = defaultdict(list)
    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            extra_info = _parse_extra_info(row.get('extra_info', {}))
            degradation_type = extra_info.get('degradation_type', 'unknown')
            # Try multiple image path fields for compatibility
            image_path = (
                extra_info.get('image_path')
                or extra_info.get('original_image')
                or (extra_info.get('create_kwargs', {}) or {}).get('image_path')
                or (extra_info.get('create_kwargs', {}) or {}).get('original_image')
            )
            if image_path and os.path.exists(image_path):
                paths_by_type[degradation_type].append(image_path)

    # Deduplicate and sample
    sampled_groups = {}
    for degradation_type in sorted(paths_by_type):
        unique_paths = sorted(set(paths_by_type[degradation_type]))
        sampled_groups[degradation_type] = _sample_paths(unique_paths, samples_per_type, rng)
    return sampled_groups


def compute_discriminability_weights(
    targeted_deltas: np.ndarray,
    generic_deltas: np.ndarray,
    uniform_mix: float,
) -> dict:
    """Compute weight vector based on discriminability of targeted vs generic tools.

    Args:
        targeted_deltas: Array of shape (N_targeted, 5) — per-metric IQA deltas
                         from targeted (specialized) tools.
        generic_deltas:  Array of shape (N_generic, 5) — per-metric IQA deltas
                         from generic tools.
        uniform_mix:     Shrinkage toward uniform weights in [0, 1].

    Returns:
        Dict with weight computation details.
    """
    targeted_mean = targeted_deltas.mean(axis=0) if targeted_deltas.size > 0 else np.zeros(5)
    generic_mean = generic_deltas.mean(axis=0) if generic_deltas.size > 0 else np.zeros(5)

    # Discriminability: how much targeted tools outperform generic ones per metric
    discriminability = targeted_mean - generic_mean

    # Only reward metrics where targeted tools are better (positive discriminability)
    positive_signal = np.maximum(0.0, discriminability)

    # Also include a baseline from the v1 deficit method (degraded image z-score deficit)
    # to ensure metrics that are generally poor on degraded images still get some weight.
    # Here we use the absolute targeted improvement as a proxy for "importance".
    importance = np.maximum(0.0, targeted_mean)

    # Combine: weight = positive_signal + importance * 0.3
    # This ensures that even if discriminability is zero for some metric,
    # it still gets some weight if targeted tools improve it substantially.
    combined_signal = positive_signal + importance * 0.3

    if combined_signal.sum() <= 1e-8:
        data_weight = _UNIFORM_WEIGHT.copy()
    else:
        data_weight = combined_signal / combined_signal.sum()

    # Shrink toward uniform to avoid overfitting
    final_weight = (1.0 - uniform_mix) * data_weight + uniform_mix * _UNIFORM_WEIGHT
    final_weight = final_weight / final_weight.sum()

    return {
        'targeted_mean_delta': targeted_mean.tolist(),
        'generic_mean_delta': generic_mean.tolist(),
        'discriminability': discriminability.tolist(),
        'positive_signal': positive_signal.tolist(),
        'importance': importance.tolist(),
        'combined_signal': combined_signal.tolist(),
        'data_weight': data_weight.tolist(),
        'final_weight': final_weight.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compute degradation-action-aware IQA weight map (v2)'
    )
    parser.add_argument(
        '--parquet', nargs='+', required=True,
        help='One or more training parquet files'
    )
    parser.add_argument(
        '--stats_json', required=True,
        help='Path to frozen IQA normalization stats JSON'
    )
    parser.add_argument(
        '--output_json', default='data/restoration/iqa_weight_map_v2.json',
        help='Output weight map JSON'
    )
    parser.add_argument(
        '--device', default='cuda:0',
        help='Device for restoration models'
    )
    parser.add_argument(
        '--iqa_device', default='cuda:3',
        help='Device for IQA models'
    )
    parser.add_argument(
        '--samples_per_type', type=int, default=32,
        help='Number of images to sample per degradation type'
    )
    parser.add_argument(
        '--uniform_mix', type=float, default=0.2,
        help='Shrinkage toward uniform weights in [0,1]'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for stratified sampling'
    )
    parser.add_argument(
        '--qalign_path', default=None,
        help='Optional QAlign checkpoint path'
    )
    parser.add_argument(
        '--actions', nargs='*', default=None,
        help='Subset of restoration actions to run (default: all 12)'
    )
    args = parser.parse_args()

    if not (0.0 <= args.uniform_mix <= 1.0):
        raise ValueError('--uniform_mix must be in [0, 1]')

    rng = np.random.default_rng(args.seed)
    actions_to_run = args.actions or ALL_RESTORATION_ACTIONS

    # Step 1: Collect and sample image paths
    print('Collecting image paths by degradation type...')
    paths_by_type = collect_image_paths_by_type(args.parquet, args.samples_per_type, rng)
    total_images = 0
    for degradation_type, paths in sorted(paths_by_type.items()):
        print(f'  {degradation_type}: {len(paths)} images')
        total_images += len(paths)
    if total_images == 0:
        raise ValueError('No valid image paths found in the provided training parquet files')
    print(f'Total images to process: {total_images}')

    # Step 2: Load IQA scorer (normalized)
    print('\nLoading normalized IQA scorer on {}...'.format(args.iqa_device))
    from iqa_reward import IQAScore

    scorer = IQAScore(
        device=args.iqa_device,
        qalign_path=args.qalign_path,
        normalize_scores=True,
        normalization_stats_path=args.stats_json,
    )

    # Step 3: Load RestorationToolkit
    print('\nLoading RestorationToolkit on {}...'.format(args.device))
    from restoration_tools.agent_tools import RestorationToolkit

    toolkit = RestorationToolkit(
        models=actions_to_run,
        device=args.device,
        load_iqa=False,
        preload=True,
        auto_unload=False,
    )

    # Step 4: Run all actions on sampled images and collect IQA deltas
    # For each (degradation_type, action, image), we compute:
    #   degraded_scores = scorer.get_iqa_score(original_image)
    #   restored_scores = scorer.get_iqa_score(restored_image)
    #   delta = restored_scores - degraded_scores  (per-metric, normalized z-score space)
    print('\nRunning restoration tools and computing IQA deltas...')
    temp_dir = tempfile.mkdtemp(prefix='iqa_weight_v2_')

    # Structure: deltas[degradation_type][action] = list of delta vectors
    deltas: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    # Also store degraded scores for v1-style deficit computation
    degraded_scores_by_type: dict[str, list[list[float]]] = defaultdict(list)

    total_runs = total_images * len(actions_to_run)
    pbar = tqdm(total=total_runs, desc='Processing')

    for degradation_type, image_paths in sorted(paths_by_type.items()):
        for image_path in image_paths:
            # Compute degraded (original) IQA scores
            try:
                degraded_scores = scorer.get_iqa_score(image_path)
                degraded_scores_by_type[degradation_type].append(degraded_scores)
            except Exception as e:
                print(f'  WARNING: IQA scoring failed for {image_path}: {e}')
                pbar.update(len(actions_to_run))
                continue

            # Run each restoration action
            for action in actions_to_run:
                try:
                    output_subdir = os.path.join(temp_dir, f'{degradation_type}_{action}')
                    os.makedirs(output_subdir, exist_ok=True)

                    result = toolkit.process_image(
                        tools=[action],
                        img_path=image_path,
                        output_dir=output_subdir,
                        is_identify=True,
                    )
                    output_path = result.get('output_path')
                    if not output_path or not os.path.exists(output_path):
                        pbar.update(1)
                        continue

                    restored_scores = scorer.get_iqa_score(output_path)
                    delta = [restored_scores[i] - degraded_scores[i] for i in range(5)]
                    deltas[degradation_type][action].append(delta)

                except Exception as e:
                    print(f'  WARNING: {action} failed on {image_path}: {e}')

                pbar.update(1)

    pbar.close()

    # Step 5: Compute weight vectors using discriminability
    print('\nComputing discriminability-based weight vectors...')
    weight_map = {}
    details = {}

    for degradation_type in sorted(paths_by_type.keys()):
        targeted_actions = DEGRADATION_TARGETED_ACTIONS.get(degradation_type, [])

        # Gather targeted deltas
        targeted_delta_rows = []
        for action in targeted_actions:
            if action in deltas[degradation_type]:
                targeted_delta_rows.extend(deltas[degradation_type][action])

        # Gather generic deltas
        generic_delta_rows = []
        for action in GENERIC_ACTIONS:
            if action in deltas[degradation_type]:
                generic_delta_rows.extend(deltas[degradation_type][action])

        targeted_deltas = np.array(targeted_delta_rows, dtype=np.float64) if targeted_delta_rows else np.zeros((0, 5))
        generic_deltas = np.array(generic_delta_rows, dtype=np.float64) if generic_delta_rows else np.zeros((0, 5))

        result = compute_discriminability_weights(targeted_deltas, generic_deltas, args.uniform_mix)
        weight_map[degradation_type] = result['final_weight']

        # Compute per-action summary
        action_summary = {}
        for action in actions_to_run:
            action_deltas = deltas[degradation_type].get(action, [])
            if action_deltas:
                action_matrix = np.array(action_deltas, dtype=np.float64)
                action_summary[action] = {
                    'count': len(action_deltas),
                    'mean_delta': action_matrix.mean(axis=0).tolist(),
                    'std_delta': action_matrix.std(axis=0).tolist(),
                    'is_targeted': action in targeted_actions,
                    'is_generic': action in GENERIC_ACTIONS,
                }
            else:
                action_summary[action] = {
                    'count': 0,
                    'mean_delta': [0.0] * 5,
                    'std_delta': [0.0] * 5,
                    'is_targeted': action in targeted_actions,
                    'is_generic': action in GENERIC_ACTIONS,
                }

        details[degradation_type] = {
            'sample_count': len(paths_by_type[degradation_type]),
            'targeted_actions': targeted_actions,
            'generic_actions': GENERIC_ACTIONS,
            'n_targeted_deltas': int(targeted_deltas.shape[0]),
            'n_generic_deltas': int(generic_deltas.shape[0]),
            'metrics': {
                metric_name: {
                    'targeted_mean_delta': float(result['targeted_mean_delta'][idx]),
                    'generic_mean_delta': float(result['generic_mean_delta'][idx]),
                    'discriminability': float(result['discriminability'][idx]),
                    'positive_signal': float(result['positive_signal'][idx]),
                    'importance': float(result['importance'][idx]),
                    'combined_signal': float(result['combined_signal'][idx]),
                    'data_weight': float(result['data_weight'][idx]),
                    'final_weight': float(result['final_weight'][idx]),
                }
                for idx, metric_name in enumerate(_METRIC_NAMES)
            },
            'action_summary': action_summary,
        }

    # Step 6: Save output
    payload = {
        'version': 2,
        'method': 'discriminability_targeted_vs_generic_with_uniform_shrinkage',
        'normalization_stats_path': str(Path(args.stats_json).expanduser().resolve()),
        'metric_order': list(_METRIC_NAMES),
        'uniform_mix': float(args.uniform_mix),
        'default_weight': _UNIFORM_WEIGHT.tolist(),
        'weights': weight_map,
        'details': details,
        'source': {
            'parquet': args.parquet,
            'samples_per_type': args.samples_per_type,
            'device': args.device,
            'iqa_device': args.iqa_device,
            'actions': actions_to_run,
            'total_images': total_images,
            'seed': args.seed,
        },
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print('\n=== v2 IQA weight map (discriminability-based) ===')
    for degradation_type, weights in sorted(weight_map.items()):
        names = list(_METRIC_NAMES)
        print(f'  {degradation_type}: {dict(zip(names, [round(w, 4) for w in weights]))}')
    print(f'\nSaved weight map to {output_path}')

    # Cleanup temp directory
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass

    print('\nDone! To use this weight map, update your tool config:')
    print(f'  iqa_weight_map_path: {output_path}')


if __name__ == '__main__':
    main()