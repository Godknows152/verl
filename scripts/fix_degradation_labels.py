"""
原地修正 data/restoration/train.parquet 和 test.parquet 中的 degradation_type 标签。

根本原因：旧版 detect_degradation_type 使用裸 'rain' 关键字，
误匹配路径中的 /train/ 目录名，导致 fog/snow 图像全部被标为 rain_streak。

本脚本会：
1. 重新推断每条记录的 degradation_type
2. 更新以下三处字段：
   - extra_info.degradation_type
   - extra_info.tools_kwargs.restore_image.create_kwargs.degradation_type
   - reward_model.ground_truth.degradation_type
3. 保存前先备份原文件 (.bak)
4. 打印修正前后的标签分布
"""
import os
import shutil
from collections import Counter

import pandas as pd

# 修正后的关键词映射（fog/snow 在 rain 兜底之前）
_DEGRADATION_KEYWORDS = {
    'night':       ['night', 'dark', 'low_light', 'lowlight', 'lol'],
    'rain_drop':   ['rain_drop', 'raindrop'],
    'rain_streak': ['rain_streak', 'rainstreak', 'streak'],
    'rain_drive':  ['rain_drive', 'driving', 'drive'],
    'snow':        ['snow'],
    'fog':         ['fog', 'haze', 'hazy'],
    'rain':        ['rain_series', '/rain/'],   # 避免误匹配 /train/
}


def detect_degradation_type(image_path: str) -> str:
    if not image_path:
        return 'unknown'
    p = image_path.lower()
    for deg_type, keywords in _DEGRADATION_KEYWORDS.items():
        if any(kw in p for kw in keywords):
            return 'rain_streak' if deg_type == 'rain' else deg_type
    return 'unknown'


def fix_parquet(path: str) -> None:
    # 备份
    bak = path + '.bak'
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"  Backed up -> {bak}")
    else:
        print(f"  Backup already exists: {bak}")

    df = pd.read_parquet(path)
    before = Counter(row.get('degradation_type', 'MISSING') for row in df['extra_info'])

    fixed_extra_info = []
    fixed_reward_model = []
    fix_count = 0

    for _, row in df.iterrows():
        ei = dict(row['extra_info'])
        rm = dict(row['reward_model'])

        image_path = ei.get('image_path', '')
        new_type = detect_degradation_type(image_path)
        old_type = ei.get('degradation_type', '')

        if old_type != new_type:
            fix_count += 1

        # 更新 extra_info
        ei['degradation_type'] = new_type
        tools_kwargs = ei.get('tools_kwargs', {})
        restore = tools_kwargs.get('restore_image', {})
        create_kwargs = restore.get('create_kwargs', {})
        create_kwargs['degradation_type'] = new_type
        restore['create_kwargs'] = create_kwargs
        tools_kwargs['restore_image'] = restore
        ei['tools_kwargs'] = tools_kwargs

        # 更新 reward_model.ground_truth
        gt = rm.get('ground_truth', {})
        if isinstance(gt, dict):
            gt['degradation_type'] = new_type
            rm['ground_truth'] = gt

        fixed_extra_info.append(ei)
        fixed_reward_model.append(rm)

    df['extra_info'] = fixed_extra_info
    df['reward_model'] = fixed_reward_model

    after = Counter(row.get('degradation_type', 'MISSING') for row in df['extra_info'])

    df.to_parquet(path, index=False)

    print(f"\n  File: {os.path.basename(path)}  ({len(df)} rows, {fix_count} fixed)")
    print(f"  Before: {dict(before)}")
    print(f"  After : {dict(after)}")


if __name__ == '__main__':
    base = '/home/LXJ/Python_Projects/verl/data/restoration'
    for name in ('train.parquet', 'test.parquet'):
        print(f"\n{'='*60}")
        fix_parquet(os.path.join(base, name))
    print('\nDone.')
