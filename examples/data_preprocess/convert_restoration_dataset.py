# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""
将图像修复数据集转换为 verl multi-turn 强化学习训练格式。

输入数据格式 (原始 parquet):
- images: 图像路径 (numpy 数组、列表或字符串)
- problem: 提示词
- answer: (忽略)

输出数据格式 (verl multi-turn):
- data_source: 数据来源标识
- agent_name: "tool_agent"
- prompt: 对话消息列表 [{"role": "system/user", "content": "..."}]
- images: 图像 bytes 列表
- reward_model: 奖励模型配置
- extra_info: 额外信息，包含工具配置和交互参数
"""

import argparse
import io
import os

import datasets
import pandas as pd
from PIL import Image


# 退化类型关键词映射
_DEGRADATION_KEYWORDS = {
    'night': ['night', 'dark', 'low_light', 'lowlight', 'lol'],
    'rain_drop': ['rain_drop', 'raindrop', 'drop'],
    'rain_streak': ['rain_streak', 'rainstreak', 'streak'],
    'rain_drive': ['rain_drive', 'driving', 'drive'],
    'rain': ['rain'],     # 兜底 rain -> rain_streak
    'snow': ['snow'],
    'fog': ['fog', 'haze', 'hazy'],
}


def detect_degradation_type(image_path: str) -> str:
    """从图像路径中自动检测退化类型。

    Returns:
        night / rain_streak / rain_drop / rain_drive / snow / fog / unknown
    """
    if not image_path:
        return "unknown"
    path_lower = image_path.lower()
    for deg_type, keywords in _DEGRADATION_KEYWORDS.items():
        if any(kw in path_lower for kw in keywords):
            # 'rain' 兜底映射到 rain_streak
            return 'rain_streak' if deg_type == 'rain' else deg_type
    return "unknown"


def create_system_prompt() -> str:
    return (
        "You are an intelligent image restoration assistant. "
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then "
        "provides the user with the answer. "
        "The reasoning process and answer are enclosed within <explanation> </explanation> "
        "and <answer> </answer> tags, respectively, "
        "i.e., <explanation> reasoning process here </explanation>"
        "<answer> answer here </answer>"
    )


def load_image_as_bytes(image_path: str) -> dict:
    """Load an image from disk and return ``{"bytes": raw_png_bytes}``."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return {"bytes": buf.getvalue()}
    except Exception as e:
        print(f"Warning: failed to load {image_path}: {e}")
        placeholder = Image.new("RGB", (1, 1), color="black")
        buf = io.BytesIO()
        placeholder.save(buf, format="PNG")
        return {"bytes": buf.getvalue()}


def make_map_fn(data_source: str, system_prompt: str):
    """Return a HuggingFace Dataset .map() function."""

    def process_fn(example, idx):
        # Normalise images field — may be ndarray, list, or string
        raw_image = example['images']
        if hasattr(raw_image, '__getitem__') and not isinstance(raw_image, str):
            image_path = str(raw_image[0]) if len(raw_image) > 0 else ""
            image_paths = [str(raw_image[i]) for i in range(len(raw_image))]
        else:
            image_path = str(raw_image)
            image_paths = [image_path]

        # Load images as bytes (verl dataset format)
        images_list = [load_image_as_bytes(p) for p in image_paths]

        user_prompt = example['problem']
        degradation_type = detect_degradation_type(image_path)

        # Ensure each image has an <image> placeholder in the user message.
        # verl's rl_dataset._build_messages replaces <image> tokens with actual images.
        existing_placeholders = user_prompt.count("<image>")
        missing = len(images_list) - existing_placeholders
        if missing > 0:
            user_content = "<image>" * missing + "\n" + user_prompt
        else:
            user_content = user_prompt

        data = {
            "data_source": data_source,
            # tool_agent handles multi-turn tool-calling rollout
            "agent_name": "tool_agent",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            "images": images_list,
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "image_path": image_path,
                    "degradation_type": degradation_type,
                },
            },
            "extra_info": {
                "index": idx,
                "image_path": image_path,
                "degradation_type": degradation_type,
                # Signals to the agent framework that per-sample tool kwargs are present
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    # Key must match the tool_schema function name ("restore_image")
                    "restore_image": {
                        "create_kwargs": {
                            "image_path": image_path,
                        },
                    },
                },
                "interaction_kwargs": {
                    # "name" selects which Interaction class to instantiate
                    "name": "image_restoration",
                    "original_image": image_path,
                    "image_path": image_path,
                    "degradation_type": degradation_type,
                },
            },
        }
        return data

    return process_fn


def convert_dataset(
    input_parquet: str,
    output_dir: str,
    train_ratio: float = 0.9,
    data_source: str = "restoration",
) -> datasets.Dataset:
    """Convert a raw restoration parquet into verl multi-turn format.

    Args:
        input_parquet: Path to input parquet file.
        output_dir: Directory to write output parquet(s).
        train_ratio: Fraction used for training (rest goes to test). Use 1.0 to skip split.
        data_source: String label stored in the ``data_source`` field.

    Returns:
        The converted HuggingFace Dataset.
    """
    df = pd.read_parquet(input_parquet)
    print(f"Read {len(df)} rows from {input_parquet}")

    for col in ('images', 'problem'):
        if col not in df.columns:
            raise ValueError(f"Input parquet is missing required column: '{col}'")

    dataset = datasets.Dataset.from_pandas(df)
    system_prompt = create_system_prompt()
    dataset = dataset.map(
        function=make_map_fn(data_source, system_prompt),
        with_indices=True,
        remove_columns=dataset.column_names,
    )

    input_basename = os.path.splitext(os.path.basename(input_parquet))[0]
    os.makedirs(output_dir, exist_ok=True)

    if train_ratio >= 1.0:
        out_path = os.path.join(output_dir, f"{input_basename}.parquet")
        dataset.to_parquet(out_path)
        print(f"Saved {len(dataset)} rows -> {out_path}")
    else:
        split = int(len(dataset) * train_ratio)
        train_ds = dataset.select(range(split))
        test_ds  = dataset.select(range(split, len(dataset)))

        train_path = os.path.join(output_dir, f"{input_basename}_train.parquet")
        test_path  = os.path.join(output_dir, f"{input_basename}_test.parquet")

        train_ds.to_parquet(train_path)
        test_ds.to_parquet(test_path)
        print(f"Train: {len(train_ds)} rows -> {train_path}")
        print(f"Test:  {len(test_ds)}  rows -> {test_path}")

    # Print degradation-type breakdown
    print("\nDegradation type breakdown:")
    counts: dict = {}
    for item in dataset:
        dtype = item['extra_info']['degradation_type']
        counts[dtype] = counts.get(dtype, 0) + 1
    for dtype, count in sorted(counts.items()):
        print(f"  {dtype}: {count}")

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert an image restoration dataset to verl multi-turn RL format"
    )
    parser.add_argument("--input_parquet", type=str, required=True,
                        help="Input parquet file path")
    parser.add_argument("--output_dir", type=str, default="data/restoration",
                        help="Output directory (default: data/restoration)")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Training split ratio (default: 0.9; use 1.0 to skip split)")
    parser.add_argument("--data_source", type=str, default="restoration",
                        help="data_source label written to each row (default: restoration)")
    args = parser.parse_args()

    convert_dataset(
        input_parquet=args.input_parquet,
        output_dir=os.path.expanduser(args.output_dir),
        train_ratio=args.train_ratio,
        data_source=args.data_source,
    )


if __name__ == "__main__":
    main()
