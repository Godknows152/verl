# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""
将图像修复数据集转换为verl multi-turn强化学习训练格式

输入数据格式 (原parquet):
- images: 图像路径 (可能是numpy数组或字符串)
- problem: 提示词
- answer: (忽略)

输出数据格式 (verl multi-turn):
- data_source: 数据来源标识
- prompt: 对话消息列表 [{"role": "system/user", "content": "..."}]
- extra_info: 额外信息，包含工具配置和交互参数
"""

import argparse
import io
import os

import datasets
import pandas as pd
from PIL import Image


def detect_degradation_type(image_path: str) -> str:
    """
    从图像路径或文件名中自动检测退化类型
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        检测到的退化类型 (night/rain_streak/rain_drop/rain_drive/snow/fog/unknown)
    """
    if not image_path:
        return "unknown"
    
    path_lower = image_path.lower()
    
    # 按退化类型关键词匹配
    if any(kw in path_lower for kw in ['night', 'dark', 'low_light', 'lowlight', 'lol']):
        return 'night'
    elif any(kw in path_lower for kw in ['rain_drop', 'raindrop', 'drop']):
        return 'rain_drop'
    elif any(kw in path_lower for kw in ['rain_streak', 'rainstreak', 'streak']):
        return 'rain_streak'
    elif any(kw in path_lower for kw in ['rain_drive', 'driving', 'drive']):
        return 'rain_drive'
    elif any(kw in path_lower for kw in ['rain']):  # 默认雨类型
        return 'rain_streak'
    elif any(kw in path_lower for kw in ['snow']):
        return 'snow'
    elif any(kw in path_lower for kw in ['fog', 'haze', 'hazy']):
        return 'fog'
    else:
        return 'unknown'


def create_system_prompt() -> str:
    """创建系统提示词"""
    return """You are an intelligent image restoration assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <explanation> </explanation> and <answer> </answer> tags, respectively, i.e., <explanation> reasoning process here </explanation><answer> answer here </answer>"""


def load_image_as_bytes(image_path: str) -> dict:
    """
    加载图像并转换为bytes格式（verl dataset期望的格式）
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        包含bytes的字典 {"bytes": image_bytes}
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return {"bytes": buffer.getvalue()}
    except Exception as e:
        print(f"Warning: Failed to load image {image_path}: {e}")
        # 返回一个1x1的占位图像
        placeholder = Image.new("RGB", (1, 1), color="black")
        buffer = io.BytesIO()
        placeholder.save(buffer, format="PNG")
        return {"bytes": buffer.getvalue()}


def make_map_fn(data_source: str, system_prompt: str):
    """创建数据转换函数"""
    def process_fn(example, idx):
        # 处理images列：可能是numpy数组、列表或字符串
        raw_image = example['images']
        if hasattr(raw_image, '__getitem__') and not isinstance(raw_image, str):
            # 如果是数组或列表，取第一个元素
            image_path = str(raw_image[0]) if len(raw_image) > 0 else ""
            image_paths = [str(raw_image[i]) for i in range(len(raw_image))]
        else:
            image_path = str(raw_image)
            image_paths = [image_path]
        
        # 加载图像为bytes格式（verl期望的格式）
        images_list = [load_image_as_bytes(p) for p in image_paths]
        
        user_prompt = example['problem']
        
        # 检测退化类型
        degradation_type = detect_degradation_type(image_path)
        
        # IMPORTANT: 在用户消息中添加 <image> 占位符
        # rl_dataset.py 的 _build_messages 会查找 <image> 并替换为实际图像
        # 没有这个占位符，images列中的图像将不会被传递给模型！
        
        # 检查 user_prompt 中已有的 <image> 占位符数量
        existing_placeholders = user_prompt.count("<image>")
        needed_placeholders = len(images_list) - existing_placeholders
        
        if needed_placeholders > 0:
            # 需要额外添加占位符
            image_placeholders = "<image>" * needed_placeholders
            user_content = f"{image_placeholders}\n{user_prompt}"
        else:
            # 已有足够的占位符，或者占位符多于图像（保持原样）
            user_content = user_prompt
        
        # 构建verl格式数据 (与geo3k_multiturn_w_tool.py保持一致)
        data = {
            "data_source": data_source,
            # agent_name指定使用tool_agent处理多轮工具调用
            "agent_name": "tool_agent",
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,  # 包含 <image> 占位符
                },
            ],
            "images": images_list,  # 单独的图像列表，会被 _build_messages 插入到 <image> 位置
            # reward_model字段是reward manager计算奖励所需的
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
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "restore_image": {
                        "create_kwargs": {
                            "image_path": image_path,
                        },
                    },
                },
                "interaction_kwargs": {
                    # name字段用于选择interaction实例
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
):
    """
    转换数据集
    
    Args:
        input_parquet: 输入parquet文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        data_source: 数据来源标识
    """
    # 读取原始数据为HuggingFace Dataset
    df = pd.read_parquet(input_parquet)
    print(f"读取到 {len(df)} 条数据")
    
    # 验证必需的列
    required_columns = ['images', 'problem']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"缺少必需的列: {col}")
    
    # 转换为HuggingFace Dataset
    dataset = datasets.Dataset.from_pandas(df)
    
    # 应用转换函数
    system_prompt = create_system_prompt()
    dataset = dataset.map(
        function=make_map_fn(data_source, system_prompt),
        with_indices=True,
        remove_columns=dataset.column_names,  # 移除原始列
    )
    # 获取输入文件的基础名（不含扩展名）
    input_basename = os.path.splitext(os.path.basename(input_parquet))[0]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为parquet（使用输入文件名作为基础）
    if train_ratio >= 1.0:
        # 不分割，直接输出同名文件
        output_path = os.path.join(output_dir, f"{input_basename}.parquet")
        dataset.to_parquet(output_path)
        print(f"输出: {len(dataset)} 条 -> {output_path}")
    else:
        # 按比例分割训练集和测试集
        train_size = int(len(dataset) * train_ratio)
        train_dataset = dataset.select(range(train_size))
        test_dataset = dataset.select(range(train_size, len(dataset)))
        
        # 分割为train和test
        train_path = os.path.join(output_dir, f"{input_basename}_train.parquet")
        test_path = os.path.join(output_dir, f"{input_basename}_test.parquet")
        
        train_dataset.to_parquet(train_path)
        test_dataset.to_parquet(test_path)
        
        print(f"训练集: {len(train_dataset)} 条 -> {train_path}")
        print(f"测试集: {len(test_dataset)} 条 -> {test_path}")
    
    # 打印退化类型统计
    print("\n退化类型统计:")
    degradation_counts = {}
    for item in dataset:
        dtype = item['extra_info']['degradation_type']
        degradation_counts[dtype] = degradation_counts.get(dtype, 0) + 1
    for dtype, count in sorted(degradation_counts.items()):
        print(f"  {dtype}: {count}")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="将图像修复数据集转换为verl训练格式")
    parser.add_argument(
        "--input_parquet",
        type=str,
        required=True,
        help="输入parquet文件路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/restoration",
        help="输出目录 (默认: ~/data/restoration)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="训练集比例 (默认: 0.9)",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="restoration",
        help="数据来源标识 (默认: restoration)",
    )
    
    args = parser.parse_args()
    
    # 展开用户目录
    output_dir = os.path.expanduser(args.output_dir)
    
    convert_dataset(
        input_parquet=args.input_parquet,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        data_source=args.data_source,
    )


if __name__ == "__main__":
    main()
