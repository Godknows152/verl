# 多轮 Rollout 示例（GSM8K）

本示例演示如何使用 SGLang 对支持工具调用的模型（如 Qwen2.5-3B）在 GSM8K 数据集上执行**多轮 rollout**。

## 使用方法

### 第一步：下载 GSM8K 数据集

```bash
cd examples/data_preprocess
python3 gsm8k_multiturn_w_tool.py
```

脚本将自动下载并预处理 GSM8K 数据集到 ~/data/gsm8k/。

### 第二步：运行多轮 Rollout

若有 8 张 GPU，使用标准 8-GPU 脚本：

```bash
cd your_verl_root_dir
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh
```

若只有 4 张 GPU，使用备用 4-GPU 脚本：

```bash
cd your_verl_root_dir
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn_4xgpu.sh 
```

## 说明

- Rollout 支持带工具调用能力的多轮对话。
- 当前工具用于 GSM8K 答案评估。
- 未来版本可能扩展到搜索工具和代码解释器工具。

---

# 图像修复多轮 GRPO（Qwen3-VL）

本示例通过多轮 GRPO 强化学习训练 **Qwen3-VL** 对退化图像（噪声、雾霾、雨天、低光照、积雪等）进行迭代修复。

## 使用方法

### 第一步：构建数据集

```bash
python examples/data_preprocess/convert_restoration_dataset.py \
  --input_parquet /path/to/raw_dataset.parquet \
  --output_dir data/restoration \
  --train_ratio 0.9
```

退化类型（`night`、`rain_streak`、`rain_drop`、`rain_drive`、`snow`、`fog`）
将从图像文件路径关键词中自动检测。

### 第二步：启动训练（4 张 GPU）

```bash
export QWEN3_VL_MODEL_PATH="/path/to/Qwen3-VL-7B-Instruct"
bash examples/sglang_multiturn/run_qwen3_vl_restoration.sh
```

可通过追加 `key=value` 参数覆盖任意 Hydra 配置项：

```bash
bash examples/sglang_multiturn/run_qwen3_vl_restoration.sh \
  trainer.n_gpus_per_node=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5
```

## 关键文件

| 文件 | 用途 |
|------|------|
| `run_qwen3_vl_restoration.sh` | 训练启动脚本 |
| `config/restoration_multiturn_grpo.yaml` | Hydra 主配置文件 |
| `config/tool_config/restoration_tool_config.yaml` | 修复工具配置（12 个模型） |
| `config/interaction_config/restoration_interaction_config.yaml` | IQA 奖励配置 |

完整领域指南（含 GPU 布局、模型说明、修改规则）请参阅
[`docs/contributing/image-restoration-guide.md`](../../docs/contributing/image-restoration-guide.md)。
