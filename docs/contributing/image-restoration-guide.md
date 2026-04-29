# 图像修复领域指南

> **在修改图像修复流水线中的任何代码之前，请先阅读本指南。**
> AGENTS.md 引用了本文件；与以下规则冲突的修改必须拒绝执行。

## 概述

本领域为**图像修复**任务新增了多轮 GRPO 强化学习支持，以 Qwen3-VL 作为策略模型。
智能体基于观察到的退化类型迭代选择修复工具，并通过 IQA（图像质量评估）指标获得奖励。

### 关键目录与文件

| 路径 | 用途 |
|------|------|
| `restoration_tools/agent_tools/` | 修复模型封装 + IQA 评分 |
| `restoration_tools/agent_tools/restoration_toolkit.py` | `RestorationToolkit` — 模型中央注册表 |
| `restoration_tools/agent_tools/iqa_reward.py` | `IQAScore` — QAlign / MANIQA / MUSIQ / CLIPIQA / NIQE |
| `restoration_tools/checkpoints/` | 预训练 IQA 模型权重（Q-Align） |
| `restoration_tools/dependences/` | 第三方依赖（BasicSR、IQA-PyTorch 等） |
| `verl/tools/restoration_tool.py` | verl `BaseTool` 封装 — 注册 `restore_image` 函数 |
| `verl/interactions/image_restoration_interaction.py` | `ImageRestorationInteraction` — 奖励计算 |
| `examples/data_preprocess/convert_restoration_dataset.py` | 数据集转换（原始 parquet → verl 多轮格式） |
| `examples/sglang_multiturn/config/restoration_multiturn_grpo.yaml` | Hydra 主训练配置 |
| `examples/sglang_multiturn/config/tool_config/restoration_tool_config.yaml` | 工具注册配置 |
| `examples/sglang_multiturn/config/interaction_config/restoration_interaction_config.yaml` | 交互 / 奖励配置 |
| `examples/sglang_multiturn/run_qwen3_vl_restoration.sh` | 一键训练启动脚本 |

---

## 快速上手

### 1. 安装依赖

```bash
# 修复模型核心依赖
cd restoration_tools/dependences
pip install -e BasicSR/
pip install -e IQA-PyTorch/

# IQA 模型权重（Q-Align）需放置于：
#   restoration_tools/checkpoints/q_align/
```

### 2. 构建训练数据集

```bash
python examples/data_preprocess/convert_restoration_dataset.py \
  --input_parquet /path/to/raw_dataset.parquet \
  --output_dir data/restoration \
  --train_ratio 0.9
```

脚本会从图像路径关键词自动检测退化类型（`night`、`rain_streak`、`rain_drop`、`rain_drive`、`snow`、`fog`）。

### 3. 启动训练

```bash
# 设置环境变量（或直接修改脚本）
export QWEN3_VL_MODEL_PATH="/path/to/Qwen3-VL-7B-Instruct"
bash examples/sglang_multiturn/run_qwen3_vl_restoration.sh
```

---

## 架构

```
Rollout 循环（SGLang）
  │
  ├─ 策略模型（Qwen3-VL）选择动作：<answer>tool_name</answer>
  │
  ├─ RestorationTool（verl/tools/restoration_tool.py）
  │    └─ RestorationToolkit（restoration_tools/agent_tools/）
  │         └─ 12 个修复模型（ESRGAN、SCUNet、Retinexformer……）
  │
  └─ ImageRestorationInteraction（verl/interactions/image_restoration_interaction.py）
       └─ IQAScore（QAlign + MANIQA + MUSIQ + CLIPIQA + NIQE）
            └─ reward = α · Δ(本步) + (1-α) · Δ(相对原图)
```

### 支持的修复动作

| 动作 | 任务 |
|------|------|
| `real_esrgan` | 超分辨率、去模糊、去噪、压缩伪影去除 |
| `scunet` | 高质量图像去噪 |
| `retinexformer_fivek` | 低光照增强 |
| `hvicidnet` | 低光照 / 曝光校正 |
| `lightdiff` | 低光照增强（扩散模型） |
| `turbo_rain` | 快速去雨 |
| `s2former` | 雨纹去除 |
| `idt` | 去雨 / 雨滴去除 |
| `ridcp` | 图像去雾 |
| `kanet` | 图像去雾 |
| `turbo_snow` | 图像去雪 |
| `snowmaster` | 高级去雪 |
| `stop` | 提前终止修复流程 |

### 各退化类型的 IQA 评分权重

权重格式为 `[qalign, maniqa, musiq, clipiqa, niqe]`，定义于
`verl/interactions/image_restoration_interaction.py::SCORE_WEIGHT_MAP`。

| 退化类型 | qalign | maniqa | musiq | clipiqa | niqe |
|---------|--------|--------|-------|---------|------|
| night（夜晚） | 2/9 | 2/9 | 0 | 2/9 | 3/9 |
| rain_streak（雨纹） | 1/5 | 1.25/5 | 1/5 | 0.75/5 | 1/5 |
| rain_drop（雨滴） | 0 | 0.5/3 | 0 | 1.25/3 | 1.25/3 |
| rain_drive（行车雨天） | 0.5/4 | 1.5/4 | 1/4 | 1/4 | 0 |
| snow（雪） | 1.5/5 | 0.75/5 | 1/5 | 0.75/5 | 1/5 |
| fog（雾/霾） | 1.5/5 | 0.5/5 | 1.5/5 | 0.5/5 | 1/5 |

---

## GPU 显存布局

以当前 4 张 GPU 配置为例：

- **GPU 0–3**：SGLang rollout
- **GPU 0–2**：额外承载 RestorationToolkit 中的修复模型（`model_devices: [cuda:0, cuda:1, cuda:2]`）
- **GPU 3**：额外承载 IQA 模型（`iqa_device: cuda:3`），因此通常是最先出现显存压力的位置

为避免与 SGLang memory-saver 产生冲突，修复模型采用**懒加载**策略：
- `preload: false` — 按需加载模型
- `auto_unload: true` — 每次推理后自动卸载模型

---

## 本领域的修改规则

1. **新增修复模型**：在 `restoration_tools/agent_tools/` 下以子包形式添加，包含 `inference.py`，
   在 `RestorationToolkit` 中注册，在 `restoration_tool.py` 的 `ALLOWED_ACTIONS` 中添加，
   并更新工具 schema 描述。

2. **修改奖励计算**：更新 `SCORE_WEIGHT_MAP` 并记录修改理由。
   任何对 `alpha`（单步奖励与绝对奖励平衡系数）的调整，PR 说明中必须包含消融实验结果。

3. **修改数据集格式**：更新 `convert_restoration_dataset.py` **并**
   验证 `ImageRestorationInteraction` 仍能正确解析 `extra_info` 字段。

4. **修改 GPU 分配**：必须同时更新 `restoration_tool_config.yaml`（工具设备）和
   `restoration_interaction_config.yaml`（IQA 设备），保持两者一致。

5. **禁止**提交实际模型权重或大型二进制文件。`restoration_tools/checkpoints/` 目录仅用于
   路径说明文档，应提供下载脚本代替直接提交文件。
