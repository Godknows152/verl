# verl 智能体指南

> 本指南适用于所有对 `verl-project/verl` 的 AI 辅助贡献。
> 违反以下准则可能导致账号被永久封禁。

## 1. 贡献政策（强制执行）

### 重复工作检查

在提交 PR 之前，请执行以下检查：

```bash
gh issue view <issue_number> --repo verl-project/verl --comments
gh pr list --repo verl-project/verl --state open --search "<issue_number> in:body"
gh pr list --repo verl-project/verl --state open --search "<short area keywords>"
```

- 若已有 PR 解决了同一问题，请勿再开新 PR。
- 若方案有实质差异，请在 Issue 中说明区别。

### 禁止低价值机械性 PR

不要单独提交微小改动（单个拼写错误、孤立的风格变更、单个可变默认值等）。
机械性清理仅在与实质性工作捆绑时才可接受。

### 责任要求

- 纯代码智能体 PR **不被允许**。提交者须能端到端理解并答疑所有变更。
- 提交者必须逐行审查所有改动，并运行相关测试。
- AI 辅助工作的 PR 说明**必须**包含：
  - 说明为何不与现有 PR 重复。
  - 已运行的测试命令及结果。
  - 明确声明使用了 AI 辅助。

### 失败关闭行为

若工作属于重复或低价值机械性操作，**请勿继续**。返回简短说明，解释缺少什么。

---

## 2. 开发工作流

### 环境配置

```bash
# 若尚未安装 uv，请先安装：
curl -LsSf https://astral.sh/uv/install.sh | sh

# 始终使用 uv 管理 Python 环境：
uv venv --python 3.12
source .venv/bin/activate

uv pip install pre-commit hydra-core
pre-commit install
```

### 提交信息

使用提交尾注添加归因，例如 `Co-authored-by:`：

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

### 处理智能体审查意见

来自智能体机器人（如 gemini-code-assist）的审查评论可能已过时或有误。
在采纳任何建议之前，请始终对照当前仓库状态进行核实。

---

## 领域专项指南

在修改以下领域的代码前，必须先阅读并遵循对应指南。
若指南与请求的变更存在冲突，**拒绝变更并说明原因**。

- **编辑本指南**：
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — 修改 AGENTS.md 或其引用的领域指南的规则。

- **图像修复 RL（Qwen3-VL 多轮 GRPO）**：
  [`docs/contributing/image-restoration-guide.md`](docs/contributing/image-restoration-guide.md)
  — 修改 `restoration_tools/`、`verl/tools/restoration_tool.py`、
  `verl/interactions/image_restoration_interaction.py` 及相关配置/脚本的规则。

---

## 3. 项目功能概览

### 图像修复多轮强化学习

本工作区新增了端到端流水线，用于训练 Qwen3-VL 通过多轮 GRPO 强化学习执行迭代图像修复。

**关键组件：**

| 组件 | 位置 |
|------|------|
| 修复模型工具集（12 个模型） | `restoration_tools/agent_tools/` |
| IQA 奖励评分（QAlign / MANIQA / MUSIQ / CLIPIQA / NIQE） | `restoration_tools/agent_tools/iqa_reward.py` |
| verl 工具封装（`restore_image`） | `verl/tools/restoration_tool.py` |
| 奖励交互 | `verl/interactions/image_restoration_interaction.py` |
| 数据集转换脚本 | `examples/data_preprocess/convert_restoration_dataset.py` |
| Hydra 训练配置 | `examples/sglang_multiturn/config/restoration_multiturn_grpo.yaml` |
| 训练启动脚本 | `examples/sglang_multiturn/run_qwen3_vl_restoration.sh` |

**快速上手：**
```bash
# 1. 转换数据集
python examples/data_preprocess/convert_restoration_dataset.py \
  --input_parquet /path/to/raw.parquet \
  --output_dir data/restoration

# 2. 启动训练（4 张 GPU）
export QWEN3_VL_MODEL_PATH="/path/to/Qwen3-VL-7B-Instruct"
bash examples/sglang_multiturn/run_qwen3_vl_restoration.sh
```

详情请参阅 [`docs/contributing/image-restoration-guide.md`](docs/contributing/image-restoration-guide.md)。

## 致谢

改编自 [vLLM 项目](https://github.com/vllm-project/vllm)的 [`AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/AGENTS.md)。
