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

## 3. 当前训练架构（图像修复多轮 GRPO）

本仓库当前主线是：
以 Qwen3-VL 作为策略模型，在多轮对话中调用图像修复工具，基于 IQA 指标增益进行 GRPO 强化学习。

### 3.1 架构总览

训练链路分为 6 层：

1. 数据层：原始样本转换为多轮工具调用格式（含 image 与 tools_kwargs）。
2. 调度层：Hydra + Ray 启动 actor/ref/rollout 与 agent loop。
3. 生成层：SGLang 负责 Qwen3-VL 推理与采样。
4. 工具层：restore_image 工具将 action 路由到 12 个修复模型之一。
5. 奖励层：根据 IQA 分数变化计算 step reward，并返回 tool feedback。
6. 优化层：GRPO + KL 约束更新 actor，周期性评测与保存。

### 3.2 核心文件与职责

| 组件 | 位置 | 职责 |
|------|------|------|
| 数据转换 | examples/data_preprocess/convert_restoration_dataset.py | 注入 hermes 工具调用样式及 per-sample tools_kwargs |
| 主训练配置 | examples/sglang_multiturn/config/restoration_multiturn_grpo.yaml | 定义 data/algorithm/actor_rollout_ref/trainer 全局参数 |
| 工具配置 | examples/sglang_multiturn/config/tool_config/restoration_tool_config.yaml | 声明 restore_image 的 schema、device、IQA 参数 |
| 工具实现 | verl/tools/restoration_tool.py | action 校验、模型调用、IQA 奖励、反馈文本生成 |
| 模型工具箱 | restoration_tools/agent_tools/restoration_toolkit.py | 12 个修复模型的加载、推理、按需卸载 |
| 启动脚本 | examples/sglang_multiturn/run_qwen3_vl_restoration.sh | 训练入口，传递模型路径与运行参数 |

### 3.3 数据与样本结构

数据经转换后，样本需要满足：

1. 多轮对话格式，支持 hermes 工具调用标签。
2. extra_info.need_tools_kwargs=true。
3. extra_info.tools_kwargs.restore_image.create_kwargs 至少包含：
  image_path、degradation_type。
4. return_raw_chat=true，return_multi_modal_inputs=false（工具回合会动态注入图像）。

degradation_type 用于选择 IQA 指标权重（night/rain/fog/snow 等），直接影响 reward 计算。

### 3.4 多轮工具调用协议（Hermes）

1. 工具函数名固定为 restore_image。
2. 参数对象只接受 action 字段。
3. action 白名单（12+1）：
  real_esrgan、scunet、retinexformer_fivek、hvicidnet、lightdiff、
  turbo_rain、s2former、idt、ridcp、kanet、turbo_snow、snowmaster、stop。
4. 非白名单 action 会返回 invalid_action，并给负奖励。

重要：模型名（例如 scunet）是 action 值，不是 hermes 的 function.name。

### 3.5 奖励与反馈机制

当前 RestorationTool 奖励逻辑：

1. 每一步计算当前图像 IQA：[QAlign, MANIQA, MUSIQ, CLIPIQA, NIQE]。
2. 使用 degradation_type 对应权重向量做加权。
3. reward = alpha * 边际提升 + (1 - alpha) * 相对原图提升。
4. 再乘 reward_scale 并裁剪到 [-10, 10]。
5. 早停规则：step<4 时调用 stop 会给明显惩罚，step>=4 时 stop 奖励转为中性。

这套机制的目标是防止“立即停止”或“完全不调工具”的策略塌缩。

### 3.6 训练与部署拓扑（当前默认）

1. rollout 引擎为 sglang，multi_turn.enable=true，format=hermes。
2. 算法为 GRPO（adv_estimator=grpo），要求 rollout.n>1。
3. actor 启用 KL loss（与 ref 联合约束）。
4. 常见部署是 4 卡：SGLang 占主要显存，工具/IQA 复用同机剩余显存。
5. 工具侧推荐 preload=false + auto_unload=true，降低与 SGLang 的显存冲突概率。

显存抖动或 Ray worker 超时时，优先检查 gpu_memory_utilization、tool preload/auto_unload、iqa_device 与 device 绑定关系。

### 3.7 关键训练超参（以当前配置为准）

1. actor lr: 1e-5。
2. entropy_coeff: 0.2。
3. kl_coef / kl_loss_coef: 0.05。
4. max_prompt_length: 2048。
5. max_response_length: 4096。
6. total_epochs: 5。
7. reward_scale（工具配置）: 5.0。

### 3.8 观测指标与故障信号

至少持续跟踪：

1. timing_s/agent_loop/tool_calls/mean。
2. num_turns/mean。
3. invalid_action 频率。
4. unknown tool 或 schema 解析错误。
5. 回答长度分布（过短常对应 no-tool 或早停策略）。

若 tool_calls 与 num_turns 同时快速下降，通常意味着策略开始规避工具调用。

### 3.9 推荐训练阶段

建议采用两阶段：

1. 先 SFT 冷启动：学习稳定的工具名与参数格式。
2. 再 GRPO：优化“何时用何工具”的策略质量。

工具名正确率不足、invalid_action 高频或回合数塌缩时，不建议直接加大 RL 奖励，应先回到 SFT 数据与格式质量。

详情请参阅 [docs/contributing/image-restoration-guide.md](docs/contributing/image-restoration-guide.md)。

## 致谢

改编自 [vLLM 项目](https://github.com/vllm-project/vllm)的 [`AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/AGENTS.md)。
