# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Image Restoration Tool for verl multi-turn rollout.

This tool wraps the RestorationToolkit located in restoration_tools/agent_tools/.
It self-contains IQA-based reward computation (no separate interactions layer
is required), and is designed for use with the standard verl ToolAgentLoop and
the hermes tool-call format.

Key design decisions:
- IQA scoring (QAlign / MANIQA / MUSIQ / CLIPIQA / NIQE) is done inside execute()
  and the reward is returned directly as tool_reward_score.
- Termination is controlled by YAML config (max_user_turns / max_assistant_turns)
  rather than a hard-stop mechanism inside the tool.
- The tool returns a feedback text message after each step so the model can
  reason about its next action.

Supported restoration actions:
- real_esrgan: Super-resolution / deblurring / denoising / compression artifact removal
- scunet: High-quality denoising
- retinexformer_fivek: Low-light enhancement
- hvicidnet: Low-light / exposure correction
- lightdiff: Low-light enhancement (diffusion model)
- turbo_rain: Fast deraining
- s2former: Rain streak removal
- idt: Deraining / raindrop removal
- ridcp: Dehazing
- kanet: Dehazing
- turbo_snow: Desnowing
- snowmaster: Advanced desnowing
"""

import logging
import os
import sys
import torch
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)

# Always capture INFO-level messages to a dedicated file so that detailed tool
# execution info is available for debugging even when VERL_LOGGING_LEVEL=WARN.
# INFO messages are NOT forwarded to the terminal / tee log.
logger.propagate = False  # prevent INFO from leaking to the root console handler
_console_level = os.getenv("VERL_LOGGING_LEVEL", "WARN").upper()
logger.setLevel(logging.INFO)  # logger must be at INFO so file handler receives INFO msgs

_console_handler = logging.StreamHandler()
_console_handler.setLevel(getattr(logging, _console_level, logging.WARNING))
_console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logger.addHandler(_console_handler)

_log_dir = os.getenv("VERL_LOG_DIR", "/tmp")
_file_handler = logging.FileHandler(os.path.join(_log_dir, "restoration_tool_info.log"), mode="a", encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logger.addHandler(_file_handler)

# Add restoration_tools/agent_tools to sys.path for importing RestorationToolkit
AGENT_TOOLS_PATH = Path(__file__).resolve().parent.parent.parent / 'restoration_tools' / 'agent_tools'
if AGENT_TOOLS_PATH.exists() and str(AGENT_TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(AGENT_TOOLS_PATH))

# Add submodule paths (Retinexformer, SCUNet, etc.)
_RESTORATION_TOOLS_PATH = AGENT_TOOLS_PATH.parent
_submodules = [
    'Retinexformer', 'HVICIDNet', 'LightenDiffusion', 'SCUNet',
    'ESRGAN', 'IDT', 'RIDCP', 'KANet', 'S2Former', 'SnowMaster', 'img2img_turbo',
]
for _submod in _submodules:
    _submod_path = _RESTORATION_TOOLS_PATH / _submod
    if _submod_path.exists() and str(_submod_path) not in sys.path:
        sys.path.insert(0, str(_submod_path))

# Allowed restoration actions (including stop)
ALLOWED_ACTIONS = {
    'real_esrgan', 'scunet', 'retinexformer_fivek', 'hvicidnet', 'lightdiff',
    'turbo_rain', 's2former', 'idt', 'ridcp', 'kanet', 'turbo_snow', 'snowmaster', 'stop',
}

# IQA metric weights per degradation type (QAlign, MANIQA, MUSIQ, CLIPIQA, NIQE)
SCORE_WEIGHT_MAP: dict[str, list[float]] = {
    'night':       [2./9,    2./9,    0.,      2./9,    3./9   ],
    'rain_streak': [1./5,    1.25/5,  1./5,    0.75/5,  1./5   ],
    'rain_drop':   [0.,      0.5/3,   0.,      1.25/3,  1.25/3 ],
    'rain_drive':  [0.5/4,   1.5/4,  1./4,    1./4,    0.      ],
    'snow':        [1.5/5,   0.75/5,  1./5,    0.75/5,  1./5   ],
    'fog':         [1.5/5,   0.5/5,   1.5/5,   0.5/5,   1./5   ],
}
DEFAULT_WEIGHT: list[float] = [0.2, 0.2, 0.2, 0.2, 0.2]
FAILURE_REWARD = -5.0
REPEAT_ACTION_PENALTY = 0.2
REPEAT_LOW_GAIN_PENALTY = 0.8
REPEAT_LOW_GAIN_THRESHOLD = 0.05
STOP_MIN_STEP = 3
STOP_IQA_DELTA_THRESHOLD = 0.25
STOP_SUCCESS_REWARD = 3.0
STOP_PARTIAL_REWARD = 1.0
STOP_EARLY_PENALTY = -1.0
STOP_CONTINUE_PENALTY = -0.5
STOP_RECENT_REWARD_WINDOW = 2
STOP_RECENT_REWARD_THRESHOLD = 0.25

# Module-level caches
_toolkit_instance = None
_iqa_instance = None


def get_toolkit(
    device: str = 'cuda',
    models: list = None,
    preload: bool = True,
    auto_unload: bool = False,
    model_devices: list[str] | None = None,
    model_device_map: dict[str, str] | None = None,
):
    """Lazy load and cache the RestorationToolkit instance."""
    global _toolkit_instance
    if _toolkit_instance is None:
        try:
            from restoration_tools.agent_tools import RestorationToolkit
            _toolkit_instance = RestorationToolkit(
                models=models,
                device=device,
                load_iqa=False,
                preload=preload,
                auto_unload=auto_unload,
                model_devices=model_devices,
                model_device_map=model_device_map,
            )
            logger.info(
                f"RestorationToolkit initialized on {device} "
                f"(preload={preload}, auto_unload={auto_unload})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize RestorationToolkit: {e}")
            raise
    return _toolkit_instance


def get_iqa_scorer(device: str = 'cuda'):
    """Lazy load and cache the IQAScore instance."""
    global _iqa_instance
    if _iqa_instance is None:
        try:
            from iqa_reward import IQAScore

            _iqa_instance = IQAScore(device=device)
            logger.info(f"IQAScore initialized on {device}")
        except Exception as e:
            logger.error(f"Failed to initialize IQAScore: {e}")
            raise
    return _iqa_instance


def _load_restoration_tool_runtime_config(tool_config_path: str) -> dict[str, Any] | None:
    """Load runtime config for RestorationTool from tool_config yaml."""
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(tool_config_path)
        for tool_item in cfg.get("tools", []):
            if tool_item.get("class_name") == "verl.tools.restoration_tool.RestorationTool":
                return dict(tool_item.get("config", {}))
    except Exception as e:
        logger.warning(f"Failed to load tool config from {tool_config_path}: {e}")
    return None


def preload_restoration_models_for_sampling(tool_config_path: str) -> bool:
    """Preload all restoration models at sampling stage start.

    Returns True if preload path was executed (or models already loaded), False otherwise.
    """
    runtime_cfg = _load_restoration_tool_runtime_config(tool_config_path)
    if runtime_cfg is None:
        return False

    # Phase-managed mode: keep models resident during rollout; unload as a batch afterwards.
    device = runtime_cfg.get("device", "cuda")
    models = runtime_cfg.get("models", None)
    model_devices = runtime_cfg.get("model_devices", None)
    model_device_map = runtime_cfg.get("model_device_map", None)
    toolkit = get_toolkit(
        device=device,
        models=models,
        preload=False,
        auto_unload=False,
        model_devices=model_devices,
        model_device_map=model_device_map,
    )
    toolkit.auto_unload = False
    toolkit.load_models()
    logger.info("Preloaded all restoration models for sampling stage")
    return True


def unload_restoration_models_after_sampling() -> bool:
    """Unload all restoration models at sampling stage end."""
    global _toolkit_instance
    if _toolkit_instance is None:
        return False
    _toolkit_instance.unload_all_models()
    logger.info("Unloaded all restoration models after sampling stage")
    return True


class RestorationTool(BaseTool):
    """A tool for iterative image restoration / degradation removal.

    Computes IQA-based step rewards inside ``execute()`` and returns them
    directly to the verl ToolAgentLoop as ``tool_reward_score``.
    No external interactions layer is required.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict] = {}

        self.device = config.get("device", "cuda")
        self.iqa_device = config.get("iqa_device", self.device)
        self.preload_models = config.get("models", None)
        self.model_devices = config.get("model_devices", [self.device])
        self.model_device_map = config.get("model_device_map", None)
        self.output_dir = config.get("output_dir", "/tmp/verl_restoration")
        self.preload = config.get("preload", True)
        # Disable per-tool-call auto-unload mode. We use phase-managed load/unload:
        # preload once at sampling start, unload once after sampling.
        configured_auto_unload = bool(config.get("auto_unload", False))
        if configured_auto_unload:
            logger.warning(
                "auto_unload=true is ignored in phase-managed mode; forcing auto_unload=false"
            )
        self.auto_unload = False
        self.use_iqa = config.get("use_iqa", True)
        self.alpha = float(config.get("alpha", 0.9))       # marginal-improvement weight
        self.beta = 1.0 - self.alpha                       # identity-improvement weight
        self.reward_scale = float(config.get("reward_scale", 1.0))
        self.repeat_action_penalty = float(config.get("repeat_action_penalty", REPEAT_ACTION_PENALTY))
        self.repeat_low_gain_penalty = float(config.get("repeat_low_gain_penalty", REPEAT_LOW_GAIN_PENALTY))
        self.repeat_low_gain_threshold = float(config.get("repeat_low_gain_threshold", REPEAT_LOW_GAIN_THRESHOLD))
        self.stop_min_step = int(config.get("stop_min_step", STOP_MIN_STEP))
        self.stop_iqa_delta_threshold = float(config.get("stop_iqa_delta_threshold", STOP_IQA_DELTA_THRESHOLD))
        self.stop_success_reward = float(config.get("stop_success_reward", STOP_SUCCESS_REWARD))
        self.stop_partial_reward = float(config.get("stop_partial_reward", STOP_PARTIAL_REWARD))
        self.stop_early_penalty = float(config.get("stop_early_penalty", STOP_EARLY_PENALTY))
        self.stop_continue_penalty = float(config.get("stop_continue_penalty", STOP_CONTINUE_PENALTY))
        self.stop_recent_reward_window = int(config.get("stop_recent_reward_window", STOP_RECENT_REWARD_WINDOW))
        self.stop_recent_reward_threshold = float(
            config.get("stop_recent_reward_threshold", STOP_RECENT_REWARD_THRESHOLD)
        )

        os.makedirs(self.output_dir, exist_ok=True)
        self._toolkit = None
        self._iqa = None

        logger.info(
            f"RestorationTool initialized: device={self.device}, iqa_device={self.iqa_device}, "
            f"use_iqa={self.use_iqa}, alpha={self.alpha}, reward_scale={self.reward_scale}, "
            f"repeat_action_penalty={self.repeat_action_penalty}, "
            f"repeat_low_gain_penalty={self.repeat_low_gain_penalty}, "
            f"stop_min_step={self.stop_min_step}, "
            f"model_devices={self.model_devices}"
        )

    @property
    def toolkit(self):
        if self._toolkit is None:
            self._toolkit = get_toolkit(
                device=self.device,
                models=self.preload_models,
                preload=self.preload,
                auto_unload=self.auto_unload,
                model_devices=self.model_devices,
                model_device_map=self.model_device_map,
            )
        return self._toolkit

    @property
    def iqa(self):
        if self._iqa is None and self.use_iqa:
            self._iqa = get_iqa_scorer(device=self.iqa_device)
        return self._iqa

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    def _get_iqa_scores(self, image_path: str) -> list[float]:
        """Compute [QAlign, MANIQA, MUSIQ, CLIPIQA, NIQE] for an image."""
        if not self.use_iqa:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        try:
            scores = self.iqa.get_iqa_score(image_path)  # returns list of 5 floats
            return list(scores)
        except Exception as e:
            logger.warning(f"IQA scoring failed for {image_path}: {e}")
            return [0.0, 0.0, 0.0, 0.0, 0.0]

    def _count_consecutive_repeats(self, actions_history: list[str], action: str) -> int:
        """Count how many trailing actions match the current action."""
        repeat_count = 0
        for previous_action in reversed(actions_history):
            if previous_action != action:
                break
            repeat_count += 1
        return repeat_count

    def _calculate_reward(
        self,
        prev_scores: list[float],
        curr_scores: list[float],
        identity_scores: list[float],
        weights: list[float],
        action: str,
        actions_history: list[str],
    ) -> dict[str, float]:
        """Compute reward and diagnostics for a restoration step."""
        prev_t = torch.tensor(prev_scores, dtype=torch.float32)
        curr_t = torch.tensor(curr_scores, dtype=torch.float32)
        iden_t = torch.tensor(identity_scores, dtype=torch.float32)
        w_t = torch.tensor(weights, dtype=torch.float32)

        marginal = ((curr_t - prev_t) * w_t).sum().item()
        identity = ((curr_t - iden_t) * w_t).sum().item()
        mixed = self.alpha * marginal + self.beta * identity
        base_reward = mixed * self.reward_scale

        repeat_count = self._count_consecutive_repeats(actions_history, action)
        repeat_penalty = 0.0
        if repeat_count > 0:
            repeat_penalty += self.repeat_action_penalty * repeat_count
            if marginal <= self.repeat_low_gain_threshold:
                repeat_penalty += self.repeat_low_gain_penalty * repeat_count

        reward = float(torch.clamp(torch.tensor(base_reward - repeat_penalty), -10.0, 10.0).item())
        return {
            "reward": reward,
            "base_reward": float(base_reward),
            "marginal": float(marginal),
            "identity": float(identity),
            "repeat_penalty": float(repeat_penalty),
            "consecutive_action_count": float(repeat_count + 1),
        }

    def _calculate_identity_delta(
        self,
        curr_scores: list[float],
        identity_scores: list[float],
        weights: list[float],
    ) -> float:
        """Compute weighted IQA improvement over the original degraded image."""
        curr_t = torch.tensor(curr_scores, dtype=torch.float32)
        iden_t = torch.tensor(identity_scores, dtype=torch.float32)
        w_t = torch.tensor(weights, dtype=torch.float32)
        return float(((curr_t - iden_t) * w_t).sum().item())

    def _calculate_stop_reward(
        self,
        step: int,
        identity_delta: float,
        recent_rewards: list[float],
    ) -> dict[str, float | bool]:
        """Reward stopping when quality is good enough or recent gains have plateaued."""
        recent_reward_mean = float(sum(recent_rewards) / len(recent_rewards)) if recent_rewards else 0.0
        plateau = bool(recent_rewards) and recent_reward_mean <= self.stop_recent_reward_threshold
        good_enough = identity_delta >= self.stop_iqa_delta_threshold

        if step < self.stop_min_step:
            reward = self.stop_early_penalty
        elif plateau and good_enough:
            reward = self.stop_success_reward
        elif plateau or good_enough:
            reward = self.stop_partial_reward
        else:
            reward = self.stop_continue_penalty

        return {
            "reward": float(torch.clamp(torch.tensor(reward), -10.0, 10.0).item()),
            "recent_reward_mean": recent_reward_mean,
            "plateau": plateau,
            "good_enough": good_enough,
        }

    def _generate_feedback(
        self,
        action: str,
        step: int,
        reward: float,
        actions_history: list[str],
        marginal: float,
        identity_delta: float,
        consecutive_action_count: int,
    ) -> str:
        """Generate human-readable feedback for the model's next turn."""
        history_str = " → ".join(actions_history) if actions_history else "none"

        lines = [
            f"Step {step}: Applied '{action}'.",
            f"Step reward: {reward:.4f}",
            f"Weighted marginal improvement: {marginal:.4f}",
            f"Improvement over original image: {identity_delta:.4f}",
            f"Action history: {history_str}",
        ]
        if consecutive_action_count > 1:
            lines.append(
                f"Consecutive uses of '{action}': {consecutive_action_count}. "
                "Repeating the same tool without clear gains is discouraged."
            )
        if step >= self.stop_min_step and marginal <= self.repeat_low_gain_threshold:
            lines.append(
                "Recent gains are small. Consider stopping now or switch to a different targeted operation."
            )
        elif step >= self.stop_min_step:
            lines.append(
                "You have completed several restoration steps. "
                "If gains keep shrinking, prefer stopping over repeating the same action."
            )
        else:
            lines.append(
                "Continue with the next restoration action or stop if the image looks good."
            )
        return "\n".join(lines)

    async def create(
        self,
        instance_id: Optional[str] = None,
        original_image: Optional[str] = None,
        image_path: Optional[str] = None,
        degradation_type: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        """Create a tool instance for a trajectory.

        Args:
            instance_id: Optional instance identifier (generated if None).
            original_image: Path to the original degraded image.
            image_path: Alias for original_image (for dataset compatibility).
            degradation_type: Degradation category, e.g. 'fog', 'night', 'snow'.
                              Used to select IQA metric weights.
        """
        if original_image is None:
            original_image = kwargs.get("create_kwargs", {}).get("original_image", None)
        if image_path is None:
            image_path = kwargs.get("create_kwargs", {}).get("image_path", None)
        if degradation_type is None:
            degradation_type = kwargs.get("create_kwargs", {}).get("degradation_type", None)
        if original_image is None and image_path is not None:
            original_image = image_path

        if instance_id is None:
            instance_id = str(uuid4())

        instance_output_dir = os.path.join(self.output_dir, instance_id)
        os.makedirs(instance_output_dir, exist_ok=True)

        weights = SCORE_WEIGHT_MAP.get(degradation_type, DEFAULT_WEIGHT)

        # Compute identity (original) IQA scores
        identity_scores = self._get_iqa_scores(original_image) if original_image else [0.0] * 5

        self._instance_dict[instance_id] = {
            "original_image": original_image,
            "current_image": original_image,
            "processed_images": [],
            "actions_history": [],
            "scores_history": [identity_scores],
            "rewards_history": [],
            "marginals_history": [],
            "identity_scores": identity_scores,
            "weights": weights,
            "step": 0,
            "output_dir": instance_output_dir,
        }

        logger.info(
            f"Created restoration instance {instance_id} for: {original_image} "
            f"(degradation={degradation_type}, identity_scores={identity_scores})"
        )
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        """Execute a restoration action and return an IQA-based step reward.

        Args:
            instance_id: The instance identifier returned by ``create``.
            parameters: Dict with key ``action`` (e.g. ``'ridcp'``, ``'scunet'``).

        Returns:
            (ToolResponse, step_reward, metrics_dict)
        """
        action = parameters.get("action", "").lower().strip()

        if action not in ALLOWED_ACTIONS:
            error_msg = (
                f"Invalid action '{action}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_ACTIONS))}"
            )
            logger.warning(error_msg)
            return (
                ToolResponse(text=error_msg),
                FAILURE_REWARD,
                {"error": "invalid_action", "skip_tool_call_reward": True},
            )

        instance = self._instance_dict.get(instance_id)
        if instance is None:
            error_msg = f"Instance {instance_id} not found"
            logger.error(error_msg)
            return (
                ToolResponse(text=error_msg),
                FAILURE_REWARD,
                {"error": "instance_not_found", "skip_tool_call_reward": True},
            )

        if action == "stop":
            step = instance["step"]
            curr_scores = instance["scores_history"][-1]
            identity_scores = instance["identity_scores"]
            weights = instance["weights"]
            identity_delta = self._calculate_identity_delta(curr_scores, identity_scores, weights)
            recent_rewards = instance["rewards_history"][-self.stop_recent_reward_window :]
            stop_info = self._calculate_stop_reward(step, identity_delta, recent_rewards)
            reward = float(stop_info["reward"])
            instance["rewards_history"].append(reward)
            logger.info(
                f"Instance {instance_id}: stop action at step {step}, "
                f"identity_delta={identity_delta:.4f}, recent_reward_mean={stop_info['recent_reward_mean']:.4f}, "
                f"plateau={stop_info['plateau']}, good_enough={stop_info['good_enough']}, reward={reward}"
            )
            reason_parts = []
            if stop_info["good_enough"]:
                reason_parts.append("quality is already good enough")
            if stop_info["plateau"]:
                reason_parts.append("recent gains are small")
            reason_text = "; ".join(reason_parts) if reason_parts else "more improvement is still possible"
            return (
                ToolResponse(text=f"Restoration stopped after {step} step(s): {reason_text}."),
                reward,
                {
                    "action": "stop",
                    "step": step,
                    "identity_delta": identity_delta,
                    "recent_reward_mean": stop_info["recent_reward_mean"],
                    "plateau": stop_info["plateau"],
                    "good_enough": stop_info["good_enough"],
                    "skip_tool_call_reward": True,
                },
            )

        current_image = instance["current_image"]
        output_dir = instance["output_dir"]

        try:
            logger.info(f"Instance {instance_id}: applying '{action}' to {current_image}")
            result = self.toolkit.process_image(
                tools=[action],
                img_path=current_image,
                output_dir=output_dir,
                is_identify=True,
            )

            output_path = result.get("output_path")
            if not output_path or not os.path.exists(output_path):
                error_msg = "Restoration failed: no output generated"
                logger.error(error_msg)
                return (
                    ToolResponse(text=error_msg),
                    FAILURE_REWARD,
                    {"error": "restoration_failed", "skip_tool_call_reward": True},
                )

            # Compute IQA scores for the new image
            curr_scores = self._get_iqa_scores(output_path)
            prev_scores = instance["scores_history"][-1]
            identity_scores = instance["identity_scores"]
            weights = instance["weights"]

            reward_info = self._calculate_reward(
                prev_scores,
                curr_scores,
                identity_scores,
                weights,
                action=action,
                actions_history=instance["actions_history"],
            )
            reward = float(reward_info["reward"])

            # Update instance state
            instance["processed_images"].append((action, output_path))
            instance["actions_history"].append(action)
            instance["current_image"] = output_path
            instance["step"] += 1
            instance["scores_history"].append(curr_scores)
            instance["rewards_history"].append(reward)
            instance["marginals_history"].append(float(reward_info["marginal"]))

            identity_delta = self._calculate_identity_delta(curr_scores, identity_scores, weights)

            # Generate feedback text
            feedback = self._generate_feedback(
                action=action,
                step=instance["step"],
                reward=reward,
                actions_history=instance["actions_history"],
                marginal=float(reward_info["marginal"]),
                identity_delta=identity_delta,
                consecutive_action_count=int(reward_info["consecutive_action_count"]),
            )

            # Build response with restored image
            pil_img = None
            try:
                from PIL import Image as PILImage
                pil_img = PILImage.open(output_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Could not load result image with PIL: {e}")

            response = ToolResponse(
                image=[pil_img] if pil_img is not None else None,
                text=feedback,
            )

            logger.info(
                f"Instance {instance_id}: '{action}' done, step={instance['step']}, "
                f"reward={reward:.4f}, marginal={reward_info['marginal']:.4f}, "
                f"identity={reward_info['identity']:.4f}, repeat_penalty={reward_info['repeat_penalty']:.4f}, "
                f"output={output_path}"
            )
            return response, reward, {
                "action": action,
                "step": instance["step"],
                "reward": reward,
                "base_reward": reward_info["base_reward"],
                "marginal": reward_info["marginal"],
                "identity_delta": identity_delta,
                "repeat_penalty": reward_info["repeat_penalty"],
                "consecutive_action_count": int(reward_info["consecutive_action_count"]),
                "iqa_scores": curr_scores,
                "input_path": current_image,
                "output_path": output_path,
            }

        except Exception as e:
            error_msg = f"Restoration error during '{action}': {e}"
            logger.exception(error_msg)
            return (
                ToolResponse(text=error_msg),
                FAILURE_REWARD,
                {"error": str(e), "skip_tool_call_reward": True},
            )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Return cumulative reward for the trajectory (sum of step rewards)."""
        instance = self._instance_dict.get(instance_id)
        if instance is None:
            return 0.0
        return float(sum(instance.get("rewards_history", [])))

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
            logger.info(f"Released restoration instance {instance_id}")

    def get_instance_state(self, instance_id: str) -> Optional[dict]:
        return self._instance_dict.get(instance_id)
