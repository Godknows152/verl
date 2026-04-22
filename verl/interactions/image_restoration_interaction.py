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
Image Restoration Interaction for verl multi-turn RL training.

This interaction calculates rewards by comparing the original degraded image
with the restored image after each restoration step.

Reward strategies:
1. IQA-based scoring: Use multiple image quality assessment metrics
   (QAlign, MANIQA, MUSIQ, CLIPIQA, NIQE) with degradation-type-specific weights
2. Custom scoring: Implement your own comparison logic

The interaction also decides whether to continue the restoration process
based on score history and configurable thresholds.
"""

import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import torch

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Add restoration_tools/agent_tools to sys.path so we can import IQAScore etc.
# Path layout:  verl/interactions/ -> ../../restoration_tools/agent_tools
AGENT_TOOLS_PATH = Path(__file__).resolve().parent.parent.parent / 'restoration_tools' / 'agent_tools'
if AGENT_TOOLS_PATH.exists() and str(AGENT_TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(AGENT_TOOLS_PATH))

# Add submodule paths (Retinexformer, SCUNet, etc.) for model imports
_RESTORATION_TOOLS_PATH = AGENT_TOOLS_PATH.parent
_submodules = [
    'Retinexformer', 'HVICIDNet', 'LightenDiffusion', 'SCUNet',
    'ESRGAN', 'IDT', 'RIDCP', 'KANet', 'S2Former', 'SnowMaster', 'img2img_turbo',
]
for _submod in _submodules:
    _submod_path = _RESTORATION_TOOLS_PATH / _submod
    if _submod_path.exists() and str(_submod_path) not in sys.path:
        sys.path.insert(0, str(_submod_path))

# Score weights for different degradation types
# Each weight corresponds to [qalign, maniqa, musiq, clipiqa, niqe]
SCORE_WEIGHT_MAP = {
    'night': [2. / 9, 2. / 9, 0, 2. / 9, 3. / 9],
    'rain_streak': [1. / 5, 1.25 / 5, 1. / 5, 0.75 / 5, 1. / 5],
    'rain_drop': [0, 0.5 / 3, 0, 1.25 / 3, 1.25 / 3],
    'rain_drive': [0.5 / 4, 1.5 / 4, 1. / 4, 1. / 4, 0],
    'snow': [1.5 / 5, 0.75 / 5, 1. / 5, 0.75 / 5, 1. / 5],
    'fog': [1.5 / 5, 0.5 / 5, 1.5 / 5, 0.5 / 5, 1 / 5],
}

# Default weight when degradation type is unknown
DEFAULT_WEIGHT = [0.2, 0.2, 0.2, 0.2, 0.2]


class ImageRestorationInteraction(BaseInteraction):
    """Interaction agent for image restoration RL training.

    This interaction:
    1. Tracks the restoration history (original image, processed images, scores)
    2. Calculates rewards by comparing images via IQA metrics
    3. Generates feedback for the model to guide the next restoration decision
    4. Decides when to terminate based on score history

    Configuration options:
        - max_iterations: Maximum number of restoration iterations (default: 10)
        - use_iqa: Whether to use IQA metrics for scoring (default: True)
        - device: Device for IQA scoring (default: 'cuda')
        - reward_scale: Scale factor for final reward (default: 1.0)
        - alpha: Marginal gain coefficient in [0, 1] (default: 0.9).
                 beta = 1 - alpha is used for identity anchoring.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}

        # Configuration
        self.max_iterations = config.get("max_iterations", 10)
        self.use_iqa = config.get("use_iqa", True)
        self.device = config.get("device", "cuda")
        self.reward_scale = config.get("reward_scale", 1.0)
        self.alpha = float(config.get("alpha", 0.9))
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        self.beta = 1.0 - self.alpha

        # Lazy load IQA scorer
        self._iqa_scorer = None

        logger.info(
            f"ImageRestorationInteraction initialized: "
            f"max_iter={self.max_iterations}, use_iqa={self.use_iqa}, "
            f"device={self.device}, alpha={self.alpha:.3f}, beta={self.beta:.3f}"
        )

    @property
    def iqa_scorer(self):
        """Lazy load IQA scorer."""
        if self._iqa_scorer is None and self.use_iqa:
            try:
                from iqa_reward import IQAScore  # located in restoration_tools/agent_tools/
                self._iqa_scorer = IQAScore(device=self.device)
                logger.info(f"IQAScore initialized on {self.device}")
            except Exception as e:
                logger.warning(f"Failed to initialize IQAScore: {e}")
                self._iqa_scorer = None
        return self._iqa_scorer

    def _get_degradation_type(self, image_path: str) -> str:
        if not image_path:
            return "unknown"
        path_lower = image_path.lower()
        for deg_type in SCORE_WEIGHT_MAP.keys():
            if deg_type in path_lower:
                return deg_type
        return "unknown"

    def _get_score_weights(self, degradation_type: str) -> list:
        return SCORE_WEIGHT_MAP.get(degradation_type, DEFAULT_WEIGHT)

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        original_image: Optional[str] = None,
        **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        degradation_type = self._get_degradation_type(original_image)
        weights = self._get_score_weights(degradation_type)

        # Calculate identity score (IQA score of the original degraded image)
        identity_score = None
        if self.use_iqa and original_image:
            scorer = self.iqa_scorer
            if scorer:
                try:
                    identity_score = await asyncio.get_event_loop().run_in_executor(
                        None, scorer.get_iqa_score, original_image
                    )
                    logger.info(f"Identity score for {instance_id}: {identity_score}")
                except Exception as e:
                    logger.warning(f"Failed to calculate identity score: {e}")

        self._instance_dict[instance_id] = {
            "original_image": original_image,
            "current_image": original_image,
            "best_image": original_image,
            "processed_images": [],
            "actions": [],
            "scores": [],
            "rewards": [],
            "iteration": 0,
            "identity_score": identity_score,
            "prev_score": identity_score,
            "degradation_type": degradation_type,
            "weights": weights,
        }

        logger.info(
            f"Started interaction {instance_id} for image: {original_image}, "
            f"degradation_type={degradation_type}"
        )
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """Generate response after a restoration step.

        Args:
            instance_id: The instance id.
            messages: The conversation messages.
            **kwargs:
                processed_image: Path to the processed image (set by the tool layer).
                action: The restoration action applied.

        Returns:
            Tuple of (should_terminate, response_text, reward, metadata).
        """
        instance = self._instance_dict.get(instance_id)
        if instance is None:
            return True, "Instance not found.", 0.0, {"error": "instance_not_found"}

        processed_image = kwargs.get("processed_image")
        action = kwargs.get("action", "unknown")

        # Handle "stop" action
        if action.lower() == "stop":
            current_iteration = instance["iteration"]
            instance["actions"].append("stop")

            if current_iteration < 4:
                early_stop_penalty = -5.0
                logger.warning(
                    f"Instance {instance_id}: Early stop at iteration {current_iteration}, "
                    f"penalty={early_stop_penalty}"
                )
                return True, "Restoration process stopped.", early_stop_penalty, {
                    "iteration": current_iteration,
                    "action": "stop",
                    "early_stop": True,
                    "total_actions": len(instance["actions"]),
                    "rewards_history": instance["rewards"],
                }

            return True, "Restoration process stopped by user request.", 0.0, {
                "iteration": current_iteration,
                "action": "stop",
                "early_stop": False,
                "total_actions": len(instance["actions"]),
                "rewards_history": instance["rewards"],
            }

        # Try to extract processed image from messages if not provided
        if not processed_image:
            processed_image = self._extract_processed_image_from_messages(messages)

        # Calculate reward
        reward = 0.0
        raw_score = None

        if processed_image and os.path.exists(str(processed_image)):
            reward, raw_score = await self._calculate_reward(instance_id, processed_image)

            instance["processed_images"].append(processed_image)
            instance["actions"].append(action)
            instance["rewards"].append(reward)
            if raw_score is not None:
                instance["scores"].append(raw_score)
                instance["prev_score"] = raw_score
            instance["iteration"] += 1
            instance["current_image"] = processed_image

            if reward > 0:
                instance["best_image"] = processed_image
        else:
            reward = -8.0  # Severe penalty for missing/invalid output image
            logger.warning(f"No valid processed image found for instance {instance_id}")
            instance["processed_images"].append(None)
            instance["actions"].append(action)
            instance["rewards"].append(reward)
            instance["iteration"] += 1

        should_terminate = self._should_terminate(instance)
        response = self._generate_feedback(instance, action, reward, should_terminate)

        metadata = {
            "iteration": instance["iteration"],
            "action": action,
            "total_actions": len(instance["actions"]),
            "rewards_history": instance["rewards"],
            "raw_score": raw_score,
        }

        logger.info(
            f"Instance {instance_id}: action={action}, reward={reward:.4f}, "
            f"terminate={should_terminate}, iteration={instance['iteration']}"
        )

        return should_terminate, response, reward, metadata

    async def _calculate_reward(
        self,
        instance_id: str,
        processed_image: str
    ) -> tuple[float, Optional[list]]:
        """Calculate the reward using IQA metrics.

        Reward formula (mixed marginal + identity-anchored):
            marginal_diff  = processed_score - prev_score
            identity_diff  = processed_score - identity_score
            mixed_score    = alpha * dot(weights, marginal_diff)
                           + beta  * dot(weights, identity_diff)
            reward = clip(mixed_score * reward_scale, -10, 10)
        """
        instance = self._instance_dict.get(instance_id)
        if instance is None:
            return -5.0, None

        identity_score = instance.get("identity_score")
        prev_score = instance.get("prev_score", identity_score)
        weights = instance.get("weights", DEFAULT_WEIGHT)

        if not self.use_iqa or self.iqa_scorer is None:
            logger.warning("IQA scorer not available, returning 0 reward")
            return 0.0, None

        if identity_score is None:
            logger.warning("Identity score not available, returning 0 reward")
            return 0.0, None
        if prev_score is None:
            prev_score = identity_score

        try:
            processed_score = await asyncio.get_event_loop().run_in_executor(
                None, self.iqa_scorer.get_iqa_score, processed_image
            )

            device = self.device
            processed_tensor = torch.tensor(processed_score, device=device, dtype=torch.float32)
            identity_tensor = torch.tensor(identity_score, device=device, dtype=torch.float32)
            prev_tensor = torch.tensor(prev_score, device=device, dtype=torch.float32)
            weights_tensor = torch.tensor(weights, device=device, dtype=torch.float32)

            marginal_diff = processed_tensor - prev_tensor
            identity_diff = processed_tensor - identity_tensor

            weighted_marginal = (marginal_diff * weights_tensor).sum()
            weighted_identity = (identity_diff * weights_tensor).sum()
            mixed_score = self.alpha * weighted_marginal + self.beta * weighted_identity

            reward = mixed_score.item() * self.reward_scale
            reward = max(-10.0, min(10.0, reward))

            logger.debug(
                f"Reward: identity={identity_score}, prev={prev_score}, "
                f"processed={processed_score}, reward={reward:.4f}"
            )
            return reward, processed_score

        except Exception as e:
            logger.warning(f"Reward calculation failed: {e}")
            return -5.0, None  # Penalty for processing failure

    def _should_terminate(self, instance: dict) -> bool:
        if instance["iteration"] >= self.max_iterations:
            logger.info("Terminating: max iterations reached")
            return True
        actions = instance["actions"]
        if actions and actions[-1].lower() == "stop":
            logger.info("Terminating: stop action received")
            return True
        return False

    def _generate_feedback(
        self,
        instance: dict,
        last_action: str,
        last_reward: float,
        should_terminate: bool,
    ) -> str:
        if should_terminate:
            if instance["iteration"] >= self.max_iterations:
                return (
                    f"Maximum iterations ({self.max_iterations}) reached. "
                    f"Restoration process complete."
                )
            return "Restoration process complete."

        rewards = instance.get("rewards", [])
        actions = instance.get("actions", [])
        history = "\n".join(
            f"Step {i+1}: Action='{actions[i]}', Reward={rewards[i]:.4f}"
            for i in range(len(rewards))
        )

        current_step = len(rewards)
        min_steps_before_stop = 4

        next_step_hint = (
            "Refer to the latest restored image and consider which tool to use for "
            "the next restoration step. It is recommended to use a different type of restoration tool."
        )

        if last_reward < 0:
            feedback = (
                f"Restoration history:\n{history}\n\n"
                f"The tool '{last_action}' received reward {last_reward:.4f}. "
                f"Negative reward indicates the restoration was not effective. "
                f"Consider using a different tool. {next_step_hint}"
            )
        else:
            feedback = (
                f"Restoration history:\n{history}\n\n"
                f"The tool '{last_action}' received reward {last_reward:.4f}. "
                f"{next_step_hint}"
            )

        if current_step >= min_steps_before_stop:
            feedback += (
                f" You have completed {current_step} restoration steps. "
                f"Continue restoration or output 'stop' if the image is sufficiently restored."
            )

        return feedback

    def _extract_processed_image_from_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> Optional[str]:
        for msg in reversed(messages):
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    content = " ".join(text_parts)
                match = re.search(r"Output saved to:\s*(\S+)", content)
                if match:
                    return match.group(1)
        return None

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        instance = self._instance_dict.get(instance_id)
        if instance is None:
            return 0.0
        rewards = instance.get("rewards", [])
        return sum(rewards) if rewards else 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            instance = self._instance_dict[instance_id]
            logger.info(
                f"Finalizing interaction {instance_id}: "
                f"iterations={instance['iteration']}, "
                f"actions={instance['actions']}, "
                f"rewards={instance['rewards']}"
            )
            del self._instance_dict[instance_id]

    def get_instance_state(self, instance_id: str) -> Optional[dict]:
        return self._instance_dict.get(instance_id)

    def get_best_image(self, instance_id: str) -> Optional[str]:
        instance = self._instance_dict.get(instance_id)
        if instance:
            return instance.get("best_image")
        return None
