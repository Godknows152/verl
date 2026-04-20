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

# Add agent_tools package to path for IQA scoring and restoration toolkit
# agent_tools is located at workspace root /agent_tools (relative to this file: ../../../../agent_tools)
AGENT_TOOLS_PATH = Path(__file__).resolve().parent.parent.parent.parent / 'agent_tools'
if str(AGENT_TOOLS_PATH.parent) not in sys.path:
    sys.path.insert(0, str(AGENT_TOOLS_PATH.parent))

# Add submodule paths for agent_tools dependencies
_submodules = ['Retinexformer', 'HVICIDNet', 'LightenDiffusion', 'SCUNet', 
               'ESRGAN', 'IDT', 'RIDCP', 'KANet', 'S2Former', 'SnowMaster', 'img2img_turbo']
for _submod in _submodules:
    _submod_path = AGENT_TOOLS_PATH / _submod
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
    2. Calculates rewards by comparing images
    3. Generates feedback for the model to guide next restoration decision
    4. Decides when to terminate based on score history

    Configuration options:
        - score_decline_threshold: Number of consecutive low scores to trigger termination
        - significant_decline: Score threshold below which is considered ineffective
        - max_iterations: Maximum number of restoration iterations
        - use_iqa: Whether to use IQA metrics for scoring (requires IQAScore)
    """

    def __init__(self, config: dict):
        """Initialize the interaction.

        Args:
            config: Configuration dict containing:
                - max_iterations: Max restoration iterations (default: 10)
                - use_iqa: Use IQA metrics for scoring (default: True)
                - device: Device for IQA scoring (default: 'cuda')
                - reward_scale: Scale factor for final reward (default: 10.0)
                - alpha: Marginal gain coefficient in [0, 1] (default: 0.9)
                  beta is computed as (1 - alpha) for identity anchoring.
        """
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
            f"max_iter={self.max_iterations}, "
            f"use_iqa={self.use_iqa}, "
            f"device={self.device}, "
            f"alpha={self.alpha:.3f}, beta={self.beta:.3f}"
        )

    @property
    def iqa_scorer(self):
        """Lazy load IQA scorer."""
        if self._iqa_scorer is None and self.use_iqa:
            try:
                from agent_tools.iqa_reward import IQAScore
                self._iqa_scorer = IQAScore(device=self.device)
                logger.info(f"IQAScore initialized on {self.device}")
            except Exception as e:
                logger.warning(f"Failed to initialize IQAScore: {e}")
                self._iqa_scorer = None
        return self._iqa_scorer

    def _get_degradation_type(self, image_path: str) -> str:
        """Detect degradation type from image path.

        Args:
            image_path: Path to the image.

        Returns:
            Detected degradation type or 'unknown'.
        """
        if not image_path:
            return "unknown"
        path_lower = image_path.lower()
        for deg_type in SCORE_WEIGHT_MAP.keys():
            if deg_type in path_lower:
                return deg_type
        return "unknown"

    def _get_score_weights(self, degradation_type: str) -> list:
        """Get score weights for a degradation type.

        Args:
            degradation_type: Type of degradation.

        Returns:
            List of weights [qalign, maniqa, musiq, clipiqa, niqe].
        """
        return SCORE_WEIGHT_MAP.get(degradation_type, DEFAULT_WEIGHT)

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        original_image: Optional[str] = None,
        **kwargs
    ) -> str:
        """Start an interaction instance.

        Args:
            instance_id: Instance identifier.
            original_image: Path to the original degraded image.

        Returns:
            The instance id.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Detect degradation type from image path
        degradation_type = self._get_degradation_type(original_image)
        weights = self._get_score_weights(degradation_type)

        # Calculate identity score (IQA score of original image)
        identity_score = None
        if self.use_iqa and self.iqa_scorer and original_image:
            try:
                identity_score = self.iqa_scorer.get_iqa_score(original_image)
                logger.info(f"Identity score for {original_image}: {identity_score}")
            except Exception as e:
                logger.warning(f"Failed to calculate identity score: {e}")

        self._instance_dict[instance_id] = {
            "original_image": original_image,
            "current_image": original_image,
            "best_image": original_image,
            "processed_images": [],  # List of image paths after each restoration
            "actions": [],
            "scores": [],  # Raw IQA scores for each step
            "rewards": [],  # Calculated rewards for each step
            "iteration": 0,
            "identity_score": identity_score,  # IQA score of original image
            "prev_score": identity_score,  # IQA score of previous restoration step
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

        This method:
        1. Extracts the restoration result from tool output
        2. Calculates the reward using IQA metrics
        3. Decides whether to continue or terminate
        4. Generates feedback for the model

        Args:
            instance_id: The instance id.
            messages: The conversation messages.
            **kwargs: Additional arguments including:
                - processed_image: Path to the processed image
                - action: The restoration action that was applied

        Returns:
            Tuple of (should_terminate, response, reward, metadata).
        """
        instance = self._instance_dict.get(instance_id)
        if instance is None:
            return True, "Instance not found.", 0.0, {"error": "instance_not_found"}

        # Get processed image and action from kwargs or messages
        processed_image = kwargs.get("processed_image")
        action = kwargs.get("action", "unknown")

        # IMPORTANT: Handle "stop" action specially
        if action.lower() == "stop":
            current_iteration = instance["iteration"]
            instance["actions"].append("stop")
            
            # Penalize early stop (before completing at least 4 restoration steps)
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
            
            logger.info(f"Instance {instance_id}: Stop action received at iteration {current_iteration}, terminating")
            return True, "Restoration process stopped by user request.", 0.0, {
                "iteration": current_iteration,
                "action": "stop",
                "early_stop": False,
                "total_actions": len(instance["actions"]),
                "rewards_history": instance["rewards"],
            }

        # Try to extract from messages if not provided
        if not processed_image:
            processed_image = self._extract_processed_image_from_messages(messages)

        # Calculate reward
        reward = 0.0
        raw_score = None
        
        if processed_image and os.path.exists(str(processed_image)):
            reward, raw_score = await self._calculate_reward(
                instance_id,
                processed_image
            )
            
            # Update instance state
            instance["processed_images"].append(processed_image)
            instance["actions"].append(action)
            instance["rewards"].append(reward)
            if raw_score is not None:
                instance["scores"].append(raw_score)
                instance["prev_score"] = raw_score
            instance["iteration"] += 1

            # Update current image (always move forward in multi-turn)
            instance["current_image"] = processed_image
            
            # Update best image if reward is positive
            if reward > 0:
                instance["best_image"] = processed_image
        else:
            reward = -8.0  # Severe penalty for invalid output / format error
            logger.warning(f"No valid processed image found for instance {instance_id}")
            
            # 保持状态记录的数组长度一致，防止后续zip合并计算或Log解析错位
            instance["processed_images"].append(None)
            instance["actions"].append(action)
            instance["rewards"].append(reward)
            instance["iteration"] += 1

        # Check termination conditions
        should_terminate = self._should_terminate(instance)

        # Generate response/feedback
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
        """Calculate the reward by comparing processed image with identity score.

        Reward calculation uses a mixed objective:
        1. Get IQA scores for processed image [qalign, maniqa, musiq, clipiqa, niqe]
        2. Compute marginal diff = processed_score - prev_score
        3. Compute identity diff = processed_score - identity_score
        4. Apply degradation-type-specific weights
        5. Blend: combined = alpha * marginal + (1-alpha) * identity
        6. Final reward = combined * scale

        Args:
            instance_id: The instance id.
            processed_image: Path to the processed image.

        Returns:
            Tuple of (reward, raw_iqa_scores).
        """
        instance = self._instance_dict.get(instance_id)
        if instance is None:
            return -5.0, None  # Severe penalty for missing instance

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
            # Get IQA scores for processed image
            processed_score = self.iqa_scorer.get_iqa_score(processed_image)
            
            # Convert to tensors for calculation
            device = self.device
            processed_tensor = torch.tensor(processed_score, device=device, dtype=torch.float32)
            identity_tensor = torch.tensor(identity_score, device=device, dtype=torch.float32)
            prev_tensor = torch.tensor(prev_score, device=device, dtype=torch.float32)
            weights_tensor = torch.tensor(weights, device=device, dtype=torch.float32)

            # Calculate marginal and identity-anchored differences
            marginal_diff = processed_tensor - prev_tensor
            identity_diff = processed_tensor - identity_tensor

            # Apply weights and blend with alpha/beta
            weighted_marginal = (marginal_diff * weights_tensor).sum()
            weighted_identity = (identity_diff * weights_tensor).sum()
            mixed_score = self.alpha * weighted_marginal + self.beta * weighted_identity

            # Scale the reward
            reward = mixed_score.item() * self.reward_scale

            # Clamp reward to reasonable range (allow wider range for absolute diff)
            reward = max(-10.0, min(10.0, reward))

            logger.debug(
                f"Reward calculation: identity={identity_score}, "
                f"prev={prev_score}, processed={processed_score}, "
                f"marginal={marginal_diff.tolist()}, identity={identity_diff.tolist()}, "
                f"reward={reward:.4f}"
            )

            return reward, processed_score

        except Exception as e:
            logger.warning(f"Reward calculation failed: {e}")
            return -5.0, None  # Severe penalty for processing failure

    async def _calculate_score(
        self,
        before_image: str,
        after_image: str
    ) -> float:
        """Calculate the restoration score by comparing before/after images.

        This is a simplified version for backward compatibility.
        For full reward calculation, use _calculate_reward().

        Args:
            before_image: Path to the image before restoration.
            after_image: Path to the image after restoration.

        Returns:
            Score value.
        """
        if self.use_iqa and self.iqa_scorer:
            try:
                before_scores = self.iqa_scorer.get_iqa_score(before_image)
                after_scores = self.iqa_scorer.get_iqa_score(after_image)
                
                # Simple improvement calculation
                improvement = sum(after_scores) - sum(before_scores)
                return improvement
            except Exception as e:
                logger.warning(f"IQA scoring failed: {e}")

        return 0.0

    def _should_terminate(self, instance: dict) -> bool:
        """Determine if the restoration process should terminate.

        Termination conditions:
        1. Reached max iterations
        2. Model outputs 'stop' action

        Args:
            instance: The instance state dict.

        Returns:
            True if should terminate, False otherwise.
        """
        # Check max iterations
        if instance["iteration"] >= self.max_iterations:
            logger.info("Terminating: max iterations reached")
            return True

        # Check if last action was 'stop'
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
        should_terminate: bool
    ) -> str:
        """Generate feedback message for the model.

        Args:
            instance: The instance state dict.
            last_action: The last restoration action.
            last_reward: The reward of the last restoration.
            should_terminate: Whether the process should terminate.

        Returns:
            Feedback message string.
        """
        if should_terminate:
            if instance["iteration"] >= self.max_iterations:
                return (
                    f"Maximum iterations ({self.max_iterations}) reached. "
                    f"Restoration process complete."
                )
            else:
                return "Restoration process complete."

        # Build history string
        rewards = instance.get("rewards", [])
        actions = instance.get("actions", [])
        history = "\n".join(
            f"Step {i+1}: Action='{actions[i]}', Reward={rewards[i]:.4f}"
            for i in range(len(rewards))
        )

        # Get current step count (iteration is updated after each successful restoration)
        current_step = len(rewards)
        min_steps_before_stop = 4
        
        # Common instruction for next step
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
        
        # Only mention 'stop' option after completing at least 4 steps
        if current_step >= min_steps_before_stop:
            feedback += (
                f" You have completed {current_step} restoration steps. "
                f"Continue restoration or output 'stop' if the image is sufficiently restored."
            )
        
        return feedback

    def _extract_processed_image_from_messages(
        self,
        messages: list[dict[str, Any]]
    ) -> Optional[str]:
        """Extract processed image path from messages.

        Args:
            messages: The conversation messages.

        Returns:
            Path to processed image or None.
        """
        # Look for tool response containing output path
        for msg in reversed(messages):
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                # Handle multi-modal content (list of content blocks)
                if isinstance(content, list):
                    # Extract text from content blocks
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    content = " ".join(text_parts)
                # Try to extract path from "Output saved to: /path/to/image"
                match = re.search(r"Output saved to:\s*(\S+)", content)
                if match:
                    return match.group(1)
        return None

    def _extract_last_assistant_response(
        self,
        messages: list[dict[str, Any]]
    ) -> Optional[str]:
        """Extract the last assistant response from messages for length calculation.

        Args:
            messages: The conversation messages.

        Returns:
            The text content of the last assistant message, or None.
        """
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Handle multi-modal content (list of content blocks)
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    return " ".join(text_parts)
                elif isinstance(content, str):
                    return content
        return None

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Calculate the final reward for the interaction.

        Args:
            instance_id: The instance id.

        Returns:
            The final cumulative reward.
        """
        instance = self._instance_dict.get(instance_id)
        if instance is None:
            return 0.0

        rewards = instance.get("rewards", [])
        if not rewards:
            return 0.0

        # Return the sum of all rewards as final reward
        return sum(rewards)

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Finalize and clean up the interaction.

        Args:
            instance_id: The instance id.
        """
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
        """Get the current state of an instance.

        Args:
            instance_id: The instance id.

        Returns:
            Instance state dict or None if not found.
        """
        return self._instance_dict.get(instance_id)

    def get_best_image(self, instance_id: str) -> Optional[str]:
        """Get the best restored image path.

        Args:
            instance_id: The instance id.

        Returns:
            Path to the best restored image or None.
        """
        instance = self._instance_dict.get(instance_id)
        if instance:
            return instance.get("best_image")
        return None
