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

This tool wraps the RestorationToolkit from AIA_Restore project,
providing image denoising, dehazing, deraining, low-light enhancement,
super-resolution and other image restoration capabilities.

Supported restoration actions:
- real_esrgan: Super-resolution/deblurring/denoising/compression artifact removal
- scunet: High-quality denoising
- retinexformer_fivek: Low-light enhancement
- hvicidnet: Low-light/exposure correction
- lightdiff: Low-light enhancement (diffusion model)
- turbo_rain: Fast deraining
- s2former: Rain streak removal
- idt: Deraining/raindrop removal
- ridcp: Dehazing
- kanet: Dehazing
- turbo_snow: Desnowing
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from PIL import Image

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Add agent_tools package to path for importing RestorationToolkit
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

# Lazy load RestorationToolkit to avoid import errors when not needed
_toolkit_instance = None

# Allowed restoration actions
ALLOWED_ACTIONS = {
    'real_esrgan', 'scunet', 'retinexformer_fivek', 'hvicidnet', 'lightdiff',
    'turbo_rain', 's2former', 'idt', 'ridcp', 'kanet', 'turbo_snow', 'snowmaster', 'stop'
}


def get_toolkit(device: str = 'cuda', models: list = None, preload: bool = True, auto_unload: bool = False):
    """Lazy load and cache the RestorationToolkit instance.
    
    Note: IQA is NOT loaded here to avoid duplicate loading.
    IQA scoring for reward calculation is handled by ImageRestorationInteraction.
    
    Args:
        device: Device to load models on ('cuda', 'cuda:0', 'cpu', etc.)
        models: List of models to load (None = all models)
        preload: If True, load all models at initialization. If False, load on demand.
        auto_unload: If True (and preload=False), automatically unload models after use.
                    This helps avoid GPU memory conflicts with SGLang.
    """
    global _toolkit_instance
    if _toolkit_instance is None:
        try:
            from agent_tools.restoration_toolkit import RestorationToolkit
            # load_iqa=False - IQA is handled by Interaction layer to avoid duplicate
            _toolkit_instance = RestorationToolkit(
                models=models, 
                device=device, 
                load_iqa=False,
                preload=preload,
                auto_unload=auto_unload
            )
            logger.info(f"RestorationToolkit initialized on {device} (preload={preload}, auto_unload={auto_unload})")
        except Exception as e:
            logger.error(f"Failed to initialize RestorationToolkit: {e}")
            raise
    return _toolkit_instance


class RestorationTool(BaseTool):
    """A tool for image restoration/degradation removal.

    This tool provides image restoration capabilities including:
    - Super-resolution (real_esrgan)
    - Denoising (scunet)
    - Low-light enhancement (retinexformer_fivek, hvicidnet, lightdiff)
    - Deraining (turbo_rain, s2former, idt)
    - Dehazing (ridcp, kanet)
    - Desnowing (turbo_snow)

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the restoration operation
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize the RestorationTool.

        Args:
            config: Configuration dict containing:
                - device: 'cuda' or 'cpu' (default: 'cuda')
                - models: List of models to preload (default: None = load all)
                - output_dir: Directory for intermediate outputs (default: /tmp/verl_restoration)
                - preload: Whether to preload all models (default: True)
                - auto_unload: Auto unload models after use when preload=False (default: False)
            tool_schema: OpenAI function tool schema
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Configuration
        self.device = config.get("device", "cuda")
        self.preload_models = config.get("models", None)
        self.output_dir = config.get("output_dir", "/tmp/verl_restoration")
        
        # Dynamic loading options (to avoid GPU memory conflicts with SGLang)
        self.preload = config.get("preload", True)
        self.auto_unload = config.get("auto_unload", False)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize toolkit (lazy loaded on first use)
        self._toolkit = None
        
        logger.info(f"RestorationTool initialized with device={self.device}, preload={self.preload}, auto_unload={self.auto_unload}")

    @property
    def toolkit(self):
        """Lazy load the restoration toolkit."""
        if self._toolkit is None:
            self._toolkit = get_toolkit(
                device=self.device, 
                models=self.preload_models,
                preload=self.preload,
                auto_unload=self.auto_unload
            )
        return self._toolkit

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        original_image: Optional[str] = None,
        image_path: Optional[str] = None,
        **kwargs
    ) -> tuple[str, ToolResponse]:
        """Create a tool instance for a trajectory.

        Args:
            instance_id: The instance id of the tool.
            original_image: Path to the original degraded image.
            image_path: Alias for original_image (for compatibility).

        Returns:
            Tuple of (instance_id, ToolResponse).
        """
        # Support both original_image and image_path as parameter name
        if original_image is None and image_path is not None:
            original_image = image_path
        
        if original_image is None:
            logger.warning(f"original_image is None! kwargs received: {kwargs}")
        
        if instance_id is None:
            instance_id = str(uuid4())

        # Create instance-specific output directory
        instance_output_dir = os.path.join(self.output_dir, instance_id)
        os.makedirs(instance_output_dir, exist_ok=True)

        self._instance_dict[instance_id] = {
            "original_image": original_image,
            "current_image": original_image,
            "processed_images": [],  # List of (action, image_path) tuples
            "actions_history": [],
            "scores_history": [],
            "output_dir": instance_output_dir,
        }

        logger.info(f"Created restoration instance {instance_id} for image: {original_image}")
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, 
        instance_id: str, 
        parameters: dict[str, Any], 
        **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """Execute the restoration operation.

        Args:
            instance_id: The instance id of the tool.
            parameters: Dict containing:
                - action: The restoration action to apply (e.g., 'ridcp', 'scunet')

        Returns:
            Tuple of (ToolResponse, step_reward, metrics).
            ToolResponse contains the processed image.
        """
        action = parameters.get("action", "").lower().strip()
        
        # Validate action
        if action not in ALLOWED_ACTIONS:
            error_msg = f"Invalid action '{action}'. Allowed: {', '.join(sorted(ALLOWED_ACTIONS))}"
            logger.warning(error_msg)
            return ToolResponse(text=error_msg), -0.1, {"error": "invalid_action"}

        # Handle stop action
        if action == "stop":
            logger.info(f"Instance {instance_id}: Stop action received")
            return ToolResponse(text="Restoration process stopped."), 0.0, {"action": "stop"}

        instance = self._instance_dict.get(instance_id)
        if instance is None:
            error_msg = f"Instance {instance_id} not found"
            logger.error(error_msg)
            return ToolResponse(text=error_msg), -0.1, {"error": "instance_not_found"}

        current_image = instance["current_image"]
        output_dir = instance["output_dir"]

        try:
            # Execute restoration
            logger.info(f"Instance {instance_id}: Applying {action} to {current_image}")
            result = self.toolkit.process_image(
                tools=[action],
                img_path=current_image,
                output_dir=output_dir,
                is_identify=True
            )

            output_path = result.get("output_path")
            if not output_path or not os.path.exists(output_path):
                error_msg = f"Restoration failed: no output generated"
                logger.error(error_msg)
                return ToolResponse(text=error_msg), -0.1, {"error": "restoration_failed"}

            # Update instance state
            instance["processed_images"].append((action, output_path))
            instance["actions_history"].append(action)
            instance["current_image"] = output_path

            # Create response with processed image
            # Process image for multi-modal output
            processed_img = None
            try:
                from verl.utils.dataset.vision_utils import process_image as verl_process_image
                # process_image expects dict format or PIL Image
                processed_img = verl_process_image({"image": output_path})
                logger.info(f"Instance {instance_id}: Image processed successfully, type={type(processed_img)}")
            except ImportError as e:
                logger.warning(f"Instance {instance_id}: vision_utils not available: {e}")
            except Exception as e:
                logger.warning(f"Instance {instance_id}: Failed to process image for multi-modal output: {e}")
            
            # Build response - include image if successfully processed
            if processed_img is not None:
                response = ToolResponse(
                    image=[processed_img],
                    text=f"Applied '{action}' restoration. Output saved to: {output_path}"
                )
                logger.info(f"Instance {instance_id}: ToolResponse created WITH image")
            else:
                # Fallback: try to load image directly with PIL
                try:
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(output_path).convert("RGB")
                    response = ToolResponse(
                        image=[pil_img],
                        text=f"Applied '{action}' restoration. Output saved to: {output_path}"
                    )
                    logger.info(f"Instance {instance_id}: ToolResponse created with PIL fallback")
                except Exception as e:
                    logger.warning(f"Instance {instance_id}: PIL fallback failed: {e}, returning text-only response")
                    response = ToolResponse(
                        text=f"Applied '{action}' restoration. Output saved to: {output_path}"
                    )

            logger.info(f"Instance {instance_id}: {action} completed, output: {output_path}")
            
            # Return 0.0 step reward - actual reward will be calculated by Interaction
            return response, 0.0, {
                "action": action,
                "input_path": current_image,
                "output_path": output_path,
            }

        except Exception as e:
            error_msg = f"Restoration error: {str(e)}"
            logger.exception(error_msg)
            return ToolResponse(text=error_msg), -0.1, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the tool.

        This method can be used for simple reward calculation based on tool state.
        For more complex reward (comparing images), use the Interaction system.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward score.
        """
        instance = self._instance_dict.get(instance_id)
        if instance is None:
            return 0.0

        # Return average of scores if available
        scores = instance.get("scores_history", [])
        if scores:
            return sum(scores) / len(scores)
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance and clean up resources.

        Args:
            instance_id: The instance id of the tool.
        """
        if instance_id in self._instance_dict:
            instance = self._instance_dict[instance_id]
            output_dir = instance.get("output_dir")
            
            # Optionally clean up intermediate files
            # Uncomment if you want to delete intermediate results
            # import shutil
            # if output_dir and os.path.exists(output_dir):
            #     shutil.rmtree(output_dir, ignore_errors=True)
            
            del self._instance_dict[instance_id]
            logger.info(f"Released restoration instance {instance_id}")

    def get_instance_state(self, instance_id: str) -> Optional[dict]:
        """Get the current state of an instance.

        Args:
            instance_id: The instance id.

        Returns:
            Instance state dict or None if not found.
        """
        return self._instance_dict.get(instance_id)

    def update_score(self, instance_id: str, score: float) -> None:
        """Update the score history for an instance.

        Args:
            instance_id: The instance id.
            score: The score to add to history.
        """
        if instance_id in self._instance_dict:
            self._instance_dict[instance_id]["scores_history"].append(score)
