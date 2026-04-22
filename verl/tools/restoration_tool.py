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

This tool wraps the RestorationToolkit located in restoration_tools/agent_tools/,
providing denoising, dehazing, deraining, low-light enhancement, super-resolution
and other image restoration capabilities.

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
- stop: Stop the restoration process
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Add restoration_tools/agent_tools to sys.path for importing RestorationToolkit
# Path layout:  verl/tools/ -> ../../restoration_tools/agent_tools
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

# Module-level toolkit cache (lazy loaded)
_toolkit_instance = None


def get_toolkit(
    device: str = 'cuda',
    models: list = None,
    preload: bool = True,
    auto_unload: bool = False,
):
    """Lazy load and cache the RestorationToolkit instance.

    Note: IQA metrics are NOT loaded here — they are managed by the
    ImageRestorationInteraction to avoid duplicate GPU allocations.

    Args:
        device: Device to load models on ('cuda', 'cuda:0', 'cpu', etc.)
        models: List of model names to load (None = load all).
        preload: If True, load all models at init time. If False, load on demand.
        auto_unload: If True (and preload=False), unload each model after use
                     to free GPU memory for SGLang.
    """
    global _toolkit_instance
    if _toolkit_instance is None:
        try:
            from restoration_toolkit import RestorationToolkit  # in agent_tools/
            _toolkit_instance = RestorationToolkit(
                models=models,
                device=device,
                load_iqa=False,  # IQA handled by Interaction layer
                preload=preload,
                auto_unload=auto_unload,
            )
            logger.info(
                f"RestorationToolkit initialized on {device} "
                f"(preload={preload}, auto_unload={auto_unload})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize RestorationToolkit: {e}")
            raise
    return _toolkit_instance


class RestorationTool(BaseTool):
    """A tool for image restoration / degradation removal.

    Provides image restoration capabilities:
    - Super-resolution (real_esrgan)
    - Denoising (scunet)
    - Low-light enhancement (retinexformer_fivek, hvicidnet, lightdiff)
    - Deraining (turbo_rain, s2former, idt)
    - Dehazing (ridcp, kanet)
    - Desnowing (turbo_snow, snowmaster)

    Methods:
        get_openai_tool_schema: Return the OpenAI-compatible tool schema.
        create: Create a tool instance for one trajectory.
        execute: Execute the restoration operation and return the processed image.
        calc_reward: Placeholder — actual rewards are computed by Interaction layer.
        release: Release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict] = {}

        self.device = config.get("device", "cuda")
        self.preload_models = config.get("models", None)
        self.output_dir = config.get("output_dir", "/tmp/verl_restoration")
        self.preload = config.get("preload", True)
        self.auto_unload = config.get("auto_unload", False)

        os.makedirs(self.output_dir, exist_ok=True)
        self._toolkit = None

        logger.info(
            f"RestorationTool initialized: device={self.device}, "
            f"preload={self.preload}, auto_unload={self.auto_unload}"
        )

    @property
    def toolkit(self):
        """Lazy load the restoration toolkit on first use."""
        if self._toolkit is None:
            self._toolkit = get_toolkit(
                device=self.device,
                models=self.preload_models,
                preload=self.preload,
                auto_unload=self.auto_unload,
            )
        return self._toolkit

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        original_image: Optional[str] = None,
        image_path: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        """Create a tool instance for a trajectory.

        Args:
            instance_id: Optional instance identifier.
            original_image: Path to the original degraded image.
            image_path: Alias for original_image (for dataset compatibility).

        Returns:
            (instance_id, ToolResponse)
        """
        if original_image is None and image_path is not None:
            original_image = image_path

        if instance_id is None:
            instance_id = str(uuid4())

        instance_output_dir = os.path.join(self.output_dir, instance_id)
        os.makedirs(instance_output_dir, exist_ok=True)

        self._instance_dict[instance_id] = {
            "original_image": original_image,
            "current_image": original_image,
            "processed_images": [],   # list of (action, output_path) tuples
            "actions_history": [],
            "scores_history": [],
            "output_dir": instance_output_dir,
        }

        logger.info(f"Created restoration instance {instance_id} for: {original_image}")
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        """Execute a restoration action.

        Args:
            instance_id: The instance identifier.
            parameters: Dict with key ``action`` (e.g., ``'ridcp'``, ``'scunet'``).

        Returns:
            (ToolResponse, step_reward, metrics)
            step_reward is always 0.0 here — final rewards are computed by the
            ImageRestorationInteraction layer.
        """
        action = parameters.get("action", "").lower().strip()

        if action not in ALLOWED_ACTIONS:
            error_msg = (
                f"Invalid action '{action}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_ACTIONS))}"
            )
            logger.warning(error_msg)
            return ToolResponse(text=error_msg), -0.1, {"error": "invalid_action"}

        if action == "stop":
            logger.info(f"Instance {instance_id}: stop action received")
            return ToolResponse(text="Restoration process stopped."), 0.0, {"action": "stop"}

        instance = self._instance_dict.get(instance_id)
        if instance is None:
            error_msg = f"Instance {instance_id} not found"
            logger.error(error_msg)
            return ToolResponse(text=error_msg), -0.1, {"error": "instance_not_found"}

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
                return ToolResponse(text=error_msg), -0.1, {"error": "restoration_failed"}

            instance["processed_images"].append((action, output_path))
            instance["actions_history"].append(action)
            instance["current_image"] = output_path

            # Build a ToolResponse that includes the processed image
            processed_img = None
            try:
                from verl.utils.dataset.vision_utils import process_image as verl_process_image
                processed_img = verl_process_image({"image": output_path})
            except Exception:
                pass

            if processed_img is not None:
                response = ToolResponse(
                    image=[processed_img],
                    text=f"Applied '{action}' restoration. Output saved to: {output_path}",
                )
            else:
                # Fallback: load with PIL directly
                try:
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(output_path).convert("RGB")
                    response = ToolResponse(
                        image=[pil_img],
                        text=f"Applied '{action}' restoration. Output saved to: {output_path}",
                    )
                except Exception as e:
                    logger.warning(f"PIL fallback failed: {e}")
                    response = ToolResponse(
                        text=f"Applied '{action}' restoration. Output saved to: {output_path}"
                    )

            logger.info(
                f"Instance {instance_id}: '{action}' completed, output: {output_path}"
            )
            return response, 0.0, {
                "action": action,
                "input_path": current_image,
                "output_path": output_path,
            }

        except Exception as e:
            error_msg = f"Restoration error: {e}"
            logger.exception(error_msg)
            return ToolResponse(text=error_msg), -0.1, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Return zero — actual rewards are computed by ImageRestorationInteraction."""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
            logger.info(f"Released restoration instance {instance_id}")

    def get_instance_state(self, instance_id: str) -> Optional[dict]:
        return self._instance_dict.get(instance_id)
