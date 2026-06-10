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

import asyncio
import json
import logging
import os
import sys
import torch
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
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

# Degradation type → recommended action affinity map.
# Values in [0, 1]: 1.0 = primary recommendation, 0.8 = strong, 0.5 = moderate.
# When affinity_bonus_scale > 0, choosing a high-affinity action yields a bonus
# on top of the IQA-based step reward, guiding the model to learn the mapping
# between degradation types and their specialized restoration tools.
DEGRADATION_ACTION_AFFINITY: dict[str, dict[str, float]] = {
    'night':       {'retinexformer_fivek': 1.0, 'hvicidnet': 1.0, 'lightdiff': 0.8},
    'rain_streak': {'s2former': 1.0, 'turbo_rain': 1.0, 'idt': 0.8},
    'rain_drop':   {'idt': 1.0, 'turbo_rain': 0.8, 's2former': 0.6},
    'rain_drive':  {'turbo_rain': 1.0, 'idt': 0.8, 's2former': 0.6},
    'fog':         {'ridcp': 1.0, 'kanet': 1.0},
    'snow':        {'turbo_snow': 1.0, 'snowmaster': 1.0},
}

# Legacy IQA metric weights per degradation type (QAlign, MANIQA, MUSIQ, CLIPIQA, NIQE).
# Prefer loading a data-driven map from local training data via ``iqa_weight_map_path``.
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
REWARD_MODE_STEP_MIXED_V1 = "step_mixed_v1"
REWARD_MODE_FINAL_IQA_V2 = "final_iqa_v2"
SUPPORTED_REWARD_MODES = {REWARD_MODE_STEP_MIXED_V1, REWARD_MODE_FINAL_IQA_V2}

# Module-level caches
_toolkit_instance = None
_runtime_pool_instance = None
_runtime_pool_config_key = None
_iqa_instances: dict[tuple[str, bool, str | None, str | None], Any] = {}


def _normalize_score_weight_vector(weights: list[float]) -> list[float]:
    if len(weights) != 5:
        raise ValueError(f"Expected 5 IQA weights, got {len(weights)}")
    tensor = torch.tensor(weights, dtype=torch.float32)
    tensor = torch.clamp(tensor, min=0.0)
    total = float(tensor.sum().item())
    if total <= 0.0:
        return list(DEFAULT_WEIGHT)
    return [float(value) for value in (tensor / total).tolist()]


def _load_score_weight_map(weight_map_path: str) -> dict[str, list[float]]:
    with open(weight_map_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    weights_payload = payload.get('weights', payload)
    if not isinstance(weights_payload, dict):
        raise ValueError(f"Invalid weight map payload in {weight_map_path}")

    normalized_map = {}
    for degradation_type, weights in weights_payload.items():
        if not isinstance(weights, list):
            raise ValueError(
                f"Weight entry for degradation '{degradation_type}' must be a list, got {type(weights)}"
            )
        normalized_map[degradation_type] = _normalize_score_weight_vector(weights)
    return normalized_map


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


def get_iqa_scorer(
    device: str = 'cuda',
    normalize_scores: bool = False,
    normalization_stats_path: str | None = None,
    qalign_path: str | None = None,
):
    """Lazy load and cache the IQAScore instance."""
    global _iqa_instances
    cache_key = (str(device), normalize_scores, normalization_stats_path, qalign_path)
    if cache_key not in _iqa_instances:
        try:
            from iqa_reward import IQAScore

            _iqa_instances[cache_key] = IQAScore(
                device=device,
                qalign_path=qalign_path,
                normalize_scores=normalize_scores,
                normalization_stats_path=normalization_stats_path,
            )
            logger.info(
                f"IQAScore initialized on {device} "
                f"(normalize_scores={normalize_scores}, stats_path={normalization_stats_path})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize IQAScore: {e}")
            raise
    return _iqa_instances[cache_key]


def _as_device_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


@dataclass
class _RestorationRuntimeWorker:
    index: int
    device: str
    iqa_device: str
    toolkit: Any
    iqa: Any | None


class _RestorationRuntimePool:
    """One full restoration+IQA runtime replica per configured GPU."""

    def __init__(
        self,
        *,
        worker_devices: list[str],
        iqa_devices: list[str],
        models: list[str] | None,
        preload: bool,
        auto_unload: bool,
        use_iqa: bool,
        normalize_iqa_scores: bool,
        iqa_stats_path: str | None,
        iqa_qalign_path: str | None,
    ):
        if not worker_devices:
            raise ValueError("worker_devices must contain at least one device")

        from restoration_tools.agent_tools import RestorationToolkit

        self.workers: list[_RestorationRuntimeWorker] = []
        self._available_worker_indices: Queue[int] = Queue()
        self._executor = ThreadPoolExecutor(
            max_workers=len(worker_devices),
            thread_name_prefix="restoration-runtime",
        )

        resolved_iqa_devices = iqa_devices or worker_devices
        for index, device in enumerate(worker_devices):
            iqa_device = resolved_iqa_devices[index % len(resolved_iqa_devices)]
            toolkit = RestorationToolkit(
                models=models,
                device=device,
                load_iqa=False,
                preload=False,
                auto_unload=auto_unload,
                model_devices=[device],
                model_device_map=None,
            )
            iqa = None
            if use_iqa:
                iqa = get_iqa_scorer(
                    device=iqa_device,
                    normalize_scores=normalize_iqa_scores,
                    normalization_stats_path=iqa_stats_path,
                    qalign_path=iqa_qalign_path,
                )
            worker = _RestorationRuntimeWorker(
                index=index,
                device=device,
                iqa_device=iqa_device,
                toolkit=toolkit,
                iqa=iqa,
            )
            self.workers.append(worker)
            self._available_worker_indices.put(index)

        if preload:
            self.preload_models()

        logger.info(
            "Restoration runtime pool initialized with workers: %s",
            [
                {"index": worker.index, "device": worker.device, "iqa_device": worker.iqa_device}
                for worker in self.workers
            ],
        )

    def _run_with_worker(self, fn):
        worker_index = self._available_worker_indices.get()
        worker = self.workers[worker_index]
        try:
            logger.info(
                "Dispatching restoration request to worker %s (tool=%s, iqa=%s)",
                worker.index,
                worker.device,
                worker.iqa_device,
            )
            return fn(worker)
        finally:
            self._available_worker_indices.put(worker_index)

    async def run(self, fn):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._run_with_worker, fn)

    def preload_models(self) -> None:
        # Model construction touches process-global torch/diffusers/import state.
        # Keep preload serialized; after models are resident, execute() still
        # dispatches inference concurrently across runtime workers.
        for worker in self.workers:
            logger.info(
                "Preloading restoration models on worker %s (%s)",
                worker.index,
                worker.device,
            )
            worker.toolkit.load_models()
        logger.info("Preloaded restoration models on %d runtime workers", len(self.workers))

    def unload_all_models(self) -> None:
        futures = [
            self._executor.submit(worker.toolkit.unload_all_models)
            for worker in self.workers
        ]
        for future in futures:
            future.result()
        logger.info("Unloaded restoration models on %d runtime workers", len(self.workers))


def get_runtime_pool(
    *,
    worker_devices: list[str],
    iqa_devices: list[str],
    models: list[str] | None,
    preload: bool,
    auto_unload: bool,
    use_iqa: bool,
    normalize_iqa_scores: bool,
    iqa_stats_path: str | None,
    iqa_qalign_path: str | None,
) -> _RestorationRuntimePool:
    """Lazy load and cache the multi-GPU restoration runtime pool."""
    global _runtime_pool_instance, _runtime_pool_config_key

    models_key = tuple(models or [])
    config_key = (
        tuple(worker_devices),
        tuple(iqa_devices),
        models_key,
        bool(auto_unload),
        bool(use_iqa),
        bool(normalize_iqa_scores),
        iqa_stats_path,
        iqa_qalign_path,
    )
    if _runtime_pool_instance is None or _runtime_pool_config_key != config_key:
        if _runtime_pool_instance is not None:
            try:
                _runtime_pool_instance.unload_all_models()
            except Exception as e:
                logger.warning("Failed to unload previous restoration runtime pool: %s", e)
        _runtime_pool_instance = _RestorationRuntimePool(
            worker_devices=worker_devices,
            iqa_devices=iqa_devices,
            models=models,
            preload=preload,
            auto_unload=auto_unload,
            use_iqa=use_iqa,
            normalize_iqa_scores=normalize_iqa_scores,
            iqa_stats_path=iqa_stats_path,
            iqa_qalign_path=iqa_qalign_path,
        )
        _runtime_pool_config_key = config_key
    return _runtime_pool_instance


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
    worker_devices = _as_device_list(runtime_cfg.get("worker_devices", None))
    iqa_devices = _as_device_list(runtime_cfg.get("iqa_devices", None))
    project_root = Path(__file__).resolve().parent.parent.parent
    iqa_stats_path = runtime_cfg.get("iqa_stats_path", None)
    iqa_qalign_path = runtime_cfg.get("iqa_qalign_path", None)
    if iqa_stats_path and not os.path.isabs(iqa_stats_path):
        iqa_stats_path = str((project_root / iqa_stats_path).resolve())
    if iqa_qalign_path and not os.path.isabs(iqa_qalign_path):
        iqa_qalign_path = str((project_root / iqa_qalign_path).resolve())
    if worker_devices:
        pool = get_runtime_pool(
            worker_devices=worker_devices,
            iqa_devices=iqa_devices,
            models=models,
            preload=False,
            auto_unload=False,
            use_iqa=bool(runtime_cfg.get("use_iqa", True)),
            normalize_iqa_scores=bool(runtime_cfg.get("normalize_iqa_scores", False)),
            iqa_stats_path=iqa_stats_path,
            iqa_qalign_path=iqa_qalign_path,
        )
        pool.preload_models()
        logger.info("Preloaded replicated restoration runtime pool for sampling stage")
        return True

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
    unloaded = False
    global _toolkit_instance, _runtime_pool_instance
    if _toolkit_instance is not None:
        _toolkit_instance.unload_all_models()
        logger.info("Unloaded all restoration models after sampling stage")
        unloaded = True
    if _runtime_pool_instance is not None:
        _runtime_pool_instance.unload_all_models()
        logger.info("Unloaded replicated restoration runtime pool after sampling stage")
        unloaded = True
    return unloaded


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
        self.worker_devices = _as_device_list(config.get("worker_devices", None))
        self.iqa_devices = _as_device_list(config.get("iqa_devices", None))
        self.use_parallel_workers = bool(self.worker_devices)
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
        self.normalize_iqa_scores = bool(config.get("normalize_iqa_scores", False))
        self.iqa_stats_path = config.get("iqa_stats_path", None)
        self.iqa_qalign_path = config.get("iqa_qalign_path", None)
        self.iqa_weight_map_path = config.get("iqa_weight_map_path", None)
        self.reward_mode = str(config.get("reward_mode", REWARD_MODE_STEP_MIXED_V1))
        if self.reward_mode not in SUPPORTED_REWARD_MODES:
            raise ValueError(
                f"Unsupported reward_mode={self.reward_mode!r}; "
                f"expected one of {sorted(SUPPORTED_REWARD_MODES)}"
            )
        self.suppress_tool_call_reward = bool(
            config.get("suppress_tool_call_reward", self.reward_mode == REWARD_MODE_FINAL_IQA_V2)
        )
        self.alpha = float(config.get("alpha", 0.9))       # marginal-improvement weight
        self.beta = 1.0 - self.alpha                       # identity-improvement weight
        self.reward_scale = float(config.get("reward_scale", 1.0))
        self.final_iqa_reward_scale = float(config.get("final_iqa_reward_scale", self.reward_scale))
        self.final_iqa_regression_penalty_scale = float(
            config.get("final_iqa_regression_penalty_scale", 1.0)
        )
        self.final_iqa_step_penalty = float(config.get("final_iqa_step_penalty", 0.0))
        self.affinity_bonus_scale = float(config.get("affinity_bonus_scale", 0.0))
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

        project_root = Path(__file__).resolve().parent.parent.parent
        if self.iqa_stats_path and not os.path.isabs(self.iqa_stats_path):
            self.iqa_stats_path = str((project_root / self.iqa_stats_path).resolve())
        if self.iqa_qalign_path and not os.path.isabs(self.iqa_qalign_path):
            self.iqa_qalign_path = str((project_root / self.iqa_qalign_path).resolve())
        if self.iqa_weight_map_path and not os.path.isabs(self.iqa_weight_map_path):
            self.iqa_weight_map_path = str((project_root / self.iqa_weight_map_path).resolve())
        if self.normalize_iqa_scores and not self.iqa_stats_path:
            raise ValueError(
                "normalize_iqa_scores=true requires iqa_stats_path generated from local training data"
            )

        self.score_weight_map = dict(SCORE_WEIGHT_MAP)
        if self.iqa_weight_map_path:
            self.score_weight_map = _load_score_weight_map(self.iqa_weight_map_path)

        os.makedirs(self.output_dir, exist_ok=True)
        self._toolkit = None
        self._iqa = None
        self._runtime_pool = None

        logger.info(
            f"RestorationTool initialized: device={self.device}, iqa_device={self.iqa_device}, "
            f"worker_devices={self.worker_devices}, iqa_devices={self.iqa_devices}, "
            f"use_iqa={self.use_iqa}, normalize_iqa_scores={self.normalize_iqa_scores}, "
            f"reward_mode={self.reward_mode}, "
            f"suppress_tool_call_reward={self.suppress_tool_call_reward}, "
            f"alpha={self.alpha}, reward_scale={self.reward_scale}, "
            f"final_iqa_reward_scale={self.final_iqa_reward_scale}, "
            f"final_iqa_regression_penalty_scale={self.final_iqa_regression_penalty_scale}, "
            f"final_iqa_step_penalty={self.final_iqa_step_penalty}, "
            f"affinity_bonus_scale={self.affinity_bonus_scale}, "
            f"repeat_action_penalty={self.repeat_action_penalty}, "
            f"repeat_low_gain_penalty={self.repeat_low_gain_penalty}, "
            f"stop_min_step={self.stop_min_step}, "
            f"model_devices={self.model_devices}, "
            f"weight_map_path={self.iqa_weight_map_path}"
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
            self._iqa = get_iqa_scorer(
                device=self.iqa_device,
                normalize_scores=self.normalize_iqa_scores,
                normalization_stats_path=self.iqa_stats_path,
                qalign_path=self.iqa_qalign_path,
            )
        return self._iqa

    @property
    def runtime_pool(self):
        if not self.use_parallel_workers:
            raise RuntimeError("runtime_pool requested but worker_devices is not configured")
        if self._runtime_pool is None:
            self._runtime_pool = get_runtime_pool(
                worker_devices=self.worker_devices,
                iqa_devices=self.iqa_devices,
                models=self.preload_models,
                preload=self.preload,
                auto_unload=self.auto_unload,
                use_iqa=self.use_iqa,
                normalize_iqa_scores=self.normalize_iqa_scores,
                iqa_stats_path=self.iqa_stats_path,
                iqa_qalign_path=self.iqa_qalign_path,
            )
        return self._runtime_pool

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

    def _get_iqa_scores_on_worker(self, worker: _RestorationRuntimeWorker, image_path: str) -> list[float]:
        """Compute IQA scores using the scorer attached to a runtime worker."""
        if not self.use_iqa or worker.iqa is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        try:
            scores = worker.iqa.get_iqa_score(image_path)
            return list(scores)
        except Exception as e:
            logger.warning(
                "IQA scoring failed for %s on worker %s (%s): %s",
                image_path,
                worker.index,
                worker.iqa_device,
                e,
            )
            return [0.0, 0.0, 0.0, 0.0, 0.0]

    async def _aget_iqa_scores(self, image_path: str) -> list[float]:
        """Async IQA scoring that uses the replicated runtime pool when enabled."""
        if self.use_parallel_workers:
            return await self.runtime_pool.run(
                lambda worker: self._get_iqa_scores_on_worker(worker, image_path)
            )
        return self._get_iqa_scores(image_path)

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
        degradation_type: str | None = None,
        best_identity_delta: float = 0.0,
    ) -> dict[str, float]:
        """Compute reward and diagnostics for a restoration step.

        Args:
            degradation_type: Optional degradation category (e.g. 'fog', 'rain_streak').
                Used to compute an affinity bonus when the action matches the degradation.
        """
        if self.reward_mode == REWARD_MODE_FINAL_IQA_V2:
            return self._calculate_final_iqa_reward_v2(
                prev_scores=prev_scores,
                curr_scores=curr_scores,
                identity_scores=identity_scores,
                weights=weights,
                action=action,
                actions_history=actions_history,
                best_identity_delta=best_identity_delta,
            )

        prev_t = torch.tensor(prev_scores, dtype=torch.float32)
        curr_t = torch.tensor(curr_scores, dtype=torch.float32)
        iden_t = torch.tensor(identity_scores, dtype=torch.float32)
        w_t = torch.tensor(weights, dtype=torch.float32)

        marginal = ((curr_t - prev_t) * w_t).sum().item()
        identity = ((curr_t - iden_t) * w_t).sum().item()
        mixed = self.alpha * marginal + self.beta * identity
        base_reward = mixed * self.reward_scale

        # Affinity bonus: reward choosing degradation-specific tools over generic ones.
        # Only awarded once per trajectory — the first time a high-affinity tool is chosen.
        # Subsequent affinity-matched actions receive no bonus so the model does not
        # learn to simply repeat the same specialized tool for extra reward.
        affinity_bonus = 0.0
        if degradation_type and self.affinity_bonus_scale > 0:
            affinity_map = DEGRADATION_ACTION_AFFINITY.get(degradation_type, {})
            affinity_score = affinity_map.get(action, 0.0)
            if affinity_score > 0:
                # Check whether an affinity bonus was already given in this trajectory
                already_given = any(
                    affinity_map.get(prev, 0.0) > 0 for prev in actions_history
                )
                if not already_given:
                    affinity_bonus = self.affinity_bonus_scale * affinity_score

        repeat_count = self._count_consecutive_repeats(actions_history, action)
        repeat_penalty = 0.0
        if repeat_count > 0:
            repeat_penalty += self.repeat_action_penalty * repeat_count
            if marginal <= self.repeat_low_gain_threshold:
                repeat_penalty += self.repeat_low_gain_penalty * repeat_count

        reward = float(torch.clamp(
            torch.tensor(base_reward - repeat_penalty + affinity_bonus), -10.0, 10.0
        ).item())
        return {
            "reward": reward,
            "base_reward": float(base_reward),
            "marginal": float(marginal),
            "identity": float(identity),
            "repeat_penalty": float(repeat_penalty),
            "affinity_bonus": float(affinity_bonus),
            "consecutive_action_count": float(repeat_count + 1),
            "best_identity_delta": float(max(best_identity_delta, identity)),
            "best_improvement": float(max(0.0, identity - best_identity_delta)),
            "regression_penalty": 0.0,
            "step_penalty": 0.0,
        }

    def _calculate_final_iqa_reward_v2(
        self,
        prev_scores: list[float],
        curr_scores: list[float],
        identity_scores: list[float],
        weights: list[float],
        action: str,
        actions_history: list[str],
        best_identity_delta: float,
    ) -> dict[str, float]:
        """Reward improvements to the trajectory-best final IQA score.

        The summed v2 reward tracks the best weighted IQA improvement achieved
        so far, instead of paying every marginal step independently. This keeps
        the objective aligned with the final restored image quality and avoids
        rewarding extra tool calls that do not produce a new best image.
        """
        prev_t = torch.tensor(prev_scores, dtype=torch.float32)
        curr_t = torch.tensor(curr_scores, dtype=torch.float32)
        iden_t = torch.tensor(identity_scores, dtype=torch.float32)
        w_t = torch.tensor(weights, dtype=torch.float32)

        marginal = ((curr_t - prev_t) * w_t).sum().item()
        identity = ((curr_t - iden_t) * w_t).sum().item()
        best_improvement = max(0.0, identity - best_identity_delta)
        regression = max(0.0, best_identity_delta - identity)
        base_reward = best_improvement * self.final_iqa_reward_scale
        regression_penalty = regression * self.final_iqa_regression_penalty_scale

        repeat_count = self._count_consecutive_repeats(actions_history, action)
        repeat_penalty = 0.0
        if repeat_count > 0:
            repeat_penalty += self.repeat_action_penalty * repeat_count
            if marginal <= self.repeat_low_gain_threshold:
                repeat_penalty += self.repeat_low_gain_penalty * repeat_count

        reward = float(torch.clamp(
            torch.tensor(
                base_reward
                - regression_penalty
                - self.final_iqa_step_penalty
                - repeat_penalty
            ),
            -10.0,
            10.0,
        ).item())
        return {
            "reward": reward,
            "base_reward": float(base_reward),
            "marginal": float(marginal),
            "identity": float(identity),
            "repeat_penalty": float(repeat_penalty),
            "affinity_bonus": 0.0,
            "consecutive_action_count": float(repeat_count + 1),
            "best_identity_delta": float(max(best_identity_delta, identity)),
            "best_improvement": float(best_improvement),
            "regression_penalty": float(regression_penalty),
            "step_penalty": float(self.final_iqa_step_penalty),
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
        degradation_type: str | None = None,
        best_identity_delta: float | None = None,
        best_improvement: float | None = None,
        regression_penalty: float = 0.0,
        step_penalty: float = 0.0,
    ) -> str:
        """Generate human-readable feedback for the model's next turn.

        Args:
            degradation_type: Optional degradation category for affinity hints.
        """
        history_str = " → ".join(actions_history) if actions_history else "none"

        if self.reward_mode == REWARD_MODE_FINAL_IQA_V2:
            best_identity_delta = identity_delta if best_identity_delta is None else best_identity_delta
            best_improvement = 0.0 if best_improvement is None else best_improvement
            lines = [
                f"Step {step}: Applied '{action}'.",
                f"Step reward: {reward:.4f}",
                f"Current improvement over original image: {identity_delta:.4f}",
                f"Trajectory-best improvement over original image: {best_identity_delta:.4f}",
                f"New best-IQA gain from this action: {best_improvement:.4f}",
                f"Weighted marginal improvement: {marginal:.4f}",
                f"Action history: {history_str}",
            ]
            if regression_penalty > 0.0:
                lines.append(
                    f"This action fell below the trajectory-best IQA "
                    f"(regression penalty: {regression_penalty:.4f})."
                )
            if step_penalty > 0.0:
                lines.append(f"Each extra tool call has a step cost of {step_penalty:.4f}.")
            if consecutive_action_count > 1:
                lines.append(
                    f"Consecutive uses of '{action}': {consecutive_action_count}. "
                    "Repeating the same tool without a new best IQA is discouraged."
                )
            if best_improvement > 0.0:
                lines.append(
                    "This action set a new trajectory-best IQA. Continue only if another "
                    "tool is likely to raise the best score further; otherwise stop."
                )
            elif step >= self.stop_min_step:
                lines.append(
                    "This action did not improve the trajectory-best IQA. Prefer stopping "
                    "or switching to a different targeted operation instead of repeating it."
                )
            else:
                lines.append(
                    "Continue exploring only if the next action is likely to improve the "
                    "trajectory-best IQA."
                )
            return "\n".join(lines)

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
        # NOTE: degradation-type affinity hints are intentionally omitted from the
        # feedback text.  The model must learn to diagnose the degradation and
        # choose appropriate tools on its own — revealing the degradation type or
        # recommending specific actions would shortcut that learning process.
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

        weights = self.score_weight_map.get(degradation_type, DEFAULT_WEIGHT)

        # Compute identity (original) IQA scores. In replicated mode this is
        # dispatched to whichever GPU worker is free, so batch create() calls
        # can score originals in parallel.
        identity_scores = await self._aget_iqa_scores(original_image) if original_image else [0.0] * 5

        self._instance_dict[instance_id] = {
            "original_image": original_image,
            "current_image": original_image,
            "processed_images": [],
            "actions_history": [],
            "scores_history": [identity_scores],
            "rewards_history": [],
            "marginals_history": [],
            "best_identity_delta": 0.0,
            "identity_scores": identity_scores,
            "weights": weights,
            "degradation_type": degradation_type,
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
                    "reward_mode": self.reward_mode,
                    "identity_delta": identity_delta,
                    "best_identity_delta": float(instance.get("best_identity_delta", identity_delta)),
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
            if self.use_parallel_workers:
                def _restore_and_score(worker: _RestorationRuntimeWorker):
                    result = worker.toolkit.process_image(
                        tools=[action],
                        img_path=current_image,
                        output_dir=output_dir,
                        is_identify=True,
                    )
                    output_path = result.get("output_path")
                    if output_path and os.path.exists(output_path):
                        curr_scores = self._get_iqa_scores_on_worker(worker, output_path)
                    else:
                        curr_scores = [0.0] * 5
                    return result, curr_scores, worker.index, worker.device, worker.iqa_device

                result, curr_scores, worker_index, worker_device, worker_iqa_device = await self.runtime_pool.run(
                    _restore_and_score
                )
                logger.info(
                    "Instance %s: worker %s completed '%s' (tool=%s, iqa=%s)",
                    instance_id,
                    worker_index,
                    action,
                    worker_device,
                    worker_iqa_device,
                )
            else:
                result = self.toolkit.process_image(
                    tools=[action],
                    img_path=current_image,
                    output_dir=output_dir,
                    is_identify=True,
                )
                curr_scores = None

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
            if curr_scores is None:
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
                degradation_type=instance.get("degradation_type"),
                best_identity_delta=float(instance.get("best_identity_delta", 0.0)),
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
            instance["best_identity_delta"] = float(reward_info["best_identity_delta"])

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
                degradation_type=instance.get("degradation_type"),
                best_identity_delta=float(reward_info["best_identity_delta"]),
                best_improvement=float(reward_info["best_improvement"]),
                regression_penalty=float(reward_info["regression_penalty"]),
                step_penalty=float(reward_info["step_penalty"]),
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
                f"identity={reward_info['identity']:.4f}, "
                f"best_identity_delta={reward_info['best_identity_delta']:.4f}, "
                f"repeat_penalty={reward_info['repeat_penalty']:.4f}, "
                f"affinity_bonus={reward_info['affinity_bonus']:.4f}, "
                f"output={output_path}"
            )
            metrics = {
                "action": action,
                "step": instance["step"],
                "reward": reward,
                "reward_mode": self.reward_mode,
                "base_reward": reward_info["base_reward"],
                "marginal": reward_info["marginal"],
                "identity_delta": identity_delta,
                "best_identity_delta": reward_info["best_identity_delta"],
                "best_improvement": reward_info["best_improvement"],
                "regression_penalty": reward_info["regression_penalty"],
                "step_penalty": reward_info["step_penalty"],
                "repeat_penalty": reward_info["repeat_penalty"],
                "affinity_bonus": reward_info["affinity_bonus"],
                "consecutive_action_count": int(reward_info["consecutive_action_count"]),
                "iqa_scores": curr_scores,
                "input_path": current_image,
                "output_path": output_path,
            }
            if self.suppress_tool_call_reward:
                metrics["skip_tool_call_reward"] = True
            return response, reward, metrics

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
