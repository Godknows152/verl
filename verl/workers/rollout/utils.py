# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import asyncio
import logging
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI

logger = logging.getLogger(__file__)


def get_max_position_embeddings(hf_config) -> int:
    max_len = getattr(hf_config, "max_position_embeddings", None)
    if max_len is None:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            max_len = getattr(text_config, "max_position_embeddings", None)

    if max_len is None:
        raise ValueError("max_position_embeddings not found in HFModelConfig!")
    return int(max_len)


class _UvicornServerAutoPort(uvicorn.Server):
    """Uvicorn Server that reports the system-assigned port when port=0."""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.actual_port: int | None = None
        self._startup_done: asyncio.Event = asyncio.Event()

    async def startup(self, sockets=None) -> None:
        try:
            await super().startup(sockets=sockets)
            if self.servers and self.config.port == 0:
                sock = self.servers[0].sockets[0]
                self.actual_port = sock.getsockname()[1]
            else:
                self.actual_port = self.config.port
        finally:
            self._startup_done.set()

    async def get_port(self) -> int | None:
        await self._startup_done.wait()
        return self.actual_port


async def run_uvicorn(app: FastAPI, server_args, server_address) -> tuple[int, asyncio.Task]:
    app.server_args = server_args
    config = uvicorn.Config(app, host=server_address, port=0, log_level="warning")
    server = _UvicornServerAutoPort(config)
    server_task = asyncio.create_task(server.serve())
    server_port = await server.get_port()
    if server_port is None:
        # server.startup() failed. await the task to re-raise exception from server.serve()
        await server_task

        # Fails on unexpected situation.
        raise RuntimeError("Unexpected: HTTP server started without reporting listened port")
    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task


async def ensure_async_iterator(iterable):
    """Convert an iterable to an async iterator."""
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item


def qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor):
    """Deduplicate consecutive image tokens in prompt_ids for Qwen2.5-VL, since vLLM will replicate the
    <|image_pad|> and <|video_pad|> token by image_data.
    For example,
    ```
    <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
    =>
    <|vision_start|><|image_pad|><|vision_end|>
    ```
    """
    if (
        processor is not None
        and hasattr(processor, "image_processor")
        and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__
    ):
        prompt_ids = np.array(prompt_ids)
        mask = np.ones(len(prompt_ids), dtype=bool)
        is_value = (prompt_ids == processor.image_token_id) | (prompt_ids == processor.video_token_id)
        mask[1:] &= ~(is_value[1:] & is_value[:-1])
        return prompt_ids[mask].tolist()
    else:
        return prompt_ids


def get_multimodal_special_token_ids(processor: Any) -> list[int]:
    """Return multimodal special token ids that should not be freely generated."""
    if processor is None:
        return []

    token_ids: list[int] = []
    tokenizer = getattr(processor, "tokenizer", None)

    for attr_name in ["image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"]:
        token_id = getattr(processor, attr_name, None)
        if isinstance(token_id, int) and token_id >= 0:
            token_ids.append(token_id)

    if tokenizer is not None:
        for token in ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0:
                token_ids.append(token_id)

    return list(dict.fromkeys(token_ids))


def apply_multimodal_generation_token_bias(
    sampling_params: dict[str, Any],
    processor: Any,
    disable_multimodal_special_token_generation: bool,
    bias: float = -100.0,
) -> dict[str, Any]:
    """Merge strong negative logit bias for multimodal control tokens into sampling params."""
    if not disable_multimodal_special_token_generation:
        return sampling_params

    blocked_token_ids = get_multimodal_special_token_ids(processor)
    if not blocked_token_ids:
        return sampling_params

    updated_sampling_params = dict(sampling_params)
    existing_logit_bias = updated_sampling_params.get("logit_bias") or {}
    merged_logit_bias = dict(existing_logit_bias)
    for token_id in blocked_token_ids:
        merged_logit_bias[token_id] = min(float(merged_logit_bias.get(token_id, 0.0)), bias)

    updated_sampling_params["logit_bias"] = merged_logit_bias
    return updated_sampling_params
