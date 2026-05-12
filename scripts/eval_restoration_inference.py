#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from statistics import fmean
from typing import Any

import pandas as pd
import torch
from omegaconf import ListConfig, OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.chat_template import apply_chat_template
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import hf_processor, hf_tokenizer, normalize_token_ids
from verl.utils.transformers_compat import get_auto_model_for_vision2seq


IQA_METRIC_NAMES = ["qalign", "maniqa", "musiq", "clipiqa", "niqe"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert image restoration assistant. "
    "Your task is to analyze a degraded image and apply appropriate restoration operations "
    "one step at a time using the available tools. "
    "After each restoration step you will receive the restored image and quality feedback. "
    "Select the most suitable restoration action for the observed degradation type, "
    "and call the 'restore_image' tool with the chosen action. "
    "Stop when the image quality is satisfactory."
)
DEFAULT_USER_PROMPT = (
    "This is an image<image> with degradation issues. Please identify the degradation type and decide which "
    "restoration method to use:\n"
    "Super-resolution/Deblurring/Denoising: real_esrgan\n"
    "Image Denoising: scunet\n"
    "Low-light Enhancement: retinexformer_fivek / hvicidnet / lightdiff\n"
    "Deraining: turbo_rain / s2former / idt\n"
    "Dehazing: ridcp / kanet\n"
    "Desnowing: turbo_snow / snowmaster"
)

_DEGRADATION_KEYWORDS = {
    "night": ["night", "dark", "low_light", "lowlight", "lol"],
    "rain_drop": ["rain_drop", "raindrop"],
    "rain_streak": ["rain_streak", "rainstreak", "streak"],
    "rain_drive": ["rain_drive", "driving", "drive"],
    "snow": ["snow"],
    "fog": ["fog", "haze", "hazy"],
    "rain": ["rain_series", "/rain/"],
}


@dataclass
class ResolvedConfig:
    train_config: str
    model_path: str
    data_paths: list[str]
    input_mode: str
    image_path: str | None
    image_dir: str | None
    tool_config_path: str
    output_dir: str
    tool_output_dir: str
    model_device: str
    torch_dtype: str
    max_samples: int
    sample_offset: int
    max_assistant_turns: int
    max_parallel_calls: int
    max_new_tokens: int
    response_length: int
    temperature: float
    top_p: float
    top_k: int
    phase_preload: bool
    attn_implementation: str | None
    trust_remote_code: bool
    seed: int | None
    dry_run: bool
    input_image_paths: list[str] = field(default_factory=list)
    recursive: bool = True
    degradation_type: str | None = None
    system_prompt_template: str = DEFAULT_SYSTEM_PROMPT
    user_prompt_template: str = DEFAULT_USER_PROMPT
    prompt_template_source: str = "builtin"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Qwen3-VL restoration checkpoint with the same multi-turn Hermes "
            "tool-calling protocol used in the current training pipeline."
        )
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="examples/sglang_multiturn/config/restoration_multiturn_grpo.yaml",
        help="Training config used to resolve default model/data/tool settings.",
    )
    parser.add_argument("--model-path", type=str, default=None, help="HF model or merged checkpoint path.")
    parser.add_argument(
        "--data-path",
        nargs="+",
        default=None,
        help=(
            "One or more parquet/json/jsonl files. In dataset mode they are evaluated directly; "
            "in --image-path/--image-dir mode they are used to recover the training prompt template."
        ),
    )
    parser.add_argument("--image-path", type=str, default=None, help="Evaluate one degraded image file.")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Evaluate all degraded images under a directory.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively scan --image-dir for images.",
    )
    parser.add_argument(
        "--degradation-type",
        type=str,
        default=None,
        help="Optional override for degradation_type when using --image-path/--image-dir.",
    )
    parser.add_argument(
        "--user-prompt",
        type=str,
        default=None,
        help="Optional override for the user prompt template in image mode.",
    )
    parser.add_argument(
        "--tool-config",
        type=str,
        default=None,
        help="Tool config yaml. Defaults to rollout.multi_turn.tool_config_path from --train-config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for summary.json, results.jsonl, and optional restored images.",
    )
    parser.add_argument(
        "--tool-output-dir",
        type=str,
        default=None,
        help="Directory for intermediate restored images. Defaults to <output-dir>/restored_images.",
    )
    parser.add_argument("--model-device", type=str, default="cuda:0", help="Model device or device_map='auto'.")
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype used for model loading.",
    )
    parser.add_argument("--max-samples", type=int, default=-1, help="Maximum number of samples to evaluate.")
    parser.add_argument("--sample-offset", type=int, default=0, help="Skip the first N dataset samples.")
    parser.add_argument(
        "--max-assistant-turns",
        type=int,
        default=None,
        help="Maximum assistant turns. Defaults to rollout.multi_turn.max_assistant_turns from --train-config.",
    )
    parser.add_argument(
        "--max-parallel-calls",
        type=int,
        default=1,
        help="Maximum tool calls to execute from a single assistant turn.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for each assistant generation step.",
    )
    parser.add_argument(
        "--response-length",
        type=int,
        default=None,
        help="Total response length used for no-tool penalty alignment. Defaults to data.max_response_length.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Optional attn_implementation passed to from_pretrained.",
    )
    parser.add_argument(
        "--phase-preload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload restoration models once before evaluation and unload after completion.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to HF model/tokenizer/processor loading.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional generation seed.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, dataset, and tool wiring without loading the model or running inference.",
    )
    return parser


def resolve_path(path_str: str | None, base_dir: Path) -> str | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def maybe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, ListConfig):
        return [str(item) for item in value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def load_train_config(path: str) -> Any:
    return OmegaConf.load(path)


def resolve_runtime_config(args: argparse.Namespace) -> ResolvedConfig:
    train_config_path = resolve_path(args.train_config, REPO_ROOT)
    if train_config_path is None or not os.path.exists(train_config_path):
        raise FileNotFoundError(f"Training config not found: {args.train_config}")

    train_cfg = load_train_config(train_config_path)
    model_path = args.model_path or OmegaConf.select(train_cfg, "actor_rollout_ref.model.path")
    if not model_path:
        raise ValueError("model path is empty; pass --model-path or set actor_rollout_ref.model.path in train config")

    image_path = resolve_path(args.image_path, REPO_ROOT)
    image_dir = resolve_path(args.image_dir, REPO_ROOT)
    if image_path and image_dir:
        raise ValueError("Use either --image-path or --image-dir, not both")

    data_paths = args.data_path or maybe_list(OmegaConf.select(train_cfg, "data.val_files"))
    input_mode = "images" if image_path or image_dir else "dataset"
    if not data_paths and input_mode == "dataset":
        raise ValueError("data path is empty; pass --data-path or set data.val_files in train config")

    tool_config_path = args.tool_config or OmegaConf.select(train_cfg, "actor_rollout_ref.rollout.multi_turn.tool_config_path")
    if not tool_config_path:
        raise ValueError(
            "tool config path is empty; pass --tool-config or set rollout.multi_turn.tool_config_path in train config"
        )

    max_assistant_turns = args.max_assistant_turns
    if max_assistant_turns is None:
        max_assistant_turns = int(OmegaConf.select(train_cfg, "actor_rollout_ref.rollout.multi_turn.max_assistant_turns") or 6)

    response_length = args.response_length
    if response_length is None:
        response_length = int(OmegaConf.select(train_cfg, "data.max_response_length") or 3072)

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(REPO_ROOT / "outputs" / "restoration_inference" / timestamp)

    tool_output_dir = args.tool_output_dir or str(Path(output_dir) / "restored_images")

    return ResolvedConfig(
        train_config=str(train_config_path),
        model_path=str(resolve_path(model_path, REPO_ROOT) or model_path),
        data_paths=[resolve_path(path, REPO_ROOT) or path for path in data_paths],
        input_mode=input_mode,
        image_path=image_path,
        image_dir=image_dir,
        recursive=bool(args.recursive),
        degradation_type=args.degradation_type,
        tool_config_path=str(resolve_path(tool_config_path, REPO_ROOT) or tool_config_path),
        output_dir=str(Path(output_dir).resolve()),
        tool_output_dir=str(Path(tool_output_dir).resolve()),
        user_prompt_template=args.user_prompt or DEFAULT_USER_PROMPT,
        prompt_template_source="cli" if args.user_prompt else "builtin",
        model_device=args.model_device,
        torch_dtype=args.torch_dtype,
        max_samples=args.max_samples,
        sample_offset=max(0, args.sample_offset),
        max_assistant_turns=max_assistant_turns,
        max_parallel_calls=max(1, args.max_parallel_calls),
        max_new_tokens=max(1, args.max_new_tokens),
        response_length=max(1, response_length),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        phase_preload=bool(args.phase_preload),
        attn_implementation=args.attn_implementation,
        trust_remote_code=bool(args.trust_remote_code),
        seed=args.seed,
        dry_run=bool(args.dry_run),
    )


def ensure_output_dirs(config: ResolvedConfig) -> None:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.tool_output_dir).mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Image.Image):
        return {"size": list(value.size), "mode": value.mode}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def detect_degradation_type(image_path: str) -> str:
    if not image_path:
        return "unknown"
    path_lower = image_path.lower()
    for degradation_type, keywords in _DEGRADATION_KEYWORDS.items():
        if any(keyword in path_lower for keyword in keywords):
            return "rain_streak" if degradation_type == "rain" else degradation_type
    return "unknown"


def load_image_as_bytes(image_path: str) -> dict[str, bytes]:
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="PNG")
            return {"bytes": buf.getvalue()}
    except Exception as exc:
        raise RuntimeError(f"Failed to load image: {image_path}") from exc


def ensure_image_placeholders(prompt_text: str, image_count: int) -> str:
    placeholder_count = prompt_text.count("<image>")
    missing = image_count - placeholder_count
    if missing > 0:
        return ("<image>" * missing) + "\n" + prompt_text
    return prompt_text


def normalize_prompt_messages(raw_prompt: Any) -> list[dict[str, Any]]:
    if hasattr(raw_prompt, "tolist") and not isinstance(raw_prompt, list):
        raw_prompt = raw_prompt.tolist()
    if isinstance(raw_prompt, tuple):
        raw_prompt = list(raw_prompt)

    normalized: list[dict[str, Any]] = []
    for message in raw_prompt or []:
        if not isinstance(message, dict):
            continue
        normalized.append(dict(message))
    return normalized


def extract_reference_prompt_templates(data_paths: list[str]) -> tuple[str, str, str]:
    system_counter: Counter[str] = Counter()
    user_counter: Counter[str] = Counter()

    for data_path in data_paths:
        path = Path(data_path)
        if not path.exists():
            continue

        if path.suffix == ".parquet":
            frame = pd.read_parquet(path, columns=["prompt"])
        elif path.suffix == ".json":
            frame = pd.read_json(path)
        elif path.suffix == ".jsonl":
            frame = pd.read_json(path, lines=True)
        else:
            continue

        if "prompt" not in frame.columns:
            continue

        for raw_prompt in frame["prompt"]:
            messages = normalize_prompt_messages(raw_prompt)
            if len(messages) >= 1 and isinstance(messages[0].get("content"), str):
                system_counter[messages[0]["content"]] += 1
            if len(messages) >= 2 and isinstance(messages[1].get("content"), str):
                user_counter[messages[1]["content"]] += 1

    system_prompt = system_counter.most_common(1)[0][0] if system_counter else DEFAULT_SYSTEM_PROMPT
    user_prompt = user_counter.most_common(1)[0][0] if user_counter else DEFAULT_USER_PROMPT
    prompt_source = "reference_dataset" if system_counter or user_counter else "builtin"
    return system_prompt, user_prompt, prompt_source


def collect_input_image_paths(config: ResolvedConfig) -> list[str]:
    if config.image_path:
        path = Path(config.image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {config.image_path}")
        return [str(path.resolve())]

    if config.image_dir:
        root = Path(config.image_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"Image directory not found: {config.image_dir}")
        iterator = root.rglob("*") if config.recursive else root.glob("*")
        image_paths = sorted(
            str(path.resolve())
            for path in iterator
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found under directory: {config.image_dir}")
        return image_paths

    return []


def build_raw_prompt_messages(messages: list[dict[str, Any]], images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_messages = copy.deepcopy(messages)
    image_offset = 0

    for message in output_messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue

        content_list: list[dict[str, Any]] = []
        segments = [segment for segment in re.split(r"(<image>)", content) if segment != ""]
        for segment in segments:
            if segment == "<image>":
                if image_offset >= len(images):
                    raise ValueError("Image placeholder count exceeds provided image count")
                content_list.append({"type": "image", **images[image_offset]})
                image_offset += 1
            else:
                content_list.append({"type": "text", "text": segment})

        message["content"] = content_list

    if image_offset != len(images):
        raise ValueError(f"Unused images remain after prompt expansion: {len(images) - image_offset}")
    return output_messages


def build_image_mode_samples(config: ResolvedConfig) -> list[dict[str, Any]]:
    base_messages = [
        {"role": "system", "content": config.system_prompt_template},
        {"role": "user", "content": ensure_image_placeholders(config.user_prompt_template, 1)},
    ]

    samples: list[dict[str, Any]] = []
    for index, image_path in enumerate(config.input_image_paths):
        image_payload = load_image_as_bytes(image_path)
        degradation_type = config.degradation_type or detect_degradation_type(image_path)
        raw_prompt = build_raw_prompt_messages(base_messages, [image_payload])
        samples.append(
            {
                "index": index,
                "data_source": "restoration",
                "raw_prompt": raw_prompt,
                "tools_kwargs": {
                    "restore_image": {
                        "create_kwargs": {
                            "image_path": image_path,
                            "degradation_type": degradation_type,
                        }
                    }
                },
                "extra_info": {
                    "index": index,
                    "image_path": image_path,
                    "degradation_type": degradation_type,
                    "need_tools_kwargs": True,
                },
            }
        )
    return samples


def named_iqa_scores(scores: list[float] | None) -> dict[str, float] | None:
    if scores is None:
        return None
    return {
        metric_name: float(metric_value)
        for metric_name, metric_value in zip(IQA_METRIC_NAMES, scores, strict=False)
    }


def flatten_images_from_messages(messages: list[dict[str, Any]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            image_payload = item.get("image")
            if isinstance(image_payload, Image.Image):
                images.append(image_payload.convert("RGB"))
                continue
            if isinstance(image_payload, str) and os.path.exists(image_payload):
                images.append(Image.open(image_payload).convert("RGB"))
                continue
            if isinstance(image_payload, dict) and "bytes" in image_payload:
                images.append(Image.open(BytesIO(image_payload["bytes"])).convert("RGB"))
                continue
            if "bytes" in item:
                images.append(Image.open(BytesIO(item["bytes"])).convert("RGB"))
                continue
            raise TypeError(f"Unsupported image payload in message: {item}")
    return images


def build_generation_inputs(
    processor: Any,
    tool_schemas: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    raw_prompt = apply_chat_template(
        processor,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tool_schemas,
    )
    images = flatten_images_from_messages(messages)
    model_inputs = processor(
        text=[raw_prompt],
        images=images or None,
        return_tensors="pt",
        do_sample_frames=False,
    )
    return raw_prompt, model_inputs


def move_model_inputs_to_device(model_inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in model_inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def make_assistant_message(content: str, tool_calls: list[FunctionCall]) -> dict[str, Any]:
    structured_calls = []
    for call in tool_calls:
        try:
            arguments = json.loads(call.arguments) if call.arguments else {}
        except json.JSONDecodeError:
            arguments = call.arguments
        structured_calls.append(
            {
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": arguments,
                },
            }
        )

    return {
        "role": "assistant",
        "content": content or "",
        "tool_calls": structured_calls,
    }


def make_tool_message(tool_response: ToolResponse) -> dict[str, Any]:
    if tool_response.image or tool_response.video:
        content: list[dict[str, Any]] = []
        if tool_response.image:
            images = tool_response.image if isinstance(tool_response.image, list) else [tool_response.image]
            for image in images:
                if image is not None:
                    content.append({"type": "image", "image": image})
        if tool_response.video:
            raise NotImplementedError("Video tool responses are not supported in restoration evaluation.")
        if tool_response.text:
            content.append({"type": "text", "text": tool_response.text})
        return {"role": "tool", "content": content}
    return {"role": "tool", "content": tool_response.text or ""}


def maybe_set_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataset(config: ResolvedConfig, tokenizer: Any, processor: Any) -> RLHFDataset:
    dataset_config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "image_key": "images",
            "video_key": "videos",
            "return_raw_chat": True,
            "return_multi_modal_inputs": False,
            "max_prompt_length": 32768,
            "filter_overlong_prompts": False,
            "filter_prompts": False,
            "shuffle": False,
        }
    )
    return RLHFDataset(
        data_files=config.data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=dataset_config,
        max_samples=-1,
    )


def load_model(config: ResolvedConfig) -> tuple[Any, Any, Any]:
    trust_remote_code = config.trust_remote_code
    tokenizer = hf_tokenizer(config.model_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(config.model_path, trust_remote_code=trust_remote_code)
    if processor is None:
        raise ValueError("Processor initialization failed; current restoration pipeline requires a VLM processor.")

    dtype = torch_dtype_from_name(config.torch_dtype)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation

    model_cls = get_auto_model_for_vision2seq()
    if config.model_device == "auto":
        model_kwargs["device_map"] = "auto"
        model = model_cls.from_pretrained(config.model_path, **model_kwargs)
    else:
        model = model_cls.from_pretrained(config.model_path, **model_kwargs)
        model = model.to(config.model_device)

    model.eval()
    return model, tokenizer, processor


def initialize_restore_tool(config: ResolvedConfig) -> Any:
    tool_list = initialize_tools_from_config(config.tool_config_path)
    tools_by_name = {tool.name: tool for tool in tool_list}
    if "restore_image" not in tools_by_name:
        available = ", ".join(sorted(tools_by_name))
        raise ValueError(f"restore_image tool not found in {config.tool_config_path}; available tools: {available}")

    tool = tools_by_name["restore_image"]
    tool.output_dir = config.tool_output_dir
    os.makedirs(tool.output_dir, exist_ok=True)
    return tool


def decode_response_text(tokenizer: Any, response_ids: list[int]) -> str:
    return tokenizer.decode(response_ids, skip_special_tokens=False)


def mean_or_none(values: list[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(fmean(filtered))


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    termination_counter = Counter(result["termination_reason"] for result in results)
    action_counter = Counter()
    by_degradation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_degradation[result["degradation_type"]].append(result)
        for action in result["actions"]:
            action_counter[action] += 1

    def build_group_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "samples": len(group),
            "mean_total_reward": mean_or_none([item.get("total_reward") for item in group]),
            "mean_identity_delta": mean_or_none([item.get("final_identity_delta") for item in group]),
            "mean_tool_calls": mean_or_none([item.get("tool_calls_executed") for item in group]),
            "mean_assistant_turns": mean_or_none([item.get("assistant_turns") for item in group]),
            "mean_elapsed_s": mean_or_none([item.get("elapsed_s") for item in group]),
            "stop_rate": sum(1 for item in group if item["termination_reason"] == "stop") / len(group),
        }

    total_elapsed = sum(float(item.get("elapsed_s", 0.0)) for item in results)
    aggregate = {
        "samples": len(results),
        "mean_total_reward": mean_or_none([item.get("total_reward") for item in results]),
        "mean_identity_delta": mean_or_none([item.get("final_identity_delta") for item in results]),
        "mean_tool_calls": mean_or_none([item.get("tool_calls_executed") for item in results]),
        "mean_assistant_turns": mean_or_none([item.get("assistant_turns") for item in results]),
        "mean_elapsed_s": mean_or_none([item.get("elapsed_s") for item in results]),
        "samples_per_second": (len(results) / total_elapsed) if total_elapsed > 0 else None,
        "termination_counts": dict(sorted(termination_counter.items())),
        "action_counts": dict(sorted(action_counter.items())),
        "by_degradation_type": {key: build_group_summary(value) for key, value in sorted(by_degradation.items())},
    }

    sample_summaries = []
    for result in results:
        sample_summaries.append(
            {
                "index": result["index"],
                "image_path": result["image_path"],
                "degradation_type": result["degradation_type"],
                "termination_reason": result["termination_reason"],
                "decision_chain": result["decision_chain"],
                "decision_chain_text": result["decision_chain_text"],
                "step_iqa_scores": [
                    {
                        "step": step["step"],
                        "assistant_turn": step["assistant_turn"],
                        "action": step["action"],
                        "effective_reward": step["effective_reward"],
                        "identity_delta": step["identity_delta"],
                        "iqa_scores": step["iqa_scores"],
                    }
                    for step in result["decision_steps"]
                ],
            }
        )

    return {
        "aggregate": aggregate,
        "sample_summaries": sample_summaries,
    }


def print_progress(sample_idx: int, total: int, result: dict[str, Any]) -> None:
    identity_delta = result.get("final_identity_delta")
    delta_text = f"{identity_delta:.4f}" if isinstance(identity_delta, (int, float)) else "n/a"
    chain_text = result.get("decision_chain_text") or "(no tool call)"
    print(
        f"[{sample_idx}/{total}] index={result['index']} deg={result['degradation_type']} "
        f"reason={result['termination_reason']} chain={chain_text} "
        f"reward={result['total_reward']:.4f} delta={delta_text} time={result['elapsed_s']:.2f}s"
    )


async def execute_tool_call(
    tool: Any,
    tool_name: str,
    instance_id: str,
    call: FunctionCall,
) -> tuple[ToolResponse, float, dict[str, Any]]:
    if call.name != tool_name:
        return (
            ToolResponse(text=f"Error when executing tool: unknown tool '{call.name}'"),
            -1.0,
            {"error": "unknown_tool", "requested_tool": call.name, "skip_tool_call_reward": True},
        )

    try:
        tool_args = json.loads(call.arguments) if call.arguments else {}
    except Exception as exc:
        return (
            ToolResponse(text=f"Error when executing tool: {exc}"),
            0.0,
            {"error": str(exc), "requested_tool": call.name, "skip_tool_call_reward": True},
        )

    return await tool.execute(instance_id, tool_args)


async def evaluate_sample(
    sample: dict[str, Any],
    tool: Any,
    tool_parser: ToolParser,
    tool_schemas: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    processor: Any,
    config: ResolvedConfig,
) -> dict[str, Any]:
    messages = copy.deepcopy(sample["raw_prompt"])
    tool_kwargs = copy.deepcopy(sample.get("tools_kwargs", {}))
    create_kwargs = copy.deepcopy(tool_kwargs.get(tool.name, {}).get("create_kwargs", {}))
    image_path = create_kwargs.get("image_path") or sample.get("extra_info", {}).get("image_path")
    degradation_type = create_kwargs.get("degradation_type") or sample.get("extra_info", {}).get("degradation_type", "unknown")
    instance_id, _ = await tool.create(**create_kwargs)

    tool_rewards: list[float] = []
    step_records: list[dict[str, Any]] = []
    assistant_records: list[dict[str, Any]] = []
    actions: list[str] = []
    termination_reason = "max_assistant_turns"
    current_image_path = image_path
    final_identity_delta: float | None = None
    final_iqa_scores: list[float] | None = None
    final_named_iqa_scores: dict[str, float] | None = None
    start_time = time.perf_counter()

    try:
        for assistant_turn in range(1, config.max_assistant_turns + 1):
            if config.seed is not None:
                maybe_set_seed(config.seed + int(sample.get("index", 0)) + assistant_turn)

            raw_prompt, model_inputs = build_generation_inputs(processor, tool_schemas, messages)
            input_device = next(model.parameters()).device
            model_inputs = move_model_inputs_to_device(model_inputs, input_device)
            input_length = int(model_inputs["input_ids"].shape[1])

            do_sample = config.temperature > 0.0
            generation_kwargs = {
                "max_new_tokens": config.max_new_tokens,
                "do_sample": do_sample,
                "temperature": config.temperature if do_sample else None,
                "top_p": config.top_p if do_sample else None,
                "top_k": config.top_k if do_sample else None,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
            }
            generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

            gen_start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**model_inputs, **generation_kwargs)
            generation_elapsed = time.perf_counter() - gen_start

            response_ids = normalize_token_ids(generated[0][input_length:])
            raw_response_text = decode_response_text(tokenizer, response_ids)
            assistant_text, tool_calls = await tool_parser.extract_tool_calls(response_ids, None)
            assistant_text = assistant_text.strip()
            assistant_records.append(
                {
                    "assistant_turn": assistant_turn,
                    "prompt_chars": len(raw_prompt),
                    "response_tokens": len(response_ids),
                    "generation_elapsed_s": generation_elapsed,
                    "assistant_text": assistant_text,
                    "raw_response_text": raw_response_text,
                    "tool_call_count": len(tool_calls),
                }
            )

            if not tool_calls:
                no_tool_penalty = ToolAgentLoop.EARLY_STOP_PENALTY
                tool_rewards.append(float(no_tool_penalty))
                if len(response_ids) > ToolAgentLoop.NO_TOOL_LENGTH_THRESHOLD:
                    denom = max(1, config.response_length - ToolAgentLoop.NO_TOOL_LENGTH_THRESHOLD)
                    ratio = (len(response_ids) - ToolAgentLoop.NO_TOOL_LENGTH_THRESHOLD) / denom
                    ratio = max(0.0, min(1.0, float(ratio)))
                    tool_rewards.append(float(-ToolAgentLoop.NO_TOOL_LENGTH_PENALTY_ALPHA * ratio))
                termination_reason = "no_tool_call"
                break

            messages.append(make_assistant_message(assistant_text, tool_calls))

            stop_triggered = False
            for call_idx, call in enumerate(tool_calls[: config.max_parallel_calls], start=1):
                tool_start = time.perf_counter()
                tool_response, tool_reward, tool_metrics = await execute_tool_call(tool, tool.name, instance_id, call)
                tool_elapsed = time.perf_counter() - tool_start

                tool_metrics = tool_metrics or {}
                tool_call_bonus = 0.0 if tool_metrics.get("skip_tool_call_reward") else ToolAgentLoop.TOOL_CALL_REWARD
                effective_reward = float(tool_reward + tool_call_bonus)
                tool_rewards.append(effective_reward)

                action = tool_metrics.get("action")
                if action:
                    actions.append(str(action))
                if tool_metrics.get("output_path"):
                    current_image_path = tool_metrics.get("output_path")
                if tool_metrics.get("identity_delta") is not None:
                    final_identity_delta = float(tool_metrics["identity_delta"])
                if tool_metrics.get("iqa_scores") is not None:
                    final_iqa_scores = list(tool_metrics["iqa_scores"])
                    final_named_iqa_scores = named_iqa_scores(final_iqa_scores)

                step_iqa_scores = named_iqa_scores(tool_metrics.get("iqa_scores"))

                record = {
                    "assistant_turn": assistant_turn,
                    "call_index": call_idx,
                    "requested_tool": call.name,
                    "requested_arguments": call.arguments,
                    "tool_elapsed_s": tool_elapsed,
                    "tool_reward": float(tool_reward),
                    "tool_call_bonus": float(tool_call_bonus),
                    "effective_reward": effective_reward,
                    "tool_response_text": tool_response.text,
                    "named_iqa_scores": step_iqa_scores,
                    **tool_metrics,
                }
                step_records.append(record)
                messages.append(make_tool_message(tool_response))

                if record.get("action") == "stop":
                    stop_triggered = True

            if stop_triggered:
                termination_reason = "stop"
                break

        elapsed = time.perf_counter() - start_time
        decision_steps = [
            {
                "step": step_index,
                "assistant_turn": record.get("assistant_turn"),
                "action": record.get("action"),
                "effective_reward": record.get("effective_reward"),
                "identity_delta": record.get("identity_delta"),
                "iqa_scores": record.get("named_iqa_scores"),
            }
            for step_index, record in enumerate(step_records, start=1)
        ]
        decision_chain = [str(action) for action in actions]
        return {
            "index": int(sample.get("index", 0)),
            "image_path": image_path,
            "degradation_type": degradation_type,
            "termination_reason": termination_reason,
            "assistant_turns": len(assistant_records),
            "tool_calls_executed": len(step_records),
            "actions": actions,
            "decision_chain": decision_chain,
            "decision_chain_text": " -> ".join(decision_chain) if decision_chain else "(no tool call)",
            "decision_steps": decision_steps,
            "total_reward": float(sum(tool_rewards)),
            "tool_rewards": tool_rewards,
            "final_identity_delta": final_identity_delta,
            "final_iqa_scores": final_iqa_scores,
            "final_named_iqa_scores": final_named_iqa_scores,
            "final_image_path": current_image_path,
            "elapsed_s": elapsed,
            "assistant_records": assistant_records,
            "step_records": step_records,
        }
    finally:
        await tool.release(instance_id)


def resolved_config_snapshot(config: ResolvedConfig) -> dict[str, Any]:
    return asdict(config)


async def maybe_preload_models(config: ResolvedConfig) -> bool:
    if not config.phase_preload:
        return False
    from verl.tools.restoration_tool import preload_restoration_models_for_sampling

    return bool(preload_restoration_models_for_sampling(config.tool_config_path))


async def maybe_unload_models(did_preload: bool) -> None:
    if not did_preload:
        return
    from verl.tools.restoration_tool import unload_restoration_models_after_sampling

    unload_restoration_models_after_sampling()


async def async_main() -> None:
    args = build_parser().parse_args()
    config = resolve_runtime_config(args)
    ensure_output_dirs(config)

    tokenizer = hf_tokenizer(config.model_path, trust_remote_code=config.trust_remote_code)
    processor = hf_processor(config.model_path, trust_remote_code=config.trust_remote_code)
    if processor is None:
        raise ValueError("Processor initialization failed; current restoration pipeline requires a VLM processor.")

    system_prompt, inferred_user_prompt, prompt_source = extract_reference_prompt_templates(config.data_paths)
    config.system_prompt_template = system_prompt
    if config.prompt_template_source != "cli":
        config.user_prompt_template = inferred_user_prompt
        config.prompt_template_source = prompt_source

    if config.input_mode == "images":
        config.input_image_paths = collect_input_image_paths(config)
        image_samples = build_image_mode_samples(config)
        total_dataset = len(image_samples)
        start_index = min(config.sample_offset, total_dataset)
        end_index = total_dataset if config.max_samples < 0 else min(total_dataset, start_index + config.max_samples)
        eval_samples = image_samples[start_index:end_index]
    else:
        dataset = create_dataset(config, tokenizer=tokenizer, processor=processor)
        total_dataset = len(dataset)
        start_index = min(config.sample_offset, total_dataset)
        end_index = total_dataset if config.max_samples < 0 else min(total_dataset, start_index + config.max_samples)
        eval_samples = dataset

    save_json(Path(config.output_dir) / "resolved_config.json", resolved_config_snapshot(config))

    tool = initialize_restore_tool(config)
    tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)]
    tool_parser = ToolParser.get_tool_parser("hermes", tokenizer)

    print(f"resolved model_path: {config.model_path}")
    print(f"resolved data_paths: {config.data_paths}")
    if config.input_mode == "images":
        print(f"resolved input_image_paths: {config.input_image_paths}")
        print(f"prompt_template_source: {config.prompt_template_source}")
        print(f"resolved user_prompt_template: {config.user_prompt_template}")
    print(f"resolved tool_config_path: {config.tool_config_path}")
    print(f"dataset size: {total_dataset}, evaluating range: [{start_index}, {end_index})")
    print(f"results will be written to: {config.output_dir}")

    if config.dry_run:
        print("dry-run completed: config, dataset, tokenizer/processor, and tool wiring are valid")
        return

    model, tokenizer, processor = load_model(config)
    tool_parser = ToolParser.get_tool_parser("hermes", tokenizer)
    maybe_set_seed(config.seed)

    results_path = Path(config.output_dir) / "results.jsonl"
    summary_path = Path(config.output_dir) / "summary.json"
    results: list[dict[str, Any]] = []
    did_preload = await maybe_preload_models(config)

    try:
        with results_path.open("w", encoding="utf-8") as fp:
            if config.input_mode == "images":
                iterator = enumerate(eval_samples, start=1)
            else:
                iterator = (
                    (offset, eval_samples[row_idx])
                    for offset, row_idx in enumerate(range(start_index, end_index), start=1)
                )

            for sample_idx, sample in iterator:
                result = await evaluate_sample(
                    sample=sample,
                    tool=tool,
                    tool_parser=tool_parser,
                    tool_schemas=tool_schemas,
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    config=config,
                )
                results.append(result)
                fp.write(json.dumps(result, ensure_ascii=False, default=json_default) + "\n")
                fp.flush()
                print_progress(sample_idx, end_index - start_index, result)
    finally:
        await maybe_unload_models(did_preload)

    summary = summarize_results(results)
    save_json(summary_path, summary)
    print("evaluation finished")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2, default=json_default))
    print(f"sample-level decision/IQA summary saved to: {summary_path}")


def main() -> None:
    import asyncio

    asyncio.run(async_main())


if __name__ == "__main__":
    main()