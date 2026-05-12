from __future__ import annotations

import argparse
import json
from pathlib import Path

from fabg_sim.core import (
    ATTRIBUTE_ORDER,
    AgentAdapterConfig,
    MultiAgentFABG,
    Qwen3VLLoRABackend,
)


BASE_MODEL_PATH = r"Qwen/Qwen3-VL-8B-Instruct"

ATTRIBUTE_AGENT_ADAPTER_PATHS = {
    "color": r"checkpoints/fabg_color_lora",
    "composition": r"checkpoints/fabg_composition_lora",
    "line": r"checkpoints/fabg_line_lora",
    "light": r"checkpoints/fabg_light_lora",
    "brushstroke": r"checkpoints/fabg_brushstroke_lora",
}

FINAL_AGENT_ADAPTER_PATH = r"checkpoints/fabg_final_lora"

MAX_FINAL_TOKENS = 256


def build_adapter_config() -> AgentAdapterConfig:
    adapter_paths = dict(ATTRIBUTE_AGENT_ADAPTER_PATHS)
    adapter_paths["final"] = FINAL_AGENT_ADAPTER_PATH
    return AgentAdapterConfig(adapter_paths=adapter_paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FAB-G with one local Qwen3-VL base model and per-agent LoRA adapters."
    )
    parser.add_argument(
        "image",
        help="Path to the artwork image.",
    )
    parser.add_argument(
        "--max-final-tokens",
        type=int,
        default=MAX_FINAL_TOKENS,
        help="Maximum tokens for the final analysis agent.",
    )
    args = parser.parse_args()

    image_path = str(Path(args.image))
    backend = Qwen3VLLoRABackend(
        base_model_path=BASE_MODEL_PATH,
        adapters=build_adapter_config(),
        final_generation_kwargs={"max_new_tokens": args.max_final_tokens},
    )
    system = MultiAgentFABG(backend=backend, attributes=ATTRIBUTE_ORDER)
    result = system.analyze(image_path)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
