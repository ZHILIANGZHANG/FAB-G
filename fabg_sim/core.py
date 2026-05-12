from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


ATTRIBUTE_ORDER = ["color", "composition", "line", "light", "brushstroke"]

ATTRIBUTE_DESCRIPTIONS = {
    "brushstroke": "brush rhythm, texture, pressure, and painterly handling",
    "color": "hue, saturation, contrast, temperature, and chromatic harmony",
    "composition": "spatial layout, balance, focal arrangement, and visual hierarchy",
    "light": "illumination, shadow, brightness, and tonal atmosphere",
    "line": "contour, direction, curvature, edge quality, and linear rhythm",
}


class LLMBackend(Protocol):
    """Minimal interface shared by mock and real multimodal LLM backends."""

    def generate(
        self,
        *,
        image_path: str,
        prompt: str,
        role: str,
        attribute: str | None = None,
        salient_attributes: Sequence[str] | None = None,
    ) -> str:
        ...


@dataclass(frozen=True)
class AgentAdapterConfig:
    adapter_paths: Mapping[str, str]

    def __post_init__(self) -> None:
        required = set(ATTRIBUTE_ORDER + ["final"])
        missing = sorted(required.difference(self.adapter_paths))
        if missing:
            raise ValueError(f"Missing LoRA adapter paths for agents: {', '.join(missing)}")

    @classmethod
    def from_json_file(cls, path: str | Path) -> AgentAdapterConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Adapter config JSON must be an object.")
        return cls(adapter_paths={str(key): str(value) for key, value in payload.items()})

    def path_for(self, *, role: str, attribute: str | None = None) -> tuple[str, str]:
        if role == "attribute_agent":
            if attribute is None:
                raise ValueError("Attribute agent calls must provide an attribute.")
            return attribute, self.adapter_paths[attribute]
        if role == "final_agent":
            return "final", self.adapter_paths["final"]
        raise ValueError(f"Unknown backend role: {role}")


@dataclass(frozen=True)
class AttributeDecision:
    attribute: str
    raw_answer: str
    is_salient: bool
    prompt: str


@dataclass(frozen=True)
class EmotionAnalysis:
    emotion: str
    arousal: str
    valence: str
    explanation: str
    prompt: str


@dataclass(frozen=True)
class FABGResult:
    image_path: str
    decisions: list[AttributeDecision]
    salience_mask: list[int]
    salient_attributes: list[str]
    emotion: str
    arousal: str
    valence: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "salience_mask": dict(zip(ATTRIBUTE_ORDER, self.salience_mask, strict=True)),
            "salient_attributes": self.salient_attributes,
            "emotion": self.emotion,
            "arousal": self.arousal,
            "valence": self.valence,
            "explanation": self.explanation,
            "attribute_decisions": [
                {
                    "attribute": decision.attribute,
                    "answer": "yes" if decision.is_salient else "no",
                    "raw_answer": decision.raw_answer,
                }
                for decision in self.decisions
            ],
        }


def parse_yes_no(text: str) -> bool:
    """Parse a constrained salience answer from an LLM response."""

    normalized = text.strip().lower()
    if not normalized:
        raise ValueError("Empty yes/no answer.")

    match = re.search(r"\b(yes|no)\b", normalized)
    if match is None:
        raise ValueError(f"Expected a yes/no answer, got: {text!r}")
    return match.group(1) == "yes"


class AttributeAgent:
    """One attribute-specific salience screener in the FAB-G first stage."""

    def __init__(self, attribute: str, backend: LLMBackend) -> None:
        if attribute not in ATTRIBUTE_DESCRIPTIONS:
            raise ValueError(f"Unknown attribute: {attribute}")
        self.attribute = attribute
        self.backend = backend

    def build_prompt(self) -> str:
        description = ATTRIBUTE_DESCRIPTIONS[self.attribute]
        return (
            "You are a trained multimodal LLM used as one FAB-G attribute agent.\n"
            f"Question: Is {self.attribute} ({description}) one of the main factors "
            "that affects the artwork's emotional expression?\n"
            "Answer with only yes or no."
        )

    def predict(self, image_path: str) -> AttributeDecision:
        prompt = self.build_prompt()
        raw_answer = self.backend.generate(
            image_path=image_path,
            prompt=prompt,
            role="attribute_agent",
            attribute=self.attribute,
        )
        return AttributeDecision(
            attribute=self.attribute,
            raw_answer=raw_answer,
            is_salient=parse_yes_no(raw_answer),
            prompt=prompt,
        )


class EmotionAnalysisAgent:
    """Final cue-constrained analysis agent in the FAB-G second stage."""

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def build_prompt(self, salient_attributes: Sequence[str]) -> str:
        if salient_attributes:
            cue_text = ", ".join(salient_attributes)
        else:
            cue_text = "none"
        return (
            "You are the final FAB-G emotion analysis agent.\n"
            "Analyze the artwork's emotion using only the selected salient formal "
            f"attributes: {cue_text}.\n"
            "Return JSON with keys emotion, arousal, valence, and explanation. "
            "The explanation must cite only selected attributes."
        )

    def analyze(self, image_path: str, salient_attributes: Sequence[str]) -> EmotionAnalysis:
        prompt = self.build_prompt(salient_attributes)
        raw_answer = self.backend.generate(
            image_path=image_path,
            prompt=prompt,
            role="final_agent",
            salient_attributes=list(salient_attributes),
        )
        payload = _parse_json_object(raw_answer)
        return EmotionAnalysis(
            emotion=str(payload.get("emotion", "unknown")),
            arousal=str(payload.get("arousal", "unknown")),
            valence=str(payload.get("valence", "unknown")),
            explanation=str(payload.get("explanation", "")),
            prompt=prompt,
        )


class MultiAgentFABG:
    """Paper-aligned FAB-G inference flow over multiple LoRA-backed agents."""

    def __init__(
        self,
        backend: LLMBackend,
        attributes: Sequence[str] = ATTRIBUTE_ORDER,
    ) -> None:
        self.attributes = list(attributes)
        self.attribute_agents = [
            AttributeAgent(attribute=attribute, backend=backend) for attribute in self.attributes
        ]
        self.final_agent = EmotionAnalysisAgent(backend=backend)

    def analyze(self, image_path: str) -> FABGResult:
        decisions = [agent.predict(image_path) for agent in self.attribute_agents]
        salient_attributes = [
            decision.attribute for decision in decisions if decision.is_salient
        ]
        salience_mask = [1 if decision.is_salient else 0 for decision in decisions]
        final = self.final_agent.analyze(image_path, salient_attributes)
        return FABGResult(
            image_path=image_path,
            decisions=decisions,
            salience_mask=salience_mask,
            salient_attributes=salient_attributes,
            emotion=final.emotion,
            arousal=final.arousal,
            valence=final.valence,
            explanation=final.explanation,
        )


class Qwen3VLLoRABackend:
    """Local Qwen3-VL backend with one base model and per-agent LoRA adapters."""

    def __init__(
        self,
        *,
        base_model_path: str,
        adapters: AgentAdapterConfig,
        model: Any | None = None,
        processor: Any | None = None,
        model_load_kwargs: Mapping[str, Any] | None = None,
        generation_kwargs: Mapping[str, Any] | None = None,
        attribute_generation_kwargs: Mapping[str, Any] | None = None,
        final_generation_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.base_model_path = base_model_path
        self.adapters = adapters
        self.model_load_kwargs = dict(model_load_kwargs or {})
        self.generation_kwargs = dict(generation_kwargs or {})
        self.attribute_generation_kwargs = dict(attribute_generation_kwargs or {})
        self.final_generation_kwargs = dict(final_generation_kwargs or {})
        self.model = model
        self.processor = processor
        self._loaded_adapters: set[str] = set()

        if self.model is None or self.processor is None:
            self._load_base_model()

    def _load_base_model(self) -> None:
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen3VLLoRABackend requires transformers with Qwen3-VL support. "
                "Install/update transformers and peft before using the local backend."
            ) from exc

        kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
        }
        kwargs.update(self.model_load_kwargs)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.base_model_path,
            **kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
        )

    def generate(
        self,
        *,
        image_path: str,
        prompt: str,
        role: str,
        attribute: str | None = None,
        salient_attributes: Sequence[str] | None = None,
    ) -> str:
        adapter_name, adapter_path = self.adapters.path_for(role=role, attribute=attribute)
        self._ensure_adapter_loaded(adapter_name, adapter_path)
        self.model.set_adapter(adapter_name)

        messages = self._build_messages(image_path=image_path, prompt=prompt)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._input_device())

        kwargs = {
            "max_new_tokens": 8 if role == "attribute_agent" else 256,
            "do_sample": False,
        }
        kwargs.update(self.generation_kwargs)
        if role == "attribute_agent":
            kwargs.update(self.attribute_generation_kwargs)
        elif role == "final_agent":
            kwargs.update(self.final_generation_kwargs)
        generated_ids = self.model.generate(**inputs, **kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()

    def _input_device(self) -> Any:
        if hasattr(self.model, "device"):
            return self.model.device
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            return "cpu"

    def _ensure_adapter_loaded(self, adapter_name: str, adapter_path: str) -> None:
        if adapter_name in self._loaded_adapters:
            return
        self.model.load_adapter(adapter_path, adapter_name=adapter_name)
        self._loaded_adapters.add(adapter_name)

    def _build_messages(self, *, image_path: str, prompt: str) -> list[dict[str, Any]]:
        path = Path(image_path)
        if path.exists():
            image_reference = path.resolve().as_uri()
        else:
            image_reference = image_path
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_reference},
                    {"type": "text", "text": prompt},
                ],
            }
        ]


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Final agent must return JSON, got: {text!r}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Final agent JSON must be an object, got: {type(payload).__name__}")
    return payload
