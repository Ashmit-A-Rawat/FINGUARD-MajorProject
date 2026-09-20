"""Local Hugging Face inference (Qwen / Mistral). Everything runs on this machine.

Safety rules: the model is loaded lazily, and if its weights are not already in the local cache
this provider REFUSES to download them unless ``allow_download`` is set. No network call is made
during generation.
"""

import time
from pathlib import Path
from typing import Any

from llm.inference.base import (
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    Message,
    ModelNotAvailableError,
)


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def is_cached_locally(model_id: str) -> bool:
    """True if ``model_id`` is a local directory with weights, or is in the Hugging Face cache."""
    directory = Path(model_id)
    if directory.is_dir():
        return (directory / "config.json").exists() and any(directory.glob("*.safetensors"))
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(model_id, local_files_only=True)
    except Exception:  # noqa: BLE001 - hub raises several error types when files are absent
        return False
    return True


class LocalHFProvider(LLMProvider):
    name = "local-hf"
    default_model = ""

    def __init__(self, model: str = "", device: str = "auto", allow_download: bool = False) -> None:
        self.model = model or self.default_model
        self._device_request = device
        self._allow_download = allow_download
        self._tokenizer: Any = None  # transformers objects are loosely typed
        self._net: Any = None
        self.device = ""

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        return [m.model_dump() for m in messages]

    def load(self) -> None:
        if self._net is not None:
            return
        if not self._allow_download and not is_cached_locally(self.model):
            raise ModelNotAvailableError(
                f"{self.model} is not in the local Hugging Face cache and downloading is disabled. "
                "Download it deliberately (LLM_ALLOW_DOWNLOAD=true) after checking "
                "`python scripts/assess_hardware.py`."
            )
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = resolve_device(self._device_request)
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        net: Any = AutoModelForCausalLM.from_pretrained(self.model, dtype=dtype)
        self._net = net.to(self.device).eval()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        import torch

        self.load()
        started = time.perf_counter()
        tokenizer, net = self._tokenizer, self._net
        text = tokenizer.apply_chat_template(
            self._prepare_messages(request.messages), tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        if request.seed is not None:
            torch.manual_seed(request.seed)
        sample = request.temperature > 0
        with torch.no_grad():
            output = net.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=sample,
                temperature=request.temperature if sample else None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return GenerationResult(
            text=tokenizer.decode(new_tokens, skip_special_tokens=True),
            provider=self.name,
            model=self.model,
            is_mock=False,
            latency_s=time.perf_counter() - started,
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            completion_tokens=int(len(new_tokens)),
        )

    def close(self) -> None:
        import gc

        self._net = self._tokenizer = None
        gc.collect()


class LocalQwenProvider(LocalHFProvider):
    name = "qwen"
    default_model = "Qwen/Qwen2.5-1.5B-Instruct"


class LocalMistralProvider(LocalHFProvider):
    """Mistral chat templates reject a separate system role, so it is folded into the user turn."""

    name = "mistral"
    default_model = "mistralai/Mistral-7B-Instruct-v0.3"

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        rest = [m.model_dump() for m in messages if m.role != "system"]
        if system and rest and rest[0]["role"] == "user":
            rest[0]["content"] = f"{system}\n\n{rest[0]['content']}"
        return rest
