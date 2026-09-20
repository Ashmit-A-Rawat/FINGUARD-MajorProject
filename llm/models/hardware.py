"""Hardware inspection and model-size recommendation. Never downloads anything."""

import os
import platform
import shutil
from dataclasses import dataclass, field


@dataclass
class HardwareProfile:
    os: str
    machine: str
    cpu_count: int
    ram_gb: float  # on Apple Silicon this is unified memory shared with the GPU
    disk_free_gb: float
    cuda_devices: list[tuple[str, float]] = field(default_factory=list)  # (name, VRAM GB)
    mps_available: bool = False

    @property
    def accelerator(self) -> str:
        if self.cuda_devices:
            return "cuda"
        return "mps" if self.mps_available else "cpu"

    @property
    def memory_budget_gb(self) -> float:
        """Memory a model may plausibly use: 90% of VRAM on CUDA, else 60% of RAM (the OS,
        editor and other processes need the rest)."""
        if self.cuda_devices:
            return 0.9 * max(v for _, v in self.cuda_devices)
        return 0.6 * self.ram_gb


@dataclass(frozen=True)
class ModelOption:
    model_id: str
    family: str  # qwen | mistral
    params_b: float
    fp16_gb: float  # weights for ONE checkpoint format in 16-bit
    note: str = ""


CANDIDATES: tuple[ModelOption, ...] = (
    ModelOption(
        "Qwen/Qwen2.5-0.5B-Instruct", "qwen", 0.49, 0.99, "smoke tests; weak at strict JSON"
    ),
    ModelOption(
        "Qwen/Qwen2.5-1.5B-Instruct", "qwen", 1.54, 3.09, "smallest size that is plausibly useful"
    ),
    ModelOption("Qwen/Qwen2.5-3B-Instruct", "qwen", 3.09, 6.17, "research licence, not Apache-2.0"),
    ModelOption("Qwen/Qwen2.5-7B-Instruct", "qwen", 7.62, 15.2, ""),
    ModelOption("mistralai/Mistral-7B-Instruct-v0.3", "mistral", 7.25, 14.5, "Apache-2.0"),
)
RUNTIME_OVERHEAD_GB = 1.0  # KV cache at about 2k tokens plus framework buffers


@dataclass(frozen=True)
class Recommendation:
    option: ModelOption
    estimated_gb: float
    fits: bool
    headroom_gb: float


def assess_hardware() -> HardwareProfile:
    import torch

    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    cuda = [
        (torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory / 1e9)
        for i in range(torch.cuda.device_count())
    ]
    return HardwareProfile(
        os=f"{platform.system()} {platform.release()}",
        machine=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        ram_gb=round(ram, 1),
        disk_free_gb=round(shutil.disk_usage(".").free / 1e9, 1),
        cuda_devices=cuda,
        mps_available=bool(torch.backends.mps.is_available()),
    )


def recommend_models(
    profile: HardwareProfile, candidates: tuple[ModelOption, ...] = CANDIDATES
) -> list[Recommendation]:
    """16-bit weights only (no quantised runtime implemented), smallest first."""
    out = []
    for option in sorted(candidates, key=lambda o: o.fp16_gb):
        estimate = option.fp16_gb + RUNTIME_OVERHEAD_GB
        fits = estimate <= profile.memory_budget_gb and option.fp16_gb <= profile.disk_free_gb
        out.append(Recommendation(option, estimate, fits, profile.memory_budget_gb - estimate))
    return out
