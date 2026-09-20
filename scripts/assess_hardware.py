"""Print the hardware profile and which local models fit. Downloads nothing."""

import sys

from llm.models.hardware import assess_hardware, recommend_models


def main() -> int:
    hw = assess_hardware()
    print(f"OS: {hw.os} ({hw.machine}); CPUs: {hw.cpu_count}; RAM: {hw.ram_gb} GB")
    print(f"Free disk: {hw.disk_free_gb} GB")
    print(
        f"Accelerator: {hw.accelerator}; CUDA: {hw.cuda_devices or 'none'}; MPS: {hw.mps_available}"
    )
    print(f"Memory budget for a model: {hw.memory_budget_gb:.1f} GB (16-bit + 1 GB overhead)\n")
    print(f"{'model':40s} {'weights':>8s} {'need':>7s} {'headroom':>9s}  fits")
    for r in recommend_models(hw):
        fits = "yes" if r.fits else "NO"
        line = f"{r.option.model_id:40s} {r.option.fp16_gb:7.2f}G {r.estimated_gb:6.1f}G"
        print(f"{line} {r.headroom_gb:8.1f}G  {fits}  {r.option.note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
