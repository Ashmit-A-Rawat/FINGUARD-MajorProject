"""Deliberately download a model's weights into the local Hugging Face cache.

Nothing is downloaded unless --yes is passed. Only inference files are fetched (safetensors,
tokenizer, config), not training checkpoints or alternative formats.

    python scripts/download_model.py Qwen/Qwen2.5-1.5B-Instruct --yes
"""

import argparse
import sys

from huggingface_hub import model_info, snapshot_download

ALLOW = ["*.json", "*.safetensors", "*.txt", "tokenizer*", "merges.txt", "vocab.*"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id")
    parser.add_argument("--yes", action="store_true", help="confirm the download")
    args = parser.parse_args()
    info = model_info(args.model_id, files_metadata=True)
    files = info.siblings or []
    size = sum((f.size or 0) for f in files if f.rfilename.endswith(".safetensors"))
    licence = info.card_data.license if info.card_data else "unknown"
    print(f"{args.model_id}: about {size / 1e9:.2f} GB of weights, licence {licence}")
    if not args.yes:
        print("Not downloading (pass --yes to confirm).")
        return 1
    path = snapshot_download(args.model_id, allow_patterns=ALLOW)
    print(f"downloaded to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
