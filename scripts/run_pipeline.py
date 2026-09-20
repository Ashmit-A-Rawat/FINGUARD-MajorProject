"""Run the data pipeline (ingest -> clean -> normalize -> consolidate) on a dataset directory."""

import argparse
import logging
import sys
from pathlib import Path

from backend.app.core.config import get_settings
from data_pipeline.pipeline import run_pipeline, write_processed


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="small")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")

    input_dir = args.input_dir or Path(settings.data_dir) / "synthetic" / args.preset
    output_dir = args.output_dir or Path(settings.data_dir) / "processed" / args.preset
    result = run_pipeline(input_dir)
    write_processed(result, output_dir)
    print(result.report.model_dump_json(indent=2))
    print(f"processed output written to {output_dir}")
    return 1 if result.quarantine else 0


if __name__ == "__main__":
    sys.exit(main())
