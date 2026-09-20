"""Generate the FIN-GUARD synthetic banking dataset.

Usage:
    python scripts/generate_synthetic_data.py --preset small
    python scripts/generate_synthetic_data.py --preset large --allow-large
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from backend.app.core.config import get_settings
from data_pipeline.synthetic.config import PRESETS, config_for_preset
from data_pipeline.synthetic.generate import generate_dataset, write_dataset


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="small")
    parser.add_argument("--seed", type=int, default=settings.synthetic_seed)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help="required for the 'large' preset (1M+ transactions, several GB of RAM)",
    )
    args = parser.parse_args(argv)

    if args.preset == "large" and not args.allow_large:
        print("Refusing to generate 'large' without --allow-large.", file=sys.stderr)
        return 2

    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")
    cfg = config_for_preset(args.preset, seed=args.seed)
    out_dir = args.output_dir or Path(settings.data_dir) / "synthetic" / args.preset
    dataset = generate_dataset(cfg)
    write_dataset(dataset, out_dir)

    m = dataset.manifest
    print(f"SYNTHETIC dataset written to {out_dir}")
    print(f"  counts:            {m['counts']}")
    print(
        f"  anomaly rows:      {m['anomaly_rows_by_type']}  (rate {m['anomaly_rate_realised']:.4f})"
    )
    print(f"  ledger issues:     {m['ledger_discrepancies_by_type']}")
    print(f"  kyc variations:    {m['kyc_variations_by_type']}")
    print(f"  entity relations:  {m['entity_relations']}")
    print(
        f"  validation:        ok={dataset.validation.ok} ({dataset.validation.checks_run} checks)"
    )
    for err in dataset.validation.errors:
        print(f"    ERROR: {err}", file=sys.stderr)
    return 0 if dataset.validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
