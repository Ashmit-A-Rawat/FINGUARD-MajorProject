"""Raw ingestion: read CSV files as untyped strings after verifying dataset integrity."""

import csv
import hashlib
import json
from pathlib import Path

TABLE_FILES = {
    "customers": "customers.csv",
    "kyc_records": "kyc_records.csv",
    "transactions": "transactions.csv",
    "ledger": "ledger.csv",
}


class IntegrityError(RuntimeError):
    """Raised when a file does not match the checksum recorded in the dataset manifest."""


def verify_manifest(directory: Path) -> bool | None:
    """Return True if all recorded checksums match, None if there is no manifest."""
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return None
    expected: dict[str, str] = json.loads(manifest_path.read_text()).get("file_sha256", {})
    for filename, digest in expected.items():
        path = directory / filename
        if not path.exists():
            raise IntegrityError(f"{filename} listed in manifest but missing")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise IntegrityError(f"{filename} does not match its manifest checksum")
    return True


def read_raw_table(path: Path) -> list[dict[str, str | None]]:
    """Read every cell as a string; typing happens later, in validation."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
