"""Schema validation with quarantine: rejected rows are kept with a reason, never dropped."""

from collections import Counter
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)

PRIMARY_KEYS = {
    "customers": "customer_id",
    "kyc_records": "document_id",
    "transactions": "transaction_id",
    "ledger": "ledger_id",
}


class Quarantined(BaseModel):
    table: str
    row_number: int  # 1-based data row in the source file
    reason: str
    detail: str
    record: dict[str, Any]


def _coerce(record: dict[str, str | None]) -> dict[str, Any]:
    out: dict[str, Any] = dict(record)
    flag = record.get("is_synthetic")
    if flag is not None and flag.casefold() == "true":
        out["is_synthetic"] = True
    return out


def validate_records(
    model: type[M],
    table: str,
    rows: list[dict[str, str | None]],
    exact_duplicates: Counter[str],
) -> tuple[list[M], list[Quarantined]]:
    key_field = PRIMARY_KEYS[table]
    accepted: list[M] = []
    quarantined: list[Quarantined] = []
    seen: dict[str, dict[str, str | None]] = {}
    for number, row in enumerate(rows, start=1):
        key = row.get(key_field)
        if key is None:
            quarantined.append(
                Quarantined(
                    table=table,
                    row_number=number,
                    reason="missing_primary_key",
                    detail=key_field,
                    record=row,
                )
            )
            continue
        if key in seen:
            if seen[key] == row:
                exact_duplicates[table] += 1
            else:
                quarantined.append(
                    Quarantined(
                        table=table,
                        row_number=number,
                        reason="duplicate_key_conflict",
                        detail=f"{key_field}={key} already seen with different content",
                        record=row,
                    )
                )
            continue
        seen[key] = row
        try:
            accepted.append(model.model_validate(_coerce(row)))
        except ValidationError as exc:
            first = exc.errors()[0]
            quarantined.append(
                Quarantined(
                    table=table,
                    row_number=number,
                    reason="schema_validation_failed",
                    detail=f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}",
                    record=row,
                )
            )
    return accepted, quarantined
