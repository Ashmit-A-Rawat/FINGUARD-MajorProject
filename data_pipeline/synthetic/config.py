"""Generator configuration and size presets."""

from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel, Field, model_validator

ANOMALY_TYPES = (
    "high_value",
    "burst",
    "unusual_time",
    "geo_change",
    "new_counterparty",
    "abnormal_velocity",
    "repeated_transfers",
)
# Anomaly types realised by inserting new rows (episodes) rather than editing existing rows.
INSERTED_ANOMALY_TYPES = ("burst", "abnormal_velocity", "repeated_transfers")

KYC_VARIATIONS = (
    "none",
    "typo",
    "abbreviation",
    "transliteration",
    "middle_name_diff",
    "name_order_swap",
    "address_mismatch",
    "dob_conflict",
)

LEDGER_DISCREPANCIES = (
    "contradictory_amount",
    "missing_reference",
    "currency_mismatch",
    "missing_ledger_entry",
    "duplicate_posting",
    "late_posting",
)


class GeneratorConfig(BaseModel):
    preset: str = "custom"
    seed: int = 42
    n_customers: int = Field(1000, ge=50)
    n_transactions: int = Field(20_000, ge=500)
    window_end: date = date(2025, 6, 30)  # fixed, never "now": keeps runs reproducible
    window_days: int = Field(180, ge=60)

    anomaly_rate: float = Field(0.02, ge=0.0, le=0.2)
    anomaly_mix: dict[str, float] = Field(
        default_factory=lambda: {
            "high_value": 0.20,
            "burst": 0.15,
            "unusual_time": 0.15,
            "geo_change": 0.12,
            "new_counterparty": 0.12,
            "abnormal_velocity": 0.10,
            "repeated_transfers": 0.16,
        }
    )

    duplicate_rate: float = Field(0.02, ge=0.0, le=0.2)
    namesake_rate: float = Field(0.01, ge=0.0, le=0.2)
    kyc_variation_mix: dict[str, float] = Field(
        default_factory=lambda: {
            "none": 0.79,
            "typo": 0.04,
            "abbreviation": 0.03,
            "transliteration": 0.03,
            "middle_name_diff": 0.03,
            "name_order_swap": 0.02,
            "address_mismatch": 0.05,
            "dob_conflict": 0.01,
        }
    )

    ledger_discrepancy_rate: float = Field(0.015, ge=0.0, le=0.2)
    ledger_discrepancy_mix: dict[str, float] = Field(
        default_factory=lambda: {
            "contradictory_amount": 0.30,
            "missing_reference": 0.20,
            "currency_mismatch": 0.10,
            "missing_ledger_entry": 0.15,
            "duplicate_posting": 0.15,
            "late_posting": 0.10,
        }
    )

    @property
    def window_start(self) -> date:
        return self.window_end - timedelta(days=self.window_days)

    @model_validator(mode="after")
    def _check_mixes(self) -> GeneratorConfig:
        for name, mix, keys in (
            ("anomaly_mix", self.anomaly_mix, ANOMALY_TYPES),
            ("kyc_variation_mix", self.kyc_variation_mix, KYC_VARIATIONS),
            ("ledger_discrepancy_mix", self.ledger_discrepancy_mix, LEDGER_DISCREPANCIES),
        ):
            if set(mix) != set(keys):
                raise ValueError(f"{name} keys must be exactly {sorted(keys)}")
            if abs(sum(mix.values()) - 1.0) > 1e-6:
                raise ValueError(f"{name} must sum to 1.0, got {sum(mix.values())}")
        return self


PRESETS: dict[str, tuple[int, int]] = {
    "small": (1_000, 20_000),
    "medium": (10_000, 250_000),
    "large": (100_000, 1_000_000),
}


def config_for_preset(preset: str, seed: int = 42) -> GeneratorConfig:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    n_customers, n_transactions = PRESETS[preset]
    return GeneratorConfig(
        preset=preset, seed=seed, n_customers=n_customers, n_transactions=n_transactions
    )
