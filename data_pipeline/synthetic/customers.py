"""Customer, KYC-record and entity-truth generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from data_pipeline.synthetic.config import KYC_VARIATIONS, GeneratorConfig
from data_pipeline.synthetic.reference_data import (
    ACCOUNT_TYPE_MULTIPLIER,
    CITIES,
    FIRST_NAME_VARIANTS,
    FIRST_NAMES,
    HOME_COUNTRIES,
    LAST_NAMES,
    OCCUPATIONS,
    STREET_SUFFIXES,
    STREETS,
)
from data_pipeline.synthetic.variations import (
    Address,
    alter_address,
    apply_name_variation,
    conflicting_dob,
    full_name,
)

DOCUMENT_TYPES = ("passport", "national_id", "drivers_license")


@dataclass(frozen=True)
class Person:
    entity_key: int
    first: str
    middle: str | None
    last: str
    dob: date
    address: Address
    occupation: str
    account_type: str
    open_date: date
    relation: str = "unique"  # unique | duplicate_of | namesake_of
    related_key: int | None = None
    name_override: str | None = None  # duplicates may carry a varied name spelling

    @property
    def name(self) -> str:
        return self.name_override or full_name(self.first, self.middle, self.last)


def _new_person(key: int, cfg: GeneratorConfig, rng: np.random.Generator) -> Person:
    codes = [c for c, _ in HOME_COUNTRIES]
    weights = np.array([w for _, w in HOME_COUNTRIES])
    country = str(rng.choice(codes, p=weights / weights.sum()))
    address = Address(
        number=int(rng.integers(1, 999)),
        street=f"{rng.choice(STREETS)} {rng.choice(STREET_SUFFIXES)}",
        city=str(rng.choice(CITIES[country])),
        country=country,
    )
    middle = str(rng.choice(FIRST_NAMES)) if rng.random() < 0.7 else None
    age_days = int(rng.integers(19 * 365, 80 * 365))
    if rng.random() < 0.9:
        open_date = cfg.window_start - timedelta(days=int(rng.integers(30, 3650)))
    else:  # newly opened account inside the observation window
        open_date = cfg.window_start + timedelta(days=int(rng.integers(0, cfg.window_days - 30)))
    return Person(
        entity_key=key,
        first=str(rng.choice(FIRST_NAMES)),
        middle=middle,
        last=str(rng.choice(LAST_NAMES)),
        dob=cfg.window_end - timedelta(days=age_days),
        address=address,
        occupation=str(rng.choice(list(OCCUPATIONS))),
        account_type=str(rng.choice(list(ACCOUNT_TYPE_MULTIPLIER), p=[0.6, 0.25, 0.15])),
        open_date=open_date,
    )


def _make_duplicate(
    base: Person, key: int, cfg: GeneratorConfig, rng: np.random.Generator
) -> Person:
    kind = str(rng.choice(["transliteration", "abbreviation", "typo", "middle_name_diff"]))
    name, _ = apply_name_variation(kind, base.first, base.middle, base.last, rng)
    address = base.address if rng.random() < 0.5 else alter_address(base.address, rng)
    open_date = cfg.window_start - timedelta(days=int(rng.integers(30, 3650)))
    return replace(
        base,
        address=address,
        open_date=open_date,
        relation="duplicate_of",
        related_key=base.entity_key,
        name_override=name,
        entity_key=key,
    )


def _make_namesake(
    base: Person, key: int, cfg: GeneratorConfig, rng: np.random.Generator
) -> Person:
    other = _new_person(key, cfg, rng)
    return replace(
        other,
        first=base.first,
        last=base.last,
        relation="namesake_of",
        related_key=base.entity_key,
    )


def _alternate_names(person: Person, rng: np.random.Generator) -> list[str]:
    alts: list[str] = []
    if person.middle:
        alts.append(f"{person.first} {person.last}")
    alts.append(f"{person.first[0]}. {person.last}")
    variants = FIRST_NAME_VARIANTS.get(person.first)
    if variants and rng.random() < 0.3:
        alts.append(f"{rng.choice(variants)} {person.last}")
    return [a for a in dict.fromkeys(alts) if a != person.name]


def generate_customers(
    cfg: GeneratorConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (customers, kyc_records, kyc_labels, entity_labels).

    Duplicate and namesake customers are shuffled among the rest, and customer ids are
    assigned after shuffling so ids never reveal the entity structure.
    """
    n_dup = round(cfg.duplicate_rate * cfg.n_customers)
    n_same = round(cfg.namesake_rate * cfg.n_customers)
    n_base = cfg.n_customers - n_dup - n_same

    people = [_new_person(k, cfg, rng) for k in range(n_base)]
    key = n_base
    for i in rng.choice(n_base, n_dup, replace=False):
        people.append(_make_duplicate(people[int(i)], key, cfg, rng))
        key += 1
    for i in rng.choice(n_base, n_same, replace=False):
        people.append(_make_namesake(people[int(i)], key, cfg, rng))
        key += 1
    order = rng.permutation(len(people))
    people = [people[int(i)] for i in order]
    customer_id_of = {p.entity_key: f"CUST-{i:06d}" for i, p in enumerate(people)}

    variation_kinds = list(KYC_VARIATIONS)
    variation_p = np.array([cfg.kyc_variation_mix[k] for k in variation_kinds])
    customer_rows: list[dict[str, object]] = []
    kyc_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    entity_rows: list[dict[str, object]] = []

    for i, p in enumerate(people):
        cid = customer_id_of[p.entity_key]
        # -- KYC document (may carry a controlled variation) --
        requested = str(rng.choice(variation_kinds, p=variation_p))
        doc_name, doc_address, doc_dob, applied = p.name, p.address, p.dob, requested
        if requested in (
            "typo",
            "abbreviation",
            "transliteration",
            "middle_name_diff",
            "name_order_swap",
        ):
            doc_name, applied = apply_name_variation(requested, p.first, p.middle, p.last, rng)
        elif requested == "address_mismatch":
            doc_address = alter_address(p.address, rng)
        elif requested == "dob_conflict":
            doc_dob = conflicting_dob(p.dob, rng)
        issue = cfg.window_end - timedelta(days=int(rng.integers(30, 7 * 365)))
        expiry = issue + timedelta(days=365 * int(rng.choice([5, 10])))
        doc_type = str(rng.choice(DOCUMENT_TYPES))
        doc_number = (
            "".join(rng.choice(list("ABCDEFGHJKLMNPRSTUVWXYZ"), 2))
            + f"{int(rng.integers(0, 10**7)):07d}"
        )
        document_id = f"DOC-{i:07d}"
        kyc_rows.append(
            {
                "document_id": document_id,
                "customer_id": cid,
                "name": doc_name,
                "date_of_birth": doc_dob,
                "address": str(doc_address),
                "document_type": doc_type,
                "document_number": doc_number,
                "issue_date": issue,
                "expiry_date": expiry,
                "is_synthetic": True,
            }
        )
        label_rows.append(
            {"document_id": document_id, "customer_id": cid, "variation_type": applied}
        )

        # -- customer master record (independent of the KYC variation label) --
        if expiry < cfg.window_end:
            status = "expired"
        else:
            status = str(rng.choice(["verified", "pending", "rejected"], p=[0.90, 0.08, 0.02]))
        customer_rows.append(
            {
                "customer_id": cid,
                "name": p.name,
                "alternate_names": "|".join(_alternate_names(p, rng)),
                "date_of_birth": p.dob,
                "address": str(p.address),
                "country": p.address.country,
                "occupation": p.occupation,
                "account_type": p.account_type,
                "account_open_date": p.open_date,
                "account_age_days": (cfg.window_end - p.open_date).days,
                "kyc_status": status,
                "is_synthetic": True,
            }
        )
        entity_key = p.related_key if p.relation == "duplicate_of" else p.entity_key
        entity_rows.append(
            {
                "customer_id": cid,
                "entity_id": f"ENT-{entity_key:06d}",
                "relation": p.relation,
                "related_customer_id": customer_id_of[p.related_key]
                if p.related_key is not None
                else "",
            }
        )

    return (
        pd.DataFrame(customer_rows),
        pd.DataFrame(kyc_rows),
        pd.DataFrame(label_rows),
        pd.DataFrame(entity_rows),
    )
