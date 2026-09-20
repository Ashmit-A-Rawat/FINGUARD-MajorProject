"""Controlled name / address / DOB corruption used to create KYC variations."""

from __future__ import annotations

import string
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from data_pipeline.synthetic.reference_data import (
    CITIES,
    FIRST_NAME_VARIANTS,
    FIRST_NAMES,
    LAST_NAME_VARIANTS,
    STREET_SUFFIXES,
    STREETS,
)


@dataclass(frozen=True)
class Address:
    number: int
    street: str
    city: str
    country: str

    def __str__(self) -> str:
        return f"{self.number} {self.street}, {self.city}, {self.country}"


def full_name(first: str, middle: str | None, last: str) -> str:
    return " ".join(part for part in (first, middle, last) if part)


def _typo(name: str, rng: np.random.Generator) -> str:
    tokens = name.split()
    eligible = [i for i, t in enumerate(tokens) if len(t) >= 4]
    if not eligible:
        return name + "x"
    for _ in range(10):
        i = int(rng.choice(eligible))
        token = tokens[i]
        pos = int(rng.integers(1, len(token) - 1))
        op = int(rng.integers(0, 4))
        if op == 0:  # swap adjacent
            new = token[:pos] + token[pos + 1] + token[pos] + token[pos + 2 :]
        elif op == 1:  # delete
            new = token[:pos] + token[pos + 1 :]
        elif op == 2:  # duplicate
            new = token[:pos] + token[pos] + token[pos:]
        else:  # substitute
            new = token[:pos] + str(rng.choice(list(string.ascii_lowercase))) + token[pos + 1 :]
        if new != token:
            tokens[i] = new
            return " ".join(tokens)
    return name + "x"


def apply_name_variation(
    kind: str, first: str, middle: str | None, last: str, rng: np.random.Generator
) -> tuple[str, str]:
    """Return (varied name, variation actually applied).

    If the requested kind is impossible for this name (e.g. no transliteration exists),
    fall back to a typo and report that, so labels always describe what really happened.
    """
    base = full_name(first, middle, last)
    if kind == "abbreviation":
        options = [f"{first[0]}. {last}" if not middle else f"{first[0]}. {middle} {last}"]
        if middle:
            options.append(f"{first} {middle[0]}. {last}")
        return str(rng.choice(options)), "abbreviation"
    if kind == "transliteration":
        first_alts = FIRST_NAME_VARIANTS.get(first, [])
        last_alts = LAST_NAME_VARIANTS.get(last, [])
        if first_alts or last_alts:
            new_first = str(rng.choice(first_alts)) if first_alts else first
            new_last = str(rng.choice(last_alts)) if last_alts else last
            return full_name(new_first, middle, new_last), "transliteration"
        return _typo(base, rng), "typo"
    if kind == "middle_name_diff":
        if middle and rng.random() < 0.5:
            return full_name(first, None, last), "middle_name_diff"
        other = str(rng.choice([n for n in FIRST_NAMES if n != middle]))
        return full_name(first, other, last), "middle_name_diff"
    if kind == "name_order_swap":
        return f"{last}, {first}" + (f" {middle}" if middle else ""), "name_order_swap"
    return _typo(base, rng), "typo"


def alter_address(address: Address, rng: np.random.Generator) -> Address:
    """Return a different address in the same country."""
    choice = int(rng.integers(0, 3))
    if choice == 0:
        number = address.number + int(rng.integers(1, 40))
        return Address(number, address.street, address.city, address.country)
    if choice == 1:
        street = f"{rng.choice(STREETS)} {rng.choice(STREET_SUFFIXES)}"
        if street == address.street:
            street += " East"
        return Address(address.number, street, address.city, address.country)
    others = [c for c in CITIES[address.country] if c != address.city]
    return Address(address.number, address.street, str(rng.choice(others)), address.country)


def conflicting_dob(dob: date, rng: np.random.Generator) -> date:
    """Return a DOB that differs from ``dob`` (day/month swap or year shift)."""
    if dob.day <= 12 and dob.day != dob.month and rng.random() < 0.5:
        return date(dob.year, dob.day, dob.month)  # day/month transposition
    shift = int(rng.integers(1, 4)) * (1 if rng.random() < 0.5 else -1)
    return dob + timedelta(days=365 * shift)
