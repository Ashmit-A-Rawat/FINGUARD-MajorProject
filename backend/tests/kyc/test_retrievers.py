import numpy as np
import pytest

from kyc.lexical.bm25 import BM25Index, char_trigrams
from kyc.lexical.entries import NameEntries
from kyc.lexical.fuzzy import fuzzy_scores

DOCS = [["ava", "adams"], ["ava", "brown"], ["noah", "adams"], ["mei", "zhang"]]


def test_bm25_rare_terms_outweigh_common_ones() -> None:
    index = BM25Index(DOCS)
    scores = index.raw_scores(["zhang"])
    assert scores.argmax() == 3 and scores[[0, 1, 2]].sum() == 0
    assert index.idf["zhang"] > index.idf["ava"]  # "ava" appears in 2 docs


def test_bm25_normalised_score_of_identical_document_is_one() -> None:
    index = BM25Index(DOCS)
    scores = index.normalized_scores(["ava", "adams"])
    assert scores[0] == pytest.approx(1.0)
    assert 0 < scores[1] < 1 and 0 < scores[2] < 1
    assert scores[3] == 0


def test_bm25_unseen_terms_lower_the_ceiling() -> None:
    index = BM25Index(DOCS)
    assert index.normalized_scores(["ava", "adams", "qqqq"]).max() < 1.0


def test_bm25_scores_bounded() -> None:
    index = BM25Index(DOCS)
    s = index.normalized_scores(["ava", "ava", "adams"])
    assert s.min() >= 0 and s.max() <= 1


def test_char_trigrams_survive_typos() -> None:
    docs = [char_trigrams("mohammed smith"), char_trigrams("noah brown")]
    index = BM25Index(docs)
    scores = index.normalized_scores(char_trigrams("mohamed smith"))  # typo
    assert scores.argmax() == 0 and scores[0] > 0.6


def test_fuzzy_scores_order() -> None:
    s = fuzzy_scores("ava adams", ["ava adams", "ava adam", "zhang mei"])
    assert s[0] == 1.0 and s[0] > s[1] > s[2]
    assert fuzzy_scores("adams ava", ["ava adams"])[0] == 1.0  # token order ignored


def test_name_entries_take_best_alias(benchmark) -> None:  # type: ignore[no-untyped-def]
    entries = NameEntries.from_customers(benchmark.customers)
    assert len(entries.starts) == len(benchmark.customers)
    values = np.arange(len(entries.texts), dtype=float)
    per_customer = entries.customer_max(values)
    ends = list(entries.starts[1:]) + [len(entries.texts)]
    assert all(per_customer[i] == ends[i] - 1 for i in range(len(ends)))
