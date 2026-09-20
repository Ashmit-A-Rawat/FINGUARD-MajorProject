from evaluation.benchmarks.kyc_benchmark import KYCBenchmark, build_kyc_benchmark


def test_benchmark_is_deterministic() -> None:
    a, b = build_kyc_benchmark(300, seed=5), build_kyc_benchmark(300, seed=5)
    assert [(q.query_id, q.split, q.gold_customer_ids) for q in a.queries] == [
        (q.query_id, q.split, q.gold_customer_ids) for q in b.queries
    ]


def test_splits_are_disjoint_by_entity(benchmark: KYCBenchmark) -> None:
    entity_splits: dict[frozenset[str], set[str]] = {}
    for q in benchmark.queries:
        entity_splits.setdefault(q.gold_customer_ids, set()).add(q.split)
    assert all(len(s) == 1 for s in entity_splits.values())
    assert {q.split for q in benchmark.queries} == {"train", "dev", "test"}


def test_gold_always_contains_query_customer_and_duplicates_are_gold(
    benchmark: KYCBenchmark,
) -> None:
    assert all(q.customer_id in q.gold_customer_ids for q in benchmark.queries)
    assert any(len(q.gold_customer_ids) > 1 for q in benchmark.queries)


def test_namesakes_are_negatives_not_gold(benchmark: KYCBenchmark) -> None:
    with_namesakes = [q for q in benchmark.queries if q.namesake_ids]
    assert with_namesakes
    for q in with_namesakes:
        assert not q.namesake_ids & q.gold_customer_ids


def test_all_hard_categories_present_and_labels_not_in_document(benchmark: KYCBenchmark) -> None:
    cats = {q.variation_type for q in benchmark.queries}
    assert {
        "none",
        "typo",
        "abbreviation",
        "middle_name_diff",
        "name_order_swap",
        "address_mismatch",
        "dob_conflict",
    } <= cats
    dumped = benchmark.queries[0].document.model_dump_json()
    assert "variation" not in dumped and "entity_id" not in dumped


def test_composition_counts_add_up(benchmark: KYCBenchmark) -> None:
    c = benchmark.composition()
    assert sum(c["queries_per_split"].values()) == c["n_queries"] == len(benchmark.customers)
