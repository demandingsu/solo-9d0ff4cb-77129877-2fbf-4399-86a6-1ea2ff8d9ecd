"""Tests for the shard-bounded two-phase salt-bucket trial placement.

Covers the optimization where a single first-fit trial must use a number of
capacity computations and temporary records bounded by the shard count, not
by the bucket count; failed trials consume no remaining capacity; and
per-bucket detail is expanded only after the whole batch is feasible.
"""

import pytest

from app.planner import (
    InfeasiblePlan,
    KeyLoad,
    ShardLimit,
    _fill_phase,
    _try_place,
    build_plan,
)


def test_big_bucket_spills_out_small_bucket_returns_to_original_shard():
    # s1 keeps 6 rows of room (10-row limit minus a 4-row cold key).
    # Hot key: 11 impressions x 1 click, m=1, minimal B = ceil(11/9) = 2,
    # split sizes [6, 5] so bucket needs are 7 rows and 6 rows.
    #   bucket 0 (7 rows) does not fit s1 (6 left) -> spills to s2;
    #   bucket 1 (6 rows) fits s1 exactly          -> returns to s1.
    plan = build_plan(
        [
            KeyLoad("cold", 4, 0, "s1"),
            KeyLoad("hot", 11, 1, "s1"),
        ],
        {s: ShardLimit(10, 100) for s in ("s1", "s2", "s3")},
    )
    (rule,) = plan["splits"]
    assert [b["shard"] for b in rule["buckets"]] == ["s2", "s1"]
    assert [b["partitioned_rows"] for b in rule["buckets"]] == [6, 5]
    assert (rule["buckets"][0]["row_start"], rule["buckets"][0]["row_end"]) == (1, 6)
    assert (rule["buckets"][1]["row_start"], rule["buckets"][1]["row_end"]) == (7, 11)
    loads = {l["shard"]: l for l in plan["shard_loads"]}
    assert (loads["s1"]["rows"], loads["s1"]["pairs"]) == (10, 5)
    assert (loads["s2"]["rows"], loads["s2"]["pairs"]) == (7, 6)
    assert loads["s3"]["hosted_buckets"] == 0
    assert all(l["within_limits"] for l in plan["shard_loads"])


def test_large_batch_fails_on_last_hot_key_and_state_is_not_published():
    # Two large hot keys that together place 6 salt buckets across three
    # shards, then a last hot key (smallest pair output, hence processed
    # last in the deterministic hot-key order) for which no room remains.
    limits = {s: ShardLimit(50, 100) for s in ("s1", "s2", "s3")}
    placed = [
        KeyLoad("big-a", 60, 2, "s1"),  # 120 pairs, B=2
        KeyLoad("big-b", 50, 2, "s2"),  # 100 pairs, B=4
    ]
    ok = build_plan(placed, limits)
    assert [r["bucket_count"] for r in ok["splits"]] == [2, 4]
    assert sum(r["bucket_count"] for r in ok["splits"]) == 6

    last = KeyLoad("last", 55, 1, "s1")  # 55 pairs < 100 -> processed last
    with pytest.raises(InfeasiblePlan, match="key 'last'"):
        build_plan(placed + [last], limits)


def test_failed_trial_consumes_no_remaining_capacity():
    # A trial that cannot host every bucket must leave the committed shard
    # state untouched, so the bucket-count retry starts from the same margin.
    from app.planner import _ShardState

    shard_states = {
        s: _ShardState(ShardLimit(50, 100)) for s in ("s1", "s2", "s3")
    }
    before = {s: (st.rows_left, st.pairs_left) for s, st in shard_states.items()}
    preference = ["s1", "s2", "s3"]
    # m=49 means one bucket already needs 50 rows and emits 49 pairs; an
    # M=10000 split cannot be hosted at the tried bucket counts -> None.
    result = _try_place(49, 10_000, 100, shard_states, preference)
    assert result is None
    after = {s: (st.rows_left, st.pairs_left) for s, st in shard_states.items()}
    assert before == after
    assert all(st.hosted_buckets == 0 for st in shard_states.values())


def test_trial_work_is_bounded_by_shard_count_not_bucket_count(monkeypatch):
    # Instrument the closed-form phase filler: with S shards it may be called
    # at most twice per trial (large-bucket phase, small-bucket phase) and the
    # landing records must never carry one entry per bucket.
    calls = []
    orig = _fill_phase

    def spy(need_rows, need_pairs, count, remaining, preference):
        calls.append((count, len(preference), len(remaining)))
        return orig(need_rows, need_pairs, count, remaining, preference)

    monkeypatch.setattr("app.planner._fill_phase", spy)
    # Heterogeneous limits: a tiny original shard forces 40 buckets while
    # only 3 shards exist, so bucket count far exceeds the shard count.
    limits = {
        "s1": ShardLimit(6, 10_000),
        "s2": ShardLimit(500, 10_000),
        "s3": ShardLimit(500, 10_000),
    }
    plan = build_plan([KeyLoad("hot", 200, 1, "s1")], limits)
    (rule,) = plan["splits"]
    bucket_count = rule["bucket_count"]
    assert bucket_count == 40  # >> 3 shards
    # two phase calls per trial; every call scans <= shard-count shards and
    # touches a remaining-capacity map sized by the shard count
    for count, shards, records in calls:
        assert shards <= 3
        assert records <= 3
    assert len(calls) % 2 == 0
    # the compact placement record lists shards, never individual buckets
    from app.planner import _place_hot_key, _ShardState

    states = {s: _ShardState(lim) for s, lim in limits.items()}
    placement = _place_hot_key(KeyLoad("hot", 200, 1, "s1"), states)
    assert placement.bucket_count == bucket_count
    assert len(placement.landed) <= 3
    assert sum(big + small for _, big, small in placement.landed) == bucket_count


def test_expansion_only_happens_after_whole_batch_is_feasible(monkeypatch):
    from app import planner

    expanded = {"n": 0}
    orig = planner._expand_placement

    def counting_expand(k, p):
        expanded["n"] += 1
        return orig(k, p)

    monkeypatch.setattr(planner, "_expand_placement", counting_expand)
    limits = {s: ShardLimit(50, 100) for s in ("s1", "s2", "s3")}

    # feasible batch: one expansion per hot key
    plan = build_plan(
        [KeyLoad("big-a", 60, 2, "s1"), KeyLoad("big-b", 50, 2, "s2")], limits
    )
    assert expanded["n"] == len(plan["splits"]) == 2

    # infeasible batch ending on the last key: no expansion at all
    expanded["n"] = 0
    with pytest.raises(InfeasiblePlan):
        build_plan(
            [
                KeyLoad("big-a", 60, 2, "s1"),
                KeyLoad("big-b", 50, 2, "s2"),
                KeyLoad("last", 55, 1, "s1"),
            ],
            limits,
        )
    assert expanded["n"] == 0


def test_exhausted_pair_budget_raises_infeasible_not_zero_division():
    # Regression: the bucket-count retry used to divide by a shard's
    # pairs_left and crashed with ZeroDivisionError once a shard's pair
    # budget was spent (observed via randomized search against the old code).
    # It must now surface the proper InfeasiblePlan instead of a 500-class
    # crash, and a zero-pair key must never take the pair-budget branch.
    with pytest.raises(InfeasiblePlan):
        build_plan(
            [
                KeyLoad("a", 1, 0, "s1"),
                KeyLoad("b", 5, 5, "s1"),    # spends s1's 10-pair budget? no, 25 > 10
                KeyLoad("c", 60, 0, "s0"),   # zero-pair row burst, no room anywhere
            ],
            {
                "s0": ShardLimit(20, 300),
                "s1": ShardLimit(20, 10),
            },
        )

    # Zero-pair key on shards whose pair budget is exactly zero still plans.
    plan = build_plan(
        [
            KeyLoad("pairload", 10, 10, "s1"),  # 100 pairs == max_pairs of s1
            KeyLoad("rowburst", 15, 0, "s2"),   # hot by rows, zero pairs
        ],
        {"s1": ShardLimit(100, 100), "s2": ShardLimit(10, 100)},
    )
    (rule,) = plan["splits"]
    assert rule["replicated_rows"] == 0
    assert sum(b["partitioned_rows"] for b in rule["buckets"]) == 15
    assert all(l["within_limits"] for l in plan["shard_loads"])


def test_many_bucket_plan_matches_first_fit_semantics():
    # A tiny original shard forces a bucket count far above the shard count;
    # verify the two-phase closed form reproduces bucket-by-bucket first-fit:
    # contiguous ranges, each landing shard able to host every bucket it
    # received, totals conserved.
    limits = {
        "s1": ShardLimit(6, 10_000),
        "s2": ShardLimit(500, 10_000),
        "s3": ShardLimit(500, 10_000),
    }
    plan = build_plan([KeyLoad("hot", 200, 1, "s1")], limits)
    (rule,) = plan["splits"]
    assert rule["bucket_count"] == 40
    sizes = [b["partitioned_rows"] for b in rule["buckets"]]
    assert sum(sizes) == 200
    assert max(sizes) - min(sizes) <= 1
    ranges = [(b["row_start"], b["row_end"]) for b in rule["buckets"]]
    assert ranges[0][0] == 1 and ranges[-1][1] == 200
    for (lo, hi), (nlo, _) in zip(ranges, ranges[1:]):
        assert hi + 1 == nlo
    by_shard: dict[str, dict[str, int]] = {}
    for b in rule["buckets"]:
        agg = by_shard.setdefault(b["shard"], {"rows": 0, "pairs": 0, "n": 0})
        agg["rows"] += b["partitioned_rows"] + b["replicated_rows"]
        agg["pairs"] += b["pairs"]
        agg["n"] += 1
    for shard, agg in by_shard.items():
        assert agg["rows"] <= limits[shard].max_rows
        assert agg["pairs"] <= limits[shard].max_pairs
    loads = {l["shard"]: l for l in plan["shard_loads"]}
    assert sum(l["hosted_buckets"] for l in loads.values()) == rule["bucket_count"]
    assert all(l["within_limits"] for l in plan["shard_loads"])
