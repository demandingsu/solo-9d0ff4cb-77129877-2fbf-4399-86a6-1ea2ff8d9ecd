import pytest

from app.planner import (
    InfeasiblePlan,
    KeyLoad,
    PlanInputError,
    ShardLimit,
    build_plan,
    ceil_div,
    split_sizes,
)

LIM = ShardLimit(max_rows=20_000, max_pairs=300_000)


def limits(*shards: str) -> dict[str, ShardLimit]:
    return {s: LIM for s in shards}


def test_ceil_div_and_split_sizes():
    assert ceil_div(10, 3) == 4
    assert ceil_div(9, 3) == 3
    assert split_sizes(10, 3) == [4, 3, 3]
    assert split_sizes(6, 3) == [2, 2, 2]
    assert sum(split_sizes(1000, 7)) == 1000


def test_cold_keys_stay_on_original_shard():
    plan = build_plan(
        [
            KeyLoad("a", 100, 20, "s1"),
            KeyLoad("b", 300, 10, "s2"),
        ],
        limits("s1", "s2"),
    )
    assert plan["splits"] == []
    loads = {l["shard"]: l for l in plan["shard_loads"]}
    assert loads["s1"]["rows"] == 120 and loads["s1"]["pairs"] == 2000
    assert loads["s2"]["rows"] == 310 and loads["s2"]["pairs"] == 3000
    assert loads["s1"]["resident_keys"] == 1
    assert all(l["within_limits"] for l in plan["shard_loads"])


def test_hot_key_split_conserves_pairs_and_rows():
    # 10_000 impressions x 40 clicks = 400_000 pairs > 300_000 -> hot.
    plan = build_plan(
        [KeyLoad("hot", 10_000, 40, "s1")],
        limits("s1", "s2"),
    )
    (rule,) = plan["splits"]
    assert rule["key"] == "hot"
    assert rule["replicated_side"] == "clicks"  # smaller side replicated
    assert rule["partitioned_side"] == "impressions"
    assert rule["replicated_rows"] == 40
    # pairs conserved: every (impression, click) pair exactly once
    assert sum(b["pairs"] for b in rule["buckets"]) == 10_000 * 40
    # partitioned side fully covered by contiguous, non-overlapping ranges
    ranges = [(b["row_start"], b["row_end"]) for b in rule["buckets"]]
    assert ranges[0][0] == 1 and ranges[-1][1] == 10_000
    for (lo, hi), (nlo, _) in zip(ranges, ranges[1:]):
        assert hi + 1 == nlo
    assert sum(b["partitioned_rows"] for b in rule["buckets"]) == 10_000
    # every bucket fits its landing shard
    for b in rule["buckets"]:
        assert b["partitioned_rows"] + b["replicated_rows"] <= LIM.max_rows
        assert b["pairs"] <= LIM.max_pairs
    assert all(l["within_limits"] for l in plan["shard_loads"])


def test_bucket_count_matches_closed_form():
    # B = max(ceil(M/(max_rows-m)), ceil(M*m/max_pairs)) = max(1, 2) = 2
    plan = build_plan([KeyLoad("hot", 10_000, 40, "s1")], limits("s1", "s2"))
    (rule,) = plan["splits"]
    assert rule["bucket_count"] == 2
    assert [b["partitioned_rows"] for b in rule["buckets"]] == [5000, 5000]


def test_bucket_spills_to_other_shard_when_original_is_full():
    # s1 keeps one bucket (200k pairs); the second no longer fits s1's
    # remaining pair budget and must land on s2.
    plan = build_plan([KeyLoad("hot", 10_000, 40, "s1")], limits("s1", "s2"))
    (rule,) = plan["splits"]
    landed = sorted(b["shard"] for b in rule["buckets"])
    assert landed == ["s1", "s2"]
    loads = {l["shard"]: l for l in plan["shard_loads"]}
    assert loads["s1"]["pairs"] == 200_000
    assert loads["s2"]["pairs"] == 200_000
    assert loads["s1"]["hosted_buckets"] == 1
    assert loads["s2"]["hosted_buckets"] == 1


def test_impressions_replicated_when_they_are_smaller():
    plan = build_plan(
        [KeyLoad("hot", 30, 9_000, "s1")],  # 270k pairs <= 300k? no: rows ok,
        limits("s1", "s2"),                 # pairs 270_000 <= 300_000 -> cold!
    )
    assert plan["splits"] == []
    plan = build_plan(
        [KeyLoad("hot", 35, 9_000, "s1")],  # 315_000 pairs -> hot
        limits("s1", "s2"),
    )
    (rule,) = plan["splits"]
    assert rule["replicated_side"] == "impressions"
    assert rule["replicated_rows"] == 35
    assert rule["partitioned_side"] == "clicks"
    assert sum(b["pairs"] for b in rule["buckets"]) == 35 * 9_000


def test_zero_click_hot_key_still_splits():
    # impressions alone exceed max_rows; clicks side is empty (m = 0).
    plan = build_plan([KeyLoad("hot", 45_000, 0, "s1")], limits("s1", "s2", "s3"))
    (rule,) = plan["splits"]
    assert rule["replicated_rows"] == 0
    assert rule["bucket_count"] == 3  # ceil(45_000 / 20_000)
    assert sum(b["pairs"] for b in rule["buckets"]) == 0
    assert sum(b["partitioned_rows"] for b in rule["buckets"]) == 45_000


def test_infeasible_when_cold_keys_alone_overflow_shard():
    with pytest.raises(InfeasiblePlan, match="unsplit keys alone"):
        build_plan(
            [KeyLoad("a", 12_000, 0, "s1"), KeyLoad("b", 9_000, 0, "s1")],
            limits("s1", "s2"),
        )


def test_infeasible_when_smaller_side_exceeds_pair_limit():
    # m = 300_001 impressions: one click row already emits m pairs > max_pairs.
    with pytest.raises(InfeasiblePlan, match="cannot be salted"):
        build_plan([KeyLoad("hot", 300_001, 400_000, "s1")], limits("s1", "s2"))


def test_infeasible_when_no_shard_has_room():
    with pytest.raises(InfeasiblePlan, match="remaining shard capacity"):
            build_plan(
                [
                    KeyLoad("cold1", 19_000, 0, "s1"),
                    KeyLoad("cold2", 19_000, 0, "s2"),
                    KeyLoad("hot", 10_000, 40, "s1"),
                ],
                limits("s1", "s2"),
            )


def test_missing_shard_limit_rejected():
    with pytest.raises(PlanInputError, match="s9"):
        build_plan([KeyLoad("a", 1, 1, "s9")], limits("s1"))


def test_default_limit_covers_unlisted_shards():
    plan = build_plan(
        [KeyLoad("a", 1, 1, "s9")], limits("s1"), default_limit=LIM
    )
    shards = {l["shard"] for l in plan["shard_loads"]}
    assert shards == {"s1", "s9"}


def test_plan_is_deterministic():
    keys = [
        KeyLoad("hot-b", 10_000, 40, "s1"),
        KeyLoad("cold", 100, 5, "s2"),
        KeyLoad("hot-a", 8_000, 60, "s2"),
    ]
    first = build_plan(keys, limits("s1", "s2", "s3"))
    second = build_plan(keys, limits("s1", "s2", "s3"))
    assert first == second
