"""Skew-join (salting) planner for ad-attribution batches.

Pairing model
-------------
Within one key, impression rows and click rows are each numbered from 1 in
their original table order.  The join output of a key is the cartesian
product ``impressions x clicks``: every (impression, click) pair must be
produced exactly once.

Hot keys
--------
A key is *hot* when its own load exceeds either limit of its original
shard::

    impressions + clicks  > max_rows(original shard)   or
    impressions * clicks  > max_pairs(original shard)

A hot key is salted:

* the side with fewer rows (``m`` rows) is **replicated** to every bucket;
* the larger side (``M`` rows) is **partitioned** into ``B`` contiguous,
  row-number-ordered ranges, one range per bucket;
* each bucket lands on exactly one shard.

Because the large-side ranges partition ``1..M`` and the small side is
fully replicated, every (impression, click) pair is emitted by exactly one
bucket -- ``sum(bucket pairs) == M * m`` always holds.

The minimal bucket count follows from the original shard's limits::

    B = max(ceil(M / (max_rows - m)), ceil(M * m / max_pairs), 1)

so that every bucket satisfies ``rows = size + m <= max_rows`` and
``pairs = size * m <= max_pairs`` on its landing shard.  If the remaining
capacity of the original shard cannot host the buckets, ``B`` is raised to
the smallest value that fits some shard's *remaining* capacity, and buckets
spill over to other shards (first-fit, original shard preferred).

Cold keys stay on their original shard and their load is counted as usual.
A plan is publishable only when every shard ends within both limits;
otherwise :class:`InfeasiblePlan` is raised and nothing is published.

The whole algorithm is deterministic: identical input always yields an
identical plan, and every rule is reproducible from row numbers alone.
"""

from __future__ import annotations

from dataclasses import dataclass


class InfeasiblePlan(Exception):
    """No salting plan can bring every shard within its limits."""


class PlanInputError(ValueError):
    """The submission itself is inconsistent (e.g. missing shard limits)."""


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def split_sizes(total: int, buckets: int) -> list[int]:
    """Near-even deterministic split; earlier buckets take the remainder."""
    base, rem = divmod(total, buckets)
    return [base + 1 if i < rem else base for i in range(buckets)]


@dataclass(frozen=True)
class KeyLoad:
    key: str
    impressions: int
    clicks: int
    shard: str

    @property
    def rows(self) -> int:
        return self.impressions + self.clicks

    @property
    def pairs(self) -> int:
        return self.impressions * self.clicks


@dataclass(frozen=True)
class ShardLimit:
    max_rows: int
    max_pairs: int


@dataclass
class _ShardState:
    limit: ShardLimit
    rows_used: int = 0
    pairs_used: int = 0
    resident_keys: int = 0
    hosted_buckets: int = 0

    @property
    def rows_left(self) -> int:
        return self.limit.max_rows - self.rows_used

    @property
    def pairs_left(self) -> int:
        return self.limit.max_pairs - self.pairs_used


def build_plan(
    keys: list[KeyLoad],
    limits: dict[str, ShardLimit],
    default_limit: ShardLimit | None = None,
) -> dict:
    """Build a JSON-ready salting plan or raise :class:`InfeasiblePlan`.

    ``limits`` maps shard id -> limit.  Shards referenced by keys but missing
    from ``limits`` fall back to ``default_limit``; if neither exists the
    submission is rejected with :class:`PlanInputError`.
    """
    states: dict[str, _ShardState] = {
        shard: _ShardState(limit) for shard, limit in limits.items()
    }
    for k in keys:
        if k.shard not in states:
            if default_limit is None:
                raise PlanInputError(
                    f"no max_rows/max_pairs configured for shard {k.shard!r} "
                    "and no default_limit given"
                )
            states[k.shard] = _ShardState(default_limit)

    # --- classify keys: hot ones get salted, the rest stay put ------------
    hot: list[KeyLoad] = []
    for k in keys:
        st = states[k.shard]
        if k.rows > st.limit.max_rows or k.pairs > st.limit.max_pairs:
            hot.append(k)
        else:
            st.rows_used += k.rows
            st.pairs_used += k.pairs
            st.resident_keys += 1

    # Cold keys alone may already overflow a shard; salting cannot fix that.
    for shard, st in states.items():
        if st.rows_used > st.limit.max_rows or st.pairs_used > st.limit.max_pairs:
            raise InfeasiblePlan(
                f"shard {shard!r} is over capacity from unsplit keys alone "
                f"(rows {st.rows_used}/{st.limit.max_rows}, "
                f"pairs {st.pairs_used}/{st.limit.max_pairs}); "
                "no salting plan can fix this"
            )

    # Deterministic processing order: biggest join output first.
    hot.sort(key=lambda k: (-k.pairs, -k.rows, k.key))
    splits = [_place_hot_key(k, states) for k in hot]

    # --- final gate: publish only if every shard meets its targets --------
    loads = []
    for shard in sorted(states):
        st = states[shard]
        within = (
            st.rows_used <= st.limit.max_rows
            and st.pairs_used <= st.limit.max_pairs
        )
        if not within:  # defensive: placement must already guarantee this
            raise InfeasiblePlan(
                f"shard {shard!r} over capacity after bucket placement "
                f"(rows {st.rows_used}/{st.limit.max_rows}, "
                f"pairs {st.pairs_used}/{st.limit.max_pairs})"
            )
        loads.append(
            {
                "shard": shard,
                "rows": st.rows_used,
                "pairs": st.pairs_used,
                "max_rows": st.limit.max_rows,
                "max_pairs": st.limit.max_pairs,
                "resident_keys": st.resident_keys,
                "hosted_buckets": st.hosted_buckets,
                "within_limits": within,
            }
        )
    return {"splits": splits, "shard_loads": loads}


def _place_hot_key(k: KeyLoad, states: dict[str, _ShardState]) -> dict:
    """Salt one hot key: pick bucket count, split ranges, land buckets."""
    orig = states[k.shard]
    max_rows, max_pairs = orig.limit.max_rows, orig.limit.max_pairs

    # Replicate the smaller side; ties replicate clicks (deterministic).
    if k.impressions >= k.clicks:
        replicated_side, partitioned_side = "clicks", "impressions"
        m, M = k.clicks, k.impressions
    else:
        replicated_side, partitioned_side = "impressions", "clicks"
        m, M = k.impressions, k.clicks

    if m > max_pairs:
        raise InfeasiblePlan(
            f"key {k.key!r}: smaller side has {m} rows, so even a bucket with a "
            f"single {partitioned_side} row emits {m} pairs > max_pairs={max_pairs} "
            f"of shard {k.shard!r}; key cannot be salted within these limits"
        )
    if m >= max_rows:
        raise InfeasiblePlan(
            f"key {k.key!r}: smaller side has {m} rows, so one bucket needs at "
            f"least {m + 1} rows > max_rows={max_rows} of shard {k.shard!r}; "
            "key cannot be salted within these limits"
        )

    buckets = max(ceil_div(M, max_rows - m), ceil_div(M * m, max_pairs), 1)
    # Landing preference: keep data local if possible, then ascend by shard id.
    preference = [k.shard] + sorted(s for s in states if s != k.shard)

    while True:
        sizes = split_sizes(M, buckets)
        # Simulate placement on remaining capacity; commit only on success.
        sim = {s: [st.rows_left, st.pairs_left] for s, st in states.items()}
        assignment: list[str] | None = []
        for size in sizes:
            need_rows, need_pairs = size + m, size * m
            target = next(
                (
                    s
                    for s in preference
                    if sim[s][0] >= need_rows and sim[s][1] >= need_pairs
                ),
                None,
            )
            if target is None:
                assignment = None
                break
            sim[target][0] -= need_rows
            sim[target][1] -= need_pairs
            assignment.append(target)

        if assignment is not None:
            for shard, st in states.items():
                st.rows_used = st.limit.max_rows - sim[shard][0]
                st.pairs_used = st.limit.max_pairs - sim[shard][1]
            break

        # Smallest bucket count that would fit some shard's remaining capacity.
        nxt: int | None = None
        for st in states.values():
            rows_left, pairs_left = st.rows_left, st.pairs_left
            if m < rows_left and m <= pairs_left:
                need = max(
                    ceil_div(M, rows_left - m), ceil_div(M * m, pairs_left), 1
                )
                if need > buckets and (nxt is None or need < nxt):
                    nxt = need
        if nxt is None:
            raise InfeasiblePlan(
                f"key {k.key!r}: cannot place {M} {partitioned_side} rows with "
                f"the {m}-row {replicated_side} side replicated to each bucket; "
                f"remaining shard capacity is insufficient "
                f"(last tried {buckets} buckets)"
            )
        buckets = nxt

    out_buckets = []
    start = 1
    for idx, (size, shard) in enumerate(zip(sizes, assignment)):
        states[shard].hosted_buckets += 1
        out_buckets.append(
            {
                "bucket": idx,
                "shard": shard,
                "row_start": start,
                "row_end": start + size - 1,
                "partitioned_rows": size,
                "replicated_rows": m,
                "pairs": size * m,
            }
        )
        start += size

    return {
        "key": k.key,
        "original_shard": k.shard,
        "replicated_side": replicated_side,
        "replicated_rows": m,
        "partitioned_side": partitioned_side,
        "partitioned_rows": M,
        "bucket_count": len(out_buckets),
        "buckets": out_buckets,
    }
