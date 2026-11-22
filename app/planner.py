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

Trial placement complexity
--------------------------
For a fixed bucket count ``B`` the near-even split has only two distinct
bucket sizes: ``rem = M mod B`` buckets of size ``base + 1`` followed by
``B - rem`` buckets of size ``base`` (see :func:`split_sizes`).  First-fit
placement of identical buckets can therefore be solved per shard in closed
form: one shard takes ``min(buckets left, rows_left // need_rows,
pairs_left // need_pairs)`` buckets before the scan moves on.  A trial is
two such phases (large buckets, then small ones), so both the number of
capacity computations and the number of temporary records are bounded by
the shard count and never grow with the bucket count.

A trial mutates only a local snapshot of the shards' remaining capacity;
the committed shard state is touched solely on success, so failed trials
consume no margin and later hot keys are always tried against the capacity
left after the buckets already placed for earlier keys.  Per-bucket detail
(the row ranges and landing shard of every bucket) is expanded only after
the whole batch passes the final capacity gate.
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


@dataclass(frozen=True)
class _Placement:
    """Compact record of one salted key's committed placement.

    ``landed`` lists, in shard preference order, ``(shard, big, small)``:
    how many large (``base + 1`` rows) and small (``base`` rows) buckets the
    shard hosts.  Its size is bounded by the shard count, independent of the
    bucket count.  Per-bucket detail is rebuilt from it by
    :func:`_expand_placement` only once the whole batch is feasible.
    """

    m: int
    M: int
    base: int
    rem: int
    landed: tuple[tuple[str, int, int], ...]

    @property
    def bucket_count(self) -> int:
        return sum(big + small for _, big, small in self.landed)


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
    # Compact trial placement: each call commits only bucket counts per
    # shard, never per-bucket detail, and failed trials leave no trace.
    placements = [_place_hot_key(k, states) for k in hot]

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

    # The whole batch is feasible: only now expand per-bucket detail.
    splits = [_expand_placement(k, p) for k, p in zip(hot, placements)]
    return {"splits": splits, "shard_loads": loads}


def _fill_phase(
    need_rows: int,
    need_pairs: int,
    count: int,
    remaining: dict[str, list[int]],
    preference: list[str],
) -> dict[str, int] | None:
    """First-fit ``count`` identical buckets onto shards in preference order.

    Buckets are identical, so each shard takes as many as fit in one closed
    form (``capacity // need``) before the scan moves to the next shard --
    the exact shard sequence a bucket-by-bucket first-fit would produce, in
    O(shard count) steps.  Mutates the local ``remaining`` snapshot only and
    returns ``{shard: buckets taken}``, or ``None`` when the shards cannot
    host all ``count`` buckets.
    """
    if count == 0:
        return {}
    taken: dict[str, int] = {}
    placed = 0
    for shard in preference:
        rows_left, pairs_left = remaining[shard]
        capacity = rows_left // need_rows
        if need_pairs > 0:
            capacity = min(capacity, pairs_left // need_pairs)
        if capacity <= 0:
            continue
        n = min(count - placed, capacity)
        remaining[shard] = [
            rows_left - n * need_rows,
            pairs_left - n * need_pairs,
        ]
        taken[shard] = n
        placed += n
        if placed == count:
            return taken
    return None


def _try_place(
    m: int,
    M: int,
    buckets: int,
    states: dict[str, _ShardState],
    preference: list[str],
) -> tuple[dict[str, int], dict[str, int], dict[str, list[int]]] | None:
    """One first-fit trial at bucket count ``buckets`` on remaining capacity.

    Runs in O(shard count) time with O(shard count) temporary records,
    independent of ``buckets``.  Returns the per-shard large/small bucket
    counts and the remaining-capacity snapshot, or ``None`` on failure.
    The snapshot is local: a failed trial changes no committed state.
    """
    base, rem = divmod(M, buckets)
    remaining = {
        shard: [st.rows_left, st.pairs_left] for shard, st in states.items()
    }
    # Large buckets (size base + 1) come first in split_sizes order, so the
    # first-fit scan must host them before any small (size base) bucket.
    big = _fill_phase(
        base + 1 + m, (base + 1) * m, rem, remaining, preference
    )
    if big is None:
        return None
    small = _fill_phase(
        base + m, base * m, buckets - rem, remaining, preference
    )
    if small is None:
        return None
    return big, small, remaining


def _place_hot_key(k: KeyLoad, states: dict[str, _ShardState]) -> _Placement:
    """Salt one hot key: pick bucket count, trial-place, commit compactly."""
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
        trial = _try_place(m, M, buckets, states, preference)
        if trial is not None:
            big, small, remaining = trial
            # Commit: derive used load from the trial's remaining snapshot and
            # count hosted buckets.  Nothing is committed on a failed trial.
            landed: list[tuple[str, int, int]] = []
            for shard in preference:
                n_big = big.get(shard, 0)
                n_small = small.get(shard, 0)
                if n_big == 0 and n_small == 0:
                    continue
                st = states[shard]
                rows_left, pairs_left = remaining[shard]
                st.rows_used = st.limit.max_rows - rows_left
                st.pairs_used = st.limit.max_pairs - pairs_left
                st.hosted_buckets += n_big + n_small
                landed.append((shard, n_big, n_small))
            return _Placement(m, M, *divmod(M, buckets), tuple(landed))

        # Smallest bucket count that would fit some shard's remaining capacity.
        nxt: int | None = None
        for st in states.values():
            rows_left, pairs_left = st.rows_left, st.pairs_left
            if m < rows_left and m <= pairs_left:
                if M * m == 0:
                    need = max(ceil_div(M, rows_left - m), 1)
                else:
                    need = max(
                        ceil_div(M, rows_left - m),
                        ceil_div(M * m, pairs_left),
                        1,
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


def _expand_placement(k: KeyLoad, placement: _Placement) -> dict:
    """Rebuild the full per-bucket rule for one committed salted key.

    The shard sequence is the one first-fit produced: all large buckets in
    preference order, then all small buckets in preference order.  Row
    ranges follow :func:`split_sizes`, so bucket numbers, landing shards and
    row-number segments are identical to a bucket-by-bucket placement.
    """
    # Replicate the smaller side; ties replicate clicks (deterministic).
    if k.impressions >= k.clicks:
        replicated_side, partitioned_side = "clicks", "impressions"
    else:
        replicated_side, partitioned_side = "impressions", "clicks"

    m = placement.m
    base, rem = placement.base, placement.rem
    sequence: list[tuple[str, int]] = []
    for shard, n_big, _ in placement.landed:
        sequence.extend((shard, base + 1) for _ in range(n_big))
    for shard, _, n_small in placement.landed:
        sequence.extend((shard, base) for _ in range(n_small))

    out_buckets = []
    start = 1
    for idx, (shard, size) in enumerate(sequence):
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
        "partitioned_rows": placement.M,
        "bucket_count": len(out_buckets),
        "buckets": out_buckets,
    }
