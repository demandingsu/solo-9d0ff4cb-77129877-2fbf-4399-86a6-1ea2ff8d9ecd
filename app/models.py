"""Request/response schemas for the planning API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class KeyLoadIn(BaseModel):
    """Row counts for one (device, campaign) key in the submitted batch."""

    key: str = Field(min_length=1, max_length=512, examples=["dev-42|camp-7"])
    impressions: int = Field(ge=0, examples=[10000])
    clicks: int = Field(ge=0, examples=[40])
    shard: str = Field(min_length=1, max_length=128, examples=["s1"])


class ShardLimitIn(BaseModel):
    """Capacity limits for one shard."""

    max_rows: int = Field(gt=0, examples=[20000])
    max_pairs: int = Field(gt=0, examples=[300000])


class PlanRequest(BaseModel):
    """One batch submission: per-key counts plus per-shard limits."""

    batch_id: str = Field(min_length=1, max_length=256, examples=["nightly-2026-11-22"])
    keys: list[KeyLoadIn] = Field(min_length=1)
    limits: dict[str, ShardLimitIn] = Field(default_factory=dict)
    default_limit: ShardLimitIn | None = Field(
        default=None,
        description="Fallback limit for shards not listed in `limits`.",
    )

    @field_validator("keys")
    @classmethod
    def _unique_keys(cls, v: list[KeyLoadIn]) -> list[KeyLoadIn]:
        seen: set[str] = set()
        for k in v:
            if k.key in seen:
                raise ValueError(f"duplicate key {k.key!r} in batch")
            seen.add(k.key)
        return v


class BucketOut(BaseModel):
    """One salt bucket: a contiguous row range of the partitioned side,
    landing on one shard, joined against the fully replicated smaller side."""

    bucket: int
    shard: str
    row_start: int = Field(description="1-based first row of the partitioned side")
    row_end: int = Field(description="1-based last row (inclusive)")
    partitioned_rows: int
    replicated_rows: int
    pairs: int


class SplitRuleOut(BaseModel):
    key: str
    original_shard: str
    replicated_side: Literal["impressions", "clicks"]
    replicated_rows: int
    partitioned_side: Literal["impressions", "clicks"]
    partitioned_rows: int
    bucket_count: int
    buckets: list[BucketOut]


class ShardLoadOut(BaseModel):
    shard: str
    rows: int
    pairs: int
    max_rows: int
    max_pairs: int
    resident_keys: int
    hosted_buckets: int
    within_limits: bool


class PlanOut(BaseModel):
    version: int
    batch_id: str
    created_at: str
    request: PlanRequest
    splits: list[SplitRuleOut]
    shard_loads: list[ShardLoadOut]


class PlanMeta(BaseModel):
    version: int
    batch_id: str
    created_at: str
