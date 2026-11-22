"""FastAPI surface for the ad-attribution skew-join planner.

Endpoints
---------
POST /plans            submit a batch; publishes a new plan version only if
                       every shard meets its limits (201), else 409 and the
                       previous version stays current
GET  /plans            list published versions (newest first)
GET  /plans/current    the currently active plan
GET  /plans/{version}  any previously published plan
GET  /health           liveness probe
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from .models import PlanMeta, PlanOut, PlanRequest
from .planner import (
    InfeasiblePlan,
    KeyLoad,
    PlanInputError,
    ShardLimit,
    build_plan,
)
from .store import PlanStore

DESCRIPTION = """规划广告归因作业的分桶（salting）方案。

* 值班员提交同批数据里各键的曝光行数、点击行数与原分片，并给出各片
  总行数 / 配对数上限。
* 单键超过原片任一上限时：小侧全量复制到各盐桶，大侧按行号连续切分，
  每对记录恰好配对一次；其余键留在原片，负载照计。
* 仅当所有分片都达标时才发布新版本；无解或发布失败时，SQLite 中的
  上一版照常可查。
"""


def create_app(store: PlanStore | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Default store is created at startup, not import time, so merely
        # importing the module never touches the filesystem.
        if app.state.store is None:
            app.state.store = PlanStore(os.environ.get("PLAN_DB", "plans.db"))
        yield

    app = FastAPI(
        title="Ad Attribution Skew-Join Planner",
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.store = store

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok"}

    @app.post(
        "/plans",
        response_model=PlanOut,
        status_code=201,
        tags=["plans"],
        summary="Submit a batch and publish a plan if all shards fit",
    )
    def submit_plan(req: PlanRequest, request: Request) -> dict:
        store: PlanStore = request.app.state.store
        keys = [
            KeyLoad(k.key, k.impressions, k.clicks, k.shard) for k in req.keys
        ]
        limits = {
            shard: ShardLimit(lim.max_rows, lim.max_pairs)
            for shard, lim in req.limits.items()
        }
        default = (
            ShardLimit(req.default_limit.max_rows, req.default_limit.max_pairs)
            if req.default_limit
            else None
        )
        try:
            plan = build_plan(keys, limits, default)
        except PlanInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except InfeasiblePlan as exc:
            # No solution: nothing is published, previous version stays live.
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "infeasible_plan",
                    "reason": str(exc),
                    "current_version": store.current_version(),
                },
            ) from exc

        payload = {"batch_id": req.batch_id, "request": req.model_dump(), **plan}
        try:
            version = store.publish(payload)
        except Exception as exc:  # publish failed: old version untouched
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "publish_failed",
                    "reason": str(exc),
                    "current_version": store.current_version(),
                },
            ) from exc
        return store.get(version)

    @app.get(
        "/plans",
        response_model=list[PlanMeta],
        tags=["plans"],
        summary="List published plan versions",
    )
    def list_plans(request: Request) -> list[dict]:
        return request.app.state.store.list()

    @app.get(
        "/plans/current",
        response_model=PlanOut,
        tags=["plans"],
        summary="Fetch the currently active plan",
    )
    def current_plan(request: Request) -> dict:
        plan = request.app.state.store.current()
        if plan is None:
            raise HTTPException(status_code=404, detail="no plan published yet")
        return plan

    @app.get(
        "/plans/{version}",
        response_model=PlanOut,
        tags=["plans"],
        summary="Fetch a specific published plan version",
    )
    def get_plan(version: int, request: Request) -> dict:
        plan = request.app.state.store.get(version)
        if plan is None:
            raise HTTPException(
                status_code=404, detail=f"no plan with version {version}"
            )
        return plan

    return app


app = create_app()
