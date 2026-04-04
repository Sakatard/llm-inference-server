#!/usr/bin/env python3
"""
TimesFM 2.5 200M — FastAPI server with reactive GPU management.
Model lives on CPU when idle, moves to GPU on request, frees VRAM after timeout.
"""

import logging
import math
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
import timesfm
from fastapi import FastAPI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("timesfm")

PATCH_SIZE = 32
MAX_HORIZON = 512
IDLE_TIMEOUT = int(__import__("os").environ.get("IDLE_TIMEOUT", "300"))

model = None
_lock = threading.Lock()
_last_active = 0.0
_on_gpu = False
_compiled_contexts: set[int] = set()


def _to_gpu():
    global _on_gpu
    if not _on_gpu:
        log.info("Moving model to GPU…")
        model.model.cuda()
        _on_gpu = True


def _to_cpu():
    global _on_gpu, _compiled_contexts
    if _on_gpu:
        log.info("Moving model to CPU, freeing VRAM…")
        model.model.cpu()
        torch.cuda.empty_cache()
        _on_gpu = False
        _compiled_contexts.clear()  # recompile needed after device change


def _idle_reaper():
    global _last_active
    while True:
        time.sleep(30)
        with _lock:
            if _on_gpu and _last_active and (time.time() - _last_active) > IDLE_TIMEOUT:
                log.info("Idle for %ds, offloading to CPU", int(time.time() - _last_active))
                _to_cpu()


def _compile_for_context(ctx: int):
    ctx = math.ceil(ctx / PATCH_SIZE) * PATCH_SIZE
    if ctx in _compiled_contexts:
        return
    log.info("Compiling for max_context=%d", ctx)
    model.compile(
        timesfm.ForecastConfig(
            max_context=ctx,
            max_horizon=MAX_HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=ctx + MAX_HORIZON <= 16384,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    _compiled_contexts.add(ctx)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    log.info("Loading TimesFM 2.5 200M (CPU)…")
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    # Keep on CPU — GPU loaded on first request
    model.model.cpu()
    torch.cuda.empty_cache()
    log.info("Model loaded on CPU, idle timeout=%ds", IDLE_TIMEOUT)
    threading.Thread(target=_idle_reaper, daemon=True).start()
    yield


app = FastAPI(title="TimesFM 2.5", lifespan=lifespan)


# ── Schemas ─────────────────────────────────────────────────────


class ForecastRequest(BaseModel):
    time_series: list[list[float]]
    horizon: int = Field(default=128, ge=1, le=512)


class ForecastResponse(BaseModel):
    point_forecast: list[list[float]]
    quantile_forecast: list[list[list[float]]]


# ── Endpoints ───────────────────────────────────────────────────


@app.get("/v1/models")
def list_models():
    return {"data": [{"id": "timesfm-2.5-200m", "object": "model", "on_gpu": _on_gpu}]}


@app.post("/v1/forecast")
def forecast(req: ForecastRequest):
    global _last_active
    with _lock:
        _to_gpu()
        _last_active = time.time()
        inputs = [np.array(ts, dtype=np.float32) for ts in req.time_series]
        max_len = max(len(ts) for ts in inputs)
        _compile_for_context(max_len)
        point, quantile = model.forecast(horizon=req.horizon, inputs=inputs)
    point = np.nan_to_num(point, nan=0.0, posinf=0.0, neginf=0.0)
    quantile = np.nan_to_num(quantile, nan=0.0, posinf=0.0, neginf=0.0)
    return ForecastResponse(
        point_forecast=point.tolist(),
        quantile_forecast=quantile.tolist(),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": "timesfm-2.5-200m", "on_gpu": _on_gpu}
