#!/usr/bin/env python3
"""TimesFM 2.5 worker — runs as subprocess, killed when idle to free GPU."""

import json, math, logging, os
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import torch
import timesfm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("timesfm")

PORT = int(os.environ.get("TIMESFM_PORT", "9182"))
PATCH_SIZE = 32
MAX_HORIZON = 512

model = None
_compiled = set()


def load_model():
    global model
    log.info("Loading TimesFM 2.5 200M…")
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    # Warmup
    _compile_for(64)
    model.forecast(horizon=1, inputs=[np.ones(32)])
    log.info("TimesFM ready on :%d", PORT)


def _compile_for(ctx):
    ctx = math.ceil(ctx / PATCH_SIZE) * PATCH_SIZE
    if ctx not in _compiled:
        model.compile(timesfm.ForecastConfig(
            max_context=ctx, max_horizon=MAX_HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=ctx + MAX_HORIZON <= 16384,
            force_flip_invariance=True, infer_is_positive=True,
            fix_quantile_crossing=True,
        ))
        _compiled.add(ctx)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "timesfm-2.5-200m"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self):
        cl = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(cl) if cl else b"{}"
        try:
            req = json.loads(raw)
            series = req["time_series"]
            horizon = min(int(req.get("horizon", 128)), MAX_HORIZON)

            inputs = [np.array(ts, dtype=np.float32) for ts in series]
            _compile_for(max(len(ts) for ts in inputs))
            point, quantile = model.forecast(horizon=horizon, inputs=inputs)

            point = np.nan_to_num(point, nan=0.0, posinf=0.0, neginf=0.0)
            quantile = np.nan_to_num(quantile, nan=0.0, posinf=0.0, neginf=0.0)
            body = json.dumps({
                "point_forecast": point.tolist(),
                "quantile_forecast": quantile.tolist(),
            }).encode()
            self.send_response(200)
        except Exception as e:
            body = json.dumps({"error": str(e)}).encode()
            self.send_response(400)

        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    load_model()
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
