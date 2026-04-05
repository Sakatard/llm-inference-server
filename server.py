#!/usr/bin/env python3
"""
Unified GPU server — manages all models as subprocesses on Tesla P40.
No GPU libraries imported in this process → GPU stays at P8 when idle.
"""

import http.server, subprocess, time, threading, json, logging
import urllib.request, urllib.error, os, sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("server")

PORT = int(os.environ.get("PORT", "8080"))
IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT", "300"))
START_TIMEOUT = int(os.environ.get("START_TIMEOUT", "120"))


# ── Subprocess service ──────────────────────────────────────────


class Service:
    def __init__(self, name, cmd, port, health_path="/v1/models", path_rewrites=None):
        self.name = name
        self.cmd = cmd
        self.port = port
        self.health_path = health_path
        self.path_rewrites = path_rewrites or {}
        self.process = None
        self.last_active = 0.0
        self.lock = threading.Lock()

    def is_healthy(self):
        try:
            r = urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}{self.health_path}", timeout=2
            )
            return r.status == 200
        except Exception:
            return False

    def start(self):
        with self.lock:
            if self.process and self.process.poll() is None and self.is_healthy():
                return True
            log.info("[%s] Starting…", self.name)
            self.process = subprocess.Popen(
                self.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            for _ in range(START_TIMEOUT):
                if self.is_healthy():
                    log.info("[%s] Ready on :%d", self.name, self.port)
                    return True
                if self.process.poll() is not None:
                    err = self.process.stderr.read().decode()[-300:]
                    log.error("[%s] Exited: %s", self.name, err)
                    return False
                time.sleep(1)
            log.error("[%s] Start timeout", self.name)
            return False

    def stop(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                log.info("[%s] Stopping (freeing VRAM)", self.name)
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                self.process = None

    def proxy(self, path, headers, body, method):
        path = self.path_rewrites.get(path, path)
        url = f"http://127.0.0.1:{self.port}{path}"
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            resp = urllib.request.urlopen(req, timeout=300)
            return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()
        except Exception as e:
            return 502, {}, str(e).encode()

    def handle(self, path, headers, body, method):
        if not self.is_healthy():
            if not self.start():
                return 503, {}, b'{"error":"Service failed to start"}'
        self.last_active = time.time()
        return self.proxy(path, headers, body, method)

    @property
    def active(self):
        return self.process is not None and self.process.poll() is None


# ── Services ────────────────────────────────────────────────────


qwen = Service("qwen", [
    "llama-server",
    "-m", "/models/Qwen3.5-0.8B-Q5_K_M.gguf",
    "--mmproj", "/models/mmproj-F32.gguf",
    "-c", "32768",
    "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
    "-ngl", "99", "--n-gpu-layers", "99",
    "--port", "9180", "--host", "127.0.0.1", "--no-mmap",
], port=9180)

whisper = Service("whisper", [
    "whisper-server",
    "-m", "/models/ggml-base.en-q5_1.bin",
    "--port", "9181", "--host", "127.0.0.1",
], port=9181, health_path="/health",
   path_rewrites={"/v1/audio/transcriptions": "/inference"})

timesfm = Service("timesfm", [
    "python3", "/app/timesfm_worker.py",
], port=9182)

ROUTES = [
    ("/v1/chat",     qwen),
    ("/v1/audio",    whisper),
    ("/v1/forecast", timesfm),
]


# ── Idle reaper ─────────────────────────────────────────────────


def idle_reaper():
    while True:
        time.sleep(30)
        now = time.time()
        for svc in (qwen, whisper, timesfm):
            if svc.active and svc.last_active and now - svc.last_active > IDLE_TIMEOUT:
                svc.stop()


# ── HTTP handler ────────────────────────────────────────────────


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def _body(self):
        cl = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(cl) if cl else None

    def _send(self, code, headers, body):
        self.send_response(code)
        for k, v in headers.items():
            if k.lower() not in ("transfer-encoding", "connection"):
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _route(self):
        body = self._body()
        for prefix, svc in ROUTES:
            if self.path.startswith(prefix):
                return self._send(*svc.handle(
                    self.path, dict(self.headers), body, self.command
                ))
        self.send_error(404)

    def do_POST(self):   self._route()
    def do_GET(self):
        if self.path == "/health":
            status = {s.name: s.active for _, s in ROUTES}
            body = json.dumps({"status": "ok", "active": status}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        self._route()
    def do_PUT(self):    self._route()
    def do_DELETE(self): self._route()
    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()


if __name__ == "__main__":
    threading.Thread(target=idle_reaper, daemon=True).start()
    log.info("Listening on :%d — idle timeout %ds", PORT, IDLE_TIMEOUT)
    httpd = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
    httpd.serve_forever()
