#!/usr/bin/env python3
"""
GPU process wrapper — starts a server binary on first request,
stops it after idle timeout to free VRAM. Container stays running.
"""

import http.server, subprocess, time, threading
import urllib.request, urllib.error, os, sys

IDLE_TIMEOUT  = int(os.environ.get("IDLE_TIMEOUT", "300"))
BACKEND_PORT  = int(os.environ.get("BACKEND_PORT", "9090"))
LISTEN_PORT   = int(os.environ.get("LISTEN_PORT", "9080"))
HEALTH_PATH   = os.environ.get("HEALTH_PATH", "/v1/models")
START_TIMEOUT = int(os.environ.get("START_TIMEOUT", "120"))

server_cmd = sys.argv[1:]
process = None
last_active = 0.0
lock = threading.Lock()


def is_healthy():
    try:
        r = urllib.request.urlopen(
            f"http://localhost:{BACKEND_PORT}{HEALTH_PATH}", timeout=2
        )
        return r.status == 200
    except Exception:
        return False


def start_backend():
    global process
    if process and process.poll() is None and is_healthy():
        return True
    print("[wrapper] Starting GPU server…", flush=True)
    process = subprocess.Popen(server_cmd)
    for _ in range(START_TIMEOUT):
        if is_healthy():
            print("[wrapper] GPU server ready", flush=True)
            return True
        if process.poll() is not None:
            print("[wrapper] GPU server exited unexpectedly", flush=True)
            return False
        time.sleep(1)
    print("[wrapper] GPU server start timeout", flush=True)
    return False


def stop_backend():
    global process
    if process and process.poll() is None:
        print("[wrapper] Stopping GPU server (freeing VRAM)…", flush=True)
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process = None


def idle_reaper():
    while True:
        time.sleep(30)
        with lock:
            if last_active and process and process.poll() is None:
                if time.time() - last_active > IDLE_TIMEOUT:
                    print(
                        f"[wrapper] Idle {int(time.time() - last_active)}s, shutting down",
                        flush=True,
                    )
                    stop_backend()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _proxy(self):
        global last_active
        with lock:
            if not is_healthy():
                if not start_backend():
                    self.send_error(503, "GPU server failed to start")
                    return
            last_active = time.time()

        cl = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(cl) if cl else None

        url = f"http://localhost:{BACKEND_PORT}{self.path}"
        try:
            req = urllib.request.Request(
                url, data=body, headers=dict(self.headers), method=self.command
            )
            resp = urllib.request.urlopen(req, timeout=300)
            self.send_response(resp.status)
            for k, v in resp.headers.items():
                if k.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(resp.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_error(502, str(e))

    def do_GET(self):
        self._proxy()

    def do_POST(self):
        self._proxy()

    def do_PUT(self):
        self._proxy()

    def do_DELETE(self):
        self._proxy()

    def do_OPTIONS(self):
        self._proxy()

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()


if __name__ == "__main__":
    print(
        f"[wrapper] Listening :{LISTEN_PORT} → backend :{BACKEND_PORT}"
        f" | idle timeout {IDLE_TIMEOUT}s",
        flush=True,
    )
    threading.Thread(target=idle_reaper, daemon=True).start()
    httpd = http.server.HTTPServer(("0.0.0.0", LISTEN_PORT), Handler)
    httpd.serve_forever()
