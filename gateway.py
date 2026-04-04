#!/usr/bin/env python3
"""
Gateway — routes API requests to the right GPU service.
Each service manages its own GPU lifecycle (load/unload on demand).
"""
import http.server, json, urllib.request, urllib.error, os, sys

PROXY_PORT = int(os.getenv("GATEWAY_PORT", "8080"))

# (host_port, API path prefix)
SERVICES = {
    "qwen":    (19080, "/v1/chat"),
    "whisper": (19081, "/v1/audio"),
    "timesfm": (19082, "/v1/forecast"),
}


def is_healthy(port, path="/v1/models"):
    try:
        r = urllib.request.urlopen(f"http://localhost:{port}{path}", timeout=2)
        return r.status == 200
    except Exception:
        return False


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def _proxy(self, port):
        cl = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(cl) if cl else None
        url = f"http://localhost:{port}{self.path}"
        try:
            req = urllib.request.Request(url, data=body, headers=dict(self.headers),
                                         method=self.command)
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

    def _route(self):
        for name, (port, prefix) in SERVICES.items():
            if self.path.startswith(prefix):
                return self._proxy(port)
        self.send_error(404)

    def do_POST(self):   self._route()
    def do_GET(self):
        if self.path == "/health":
            active = {n: is_healthy(p) for n, (p, _) in SERVICES.items()}
            body = json.dumps({"status": "ok", "active": active}).encode()
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
    print(f"[gateway] Router on :{PROXY_PORT}", flush=True, file=sys.stderr)
    httpd = http.server.HTTPServer(("0.0.0.0", PROXY_PORT), Handler)
    httpd.serve_forever()
