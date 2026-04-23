# GPU Server API Reference

Single-container GPU inference server on a Tesla P40.

The server exposes multiple routed backends behind one API surface:
- `qwen` for default chat
- `qwen-planner` for planner-oriented chat requests
- `whisper` for OpenAI-compatible audio transcription
- `timesfm` for general-purpose time-series forecasting
- `qwen-transcribe` in the model registry for routed compatibility

**Base URL:** `http://<host>:8088`

Models load on first use and unload after the configured idle timeout. The application default is `300` seconds if `IDLE_TIMEOUT` is unset; the current `docker-compose.yaml` sets `IDLE_TIMEOUT=1200` (20 minutes).

---

## Health Check

```
GET /health
```

Returns which backend subprocesses are currently active/running.

```bash
curl http://localhost:8088/health
```

```json
{
  "status": "ok",
  "active": {
    "qwen": false,
    "qwen-transcribe": false,
    "whisper": false,
    "timesfm": false,
    "qwen-planner": false
  }
}
```

`active: true` means that backend subprocess is currently running. `false` means it is not running and the next request will cold-start it.

---

## Model Registry

```
GET /v1/models
```

OpenAI-style model listing for the server's routed backends.

```bash
curl http://localhost:8088/v1/models
```

```json
{
  "object": "list",
  "data": [
    {"id": "qwen", "object": "model", "owned_by": "local"},
    {"id": "qwen-transcribe", "object": "model", "owned_by": "local"},
    {"id": "whisper", "object": "model", "owned_by": "local"},
    {"id": "timesfm", "object": "model", "owned_by": "local"},
    {"id": "qwen-planner", "object": "model", "owned_by": "local"}
  ]
}
```

Operator notes:
- `qwen` is the default chat target.
- `qwen-planner` is an alternate chat target selected via the request `model` field.
- `whisper` and `timesfm` are endpoint-specific backends.
- `qwen-transcribe` appears in the registry for routed transcription compatibility. It is not selected through `/v1/chat/completions`; chat model routing currently recognizes `qwen` and `qwen-planner`, while transcription uses its dedicated route.

---

## Chat Completions

OpenAI-compatible chat API.

Operator note: requests on the default `qwen` route are not forwarded fully unchanged. If the first message is not a `system` message and `/app/system-prompt.txt` exists, the proxy prepends that file as a system prompt before sending the request upstream. `qwen-planner` requests are routed separately and are not selected by that default fallback path.

```
POST /v1/chat/completions
```

The server routes chat requests by the JSON `model` field:
- `qwen` -> default chat backend
- `qwen-planner` -> planner backend
- omitted or unrecognized `model` -> `qwen`

### Request

```bash
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

Planner selection example:

```bash
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-planner",
    "messages": [
      {"role": "user", "content": "Create a step-by-step rollout plan for a model migration."}
    ]
  }'
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `qwen` | Supported chat selections: `qwen`, `qwen-planner`. Omitted or unknown values route to `qwen`. |
| `messages` | array | required | Array of `{role, content}` objects |
| `max_tokens` | int | none at proxy layer | Optional generation cap; forwarded as-is when provided, otherwise the selected backend decides |
| `temperature` | float | none at proxy layer | Optional sampling temperature; forwarded as-is when provided |
| `top_p` | float | none at proxy layer | Optional nucleus sampling threshold; forwarded as-is when provided |
| `stream` | bool | false | Accepted by upstream chat backend, but current Python proxy buffers the full response before returning it |
| `stop` | array | null | Stop sequences |

### Response

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "model": "Qwen3.6-27B-Q4_K_M.gguf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "2 + 2 = 4.",
        "reasoning_content": ""
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 8,
    "total_tokens": 33
  }
}
```

### Streaming

```bash
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Count from 1 to 3"}],
    "stream": true
  }'
```

Current proxy behavior: the upstream chat backend may support streaming, but `server.py` currently reads the full upstream body before responding. In practice, clients should treat responses from this endpoint as buffered unless the proxy is changed to forward chunks.

### Multimodal Note

Do not assume `/v1/chat/completions` supports image input on the `qwen` route. The currently configured `qwen` chat backend is text-only in this proxy configuration. If you send multimodal chat payloads here, behavior depends on the upstream backend and should not be treated as supported by this API surface.

### Cold Start

After an idle unload, the next chat request cold-starts the selected backend before serving traffic.

---

## Audio Transcription

OpenAI-compatible audio transcription API.

```
POST /v1/audio/transcriptions
```

Routing is endpoint-based here: requests to `/v1/audio/transcriptions` go to the Whisper backend.

### Request

```bash
curl -X POST http://localhost:8088/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "model=whisper" \
  -F "response_format=json"
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file (`wav`, `mp3`, `m4a`, `ogg`, `flac`) |
| `model` | string | required | Use `whisper` for OpenAI-compatible clients |
| `response_format` | string | none at proxy layer | Forwarded to the Whisper backend as provided. Common backend values include `json`, `text`, `verbose_json`, `vtt`, and `srt`. |
| `language` | string | none at proxy layer | Optional language hint forwarded to the Whisper backend, for example `en`. Validation and behavior are backend-defined. |
| `temperature` | float | none at proxy layer | Optional decoding temperature forwarded as-is to the Whisper backend. |

### Response (json)

```json
{
  "text": "Hello, this is a test recording."
}
```

### Response (verbose_json)

```json
{
  "text": "Hello, this is a test recording.",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, this is a test recording."
    }
  ],
  "language": "en"
}
```

### Cold Start

After an idle unload, the next transcription request cold-starts Whisper before serving traffic.

## Qwen Transcribe Compatibility Route

Compatibility route for the lightweight `qwen-transcribe` backend.

```
POST /v1/transcribe
```

Operator notes:
- This route is exposed directly by the proxy and rewritten upstream to the qwen-transcribe backend's `/v1/chat/completions` endpoint.
- It exists for routed compatibility with the registered `qwen-transcribe` model, not as an OpenAI Whisper replacement.
- Unlike `/v1/audio/transcriptions`, this route does not target Whisper and should not be documented or treated as the same API contract.
- `/v1/audio/transcriptions` remains the OpenAI-compatible transcription endpoint for Whisper-style clients.

Use this route only when you specifically need the qwen-transcribe backend behavior exposed by this proxy.

### Cold Start

After an idle unload, the next `/v1/transcribe` request cold-starts qwen-transcribe before serving traffic.

---

## Forecasting (TimesFM 2.5)

General-purpose zero-shot time-series forecasting backed by TimesFM 2.5 200M.

```
POST /v1/forecast
```

Request body shape matches the worker contract exactly:
- `time_series`: batch of one or more numeric series
- `horizon`: forecast length, capped server-side at `512`

### Request

```bash
curl -X POST http://localhost:8088/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "time_series": [[10, 20, 15, 25, 20, 30, 25, 35, 30, 40]],
    "horizon": 5
  }'
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_series` | array of arrays | required | Batch of 1+ numeric time series |
| `horizon` | int | 128 | Number of future steps to forecast; values above `512` are truncated server-side |

### Response

```json
{
  "point_forecast": [[35.2, 42.1, 37.8, 45.3, 40.1]],
  "quantile_forecast": [
    [
      [28.2, 30.5, 32.8, 34.1, 35.0, 35.4, 36.1, 38.5, 40.7, 42.3],
      [34.8, 36.2, 38.1, 40.0, 41.5, 42.7, 43.9, 45.8, 48.6, 52.3],
      [31.1, 33.0, 34.6, 36.0, 37.4, 38.2, 39.7, 41.5, 43.8, 46.0],
      [37.2, 39.1, 41.0, 42.8, 44.0, 45.1, 46.6, 48.9, 51.5, 54.2],
      [33.0, 35.4, 37.1, 38.8, 40.0, 41.2, 42.6, 44.7, 47.9, 50.8]
    ]
  ]
}
```

### Response Fields

| Field | Shape | Description |
|-------|-------|-------------|
| `point_forecast` | `(batch, horizon)` | Median prediction for each series |
| `quantile_forecast` | `(batch, horizon, 10)` | Quantile predictions per step |

### Input Guidelines

- Use at least a few dozen historical points when possible.
- Keep each series ordered oldest -> newest.
- Send batched series in one request when you want one inference pass over multiple inputs.

### Operator Note

Forecasting requests can be slower than chat and transcription after idle periods or on larger batches. For downstream systems that do not require inline blocking forecasts, prefer a cache-first plus background-refresh pattern.

### Cold Start

After an idle unload, the next forecast request cold-starts TimesFM before serving traffic.

---

## Integration Examples

### Python

```python
import requests

BASE = "http://localhost:8088"

# Chat
resp = requests.post(f"{BASE}/v1/chat/completions", json={
    "model": "qwen",
    "messages": [{"role": "user", "content": "Hello"}]
})
print(resp.json()["choices"][0]["message"]["content"])

# Forecast
resp = requests.post(f"{BASE}/v1/forecast", json={
    "time_series": [[1, 2, 3, 4, 5]],
    "horizon": 3,
})
forecast = resp.json()
median = forecast["point_forecast"][0]

# Transcribe
with open("audio.wav", "rb") as f:
    resp = requests.post(
        f"{BASE}/v1/audio/transcriptions",
        files={"file": f},
        data={"model": "whisper", "response_format": "json"},
    )
print(resp.json())
```

### JavaScript/TypeScript

```javascript
import OpenAI from "openai";

const BASE = "http://localhost:8088";
const client = new OpenAI({ baseURL: `${BASE}/v1`, apiKey: "unused" });

const chat = await client.chat.completions.create({
  model: "qwen",
  messages: [{ role: "user", content: "Hello" }],
});

const resp = await fetch(`${BASE}/v1/forecast`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    time_series: [[1, 2, 3, 4, 5]],
    horizon: 3,
  }),
});

const forecast = await resp.json();
```

### OpenAI SDK Compatibility

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8088/v1", api_key="unused")

resp = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Hello"}],
)
```

---

## GPU Lifecycle

All backends share one Tesla P40 (24 GB VRAM). Each backend loads independently on first request and unloads after the configured idle timeout.

| State | VRAM | Power | GPU Temp |
|-------|------|-------|----------|
| All idle | 3 MiB | 12W | ~35°C |
| TimesFM only | 1.1 GB | 55W | ~45°C |
| Qwen only | 2.2 GB | 55W | ~48°C |
| Multiple backends loaded | ~3.5 GB | 60W | ~55°C |

- **Application default idle timeout**: 300 seconds if `IDLE_TIMEOUT` is unset
- **Current compose setting**: 1200 seconds (20 minutes)
- **Cold start behavior**: first request after unload starts the selected backend on demand
- The GPU returns to low-power idle when no backends are loaded

### Check What's Loaded

```bash
curl http://localhost:8088/health
```

---

## Error Responses

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Guaranteed for TimesFM worker request parsing/validation failures |
| 404 | Proxy-owned unknown endpoint response |
| 502 | Proxy-to-backend forwarding/connectivity failure |
| 503 | Proxy-owned backend startup failure |

Error behavior depends on which layer rejects the request:

- Proxy-owned `404` responses come from the Python HTTP server for unknown routes
- Proxy-owned `503` responses are returned when a managed backend fails to start
- TimesFM guarantees JSON `400` responses such as `{"error": "..."}` when its worker cannot parse or validate the request
- Backend HTTP error statuses are generally forwarded as-is, while local proxy forwarding/connectivity failures return `502`
- Many other response bodies are backend-dependent and may be JSON, plain text, or HTML

Do not assume every non-200 response is JSON unless you normalize errors in front of this server.

---

## Docker

```bash
# Start
docker compose up -d

# Stop
docker compose down

# Recreate after env changes or rebuild-relevant changes
docker compose down -v && docker compose up -d --force-recreate --build

# View logs
docker logs llm-inference-server

# GPU status
nvidia-smi
```