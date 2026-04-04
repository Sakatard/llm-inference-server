# GPU Server API Reference

Single-container GPU server running on Tesla P40. All models load on first request and unload after 5 minutes idle.

**Base URL:** `http://<host>:8080`

---

## Health Check

```
GET /health
```

Returns which models are currently loaded on the GPU.

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "ok",
  "active": {
    "qwen": false,
    "whisper": false,
    "timesfm": false
  }
}
```

`active: true` means the model is loaded on GPU. `false` means idle (next request will cold-start it in ~5-15s).

---

## Chat Completions (Qwen 3.5)

OpenAI-compatible chat API. Proxied to llama-server internally.

```
POST /v1/chat/completions
```

### Request

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
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

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Any value accepted (single model served) |
| `messages` | array | required | Array of `{role, content}` objects |
| `max_tokens` | int | 256 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0.0 = deterministic) |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `stream` | bool | false | Stream response as SSE events |
| `stop` | array | null | Stop sequences |

### Response

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "model": "Qwen3.5-0.8B-Q5_K_M.gguf",
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
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

Returns Server-Sent Events (`text/event-stream`), each line:

```
data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}
```

Final event:

```
data: [DONE]
```

### Vision (Multimodal)

Qwen 3.5 supports image inputs via base64 or URL:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
      }
    ],
    "max_tokens": 200
  }'
```

### Cold Start

First request after idle takes ~8-12s (loading 0.8B model to GPU). Subsequent requests are instant.

---

## Audio Transcription (Whisper)

OpenAI-compatible audio transcription API. Proxied to whisper-server internally.

```
POST /v1/audio/transcriptions
```

### Request

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "model=whisper" \
  -F "response_format=json"
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file (wav, mp3, m4a, ogg, flac) |
| `model` | string | required | Any value accepted (single model served) |
| `response_format` | string | "json" | `json`, `text`, `verbose_json`, `vtt`, `srt` |
| `language` | string | "en" | Language code (model is English-only) |
| `temperature` | float | 0.0 | Sampling temperature |

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

First request after idle takes ~3-5s (loading whisper-base.en to GPU). Subsequent requests are instant.

---

## Time-Series Forecast (TimesFM 2.5)

Google's TimesFM 2.5 200M foundation model for zero-shot time-series forecasting.

```
POST /v1/forecast
```

### Request

```bash
curl -X POST http://localhost:8080/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "time_series": [[10, 20, 15, 25, 20, 30, 25, 35, 30, 40]],
    "horizon": 5
  }'
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_series` | array of arrays | required | Batch of 1+ time series (each a list of floats) |
| `horizon` | int | 128 | Number of future steps to forecast (1–512) |

### Response

```json
{
  "point_forecast": [[35.2, 42.1, 37.8, 45.3, 40.1]],
  "quantile_forecast": [
    [
      [34.1, 28.2, 30.5, 32.8, 35.2, 35.2, 36.1, 38.5, 42.3, 42.3],
      [41.0, 32.1, 36.4, 39.2, 41.0, 42.1, 43.5, 45.8, 49.1, 52.3],
      ...
    ]
  ]
}
```

### Response Fields

| Field | Shape | Description |
|-------|-------|-------------|
| `point_forecast` | `(batch, horizon)` | Median prediction for each series |
| `quantile_forecast` | `(batch, horizon, 10)` | Quantile predictions per step |

**Quantile indices** (each row of the 10-element quantile array):

| Index | Quantile | Meaning |
|-------|----------|---------|
| 0 | mean | Mean prediction |
| 1 | 10th | Very low estimate |
| 2 | 20th | Low estimate |
| 3 | 30th | Below average |
| 4 | 40th | Slightly below median |
| 5 | 50th | Median |
| 6 | 60th | Slightly above median |
| 7 | 70th | Above average |
| 8 | 80th | High estimate |
| 9 | 90th | Very high estimate |

### Batch Forecasting

Send multiple series in one request:

```bash
curl -X POST http://localhost:8080/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "time_series": [
      [100, 120, 115, 130, 125, 140],
      [50, 48, 52, 47, 53, 46],
      [1, 2, 4, 8, 16, 32, 64]
    ],
    "horizon": 12
  }'
```

Returns forecasts for all three series in one response.

### Input Guidelines

- **Minimum length**: 32 values (shorter inputs get padded, may reduce accuracy)
- **Recommended length**: 100–2000 values for best results
- **Maximum length**: 16,000 values (model context limit)
- **No frequency indicator needed**: TimesFM 2.5 infers periodicity automatically
- **Handles any scale**: Inputs are normalized internally
- **NaN handling**: Leading NaNs are stripped, internal NaNs are interpolated

### Cold Start

First request after idle takes ~12-15s (loading 200M model to GPU + JIT compilation). Subsequent requests are instant.

---

## Integration Examples

### Python

```python
import requests

BASE = "http://localhost:8080"

# Chat
resp = requests.post(f"{BASE}/v1/chat/completions", json={
    "model": "qwen",
    "messages": [{"role": "user", "content": "Summarize this data"}],
    "max_tokens": 200,
})
print(resp.json()["choices"][0]["message"]["content"])

# Forecast
resp = requests.post(f"{BASE}/v1/forecast", json={
    "time_series": [sales_data],
    "horizon": 30,
})
forecast = resp.json()
median = forecast["point_forecast"][0]
upper = [q[9] for q in forecast["quantile_forecast"][0]]  # 90th percentile
lower = [q[1] for q in forecast["quantile_forecast"][0]]  # 10th percentile

# Transcribe
with open("audio.wav", "rb") as f:
    resp = requests.post(f"{BASE}/v1/audio/transcriptions",
        files={"file": f},
        data={"model": "whisper", "response_format": "json"},
    )
print(resp.json()["text"])
```

### JavaScript/TypeScript

```typescript
const BASE = "http://localhost:8080";

// Chat (OpenAI SDK compatible)
import OpenAI from "openai";
const client = new OpenAI({ baseURL: `${BASE}/v1`, apiKey: "unused" });

const chat = await client.chat.completions.create({
  model: "qwen",
  messages: [{ role: "user", content: "Hello" }],
});

// Forecast
const resp = await fetch(`${BASE}/v1/forecast`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    time_series: [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
    horizon: 5,
  }),
});
const { point_forecast, quantile_forecast } = await resp.json();
```

### OpenAI SDK Compatibility

The chat endpoint is fully compatible with the OpenAI Python/JS SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

---

## GPU Lifecycle

All models share one Tesla P40 (24 GB VRAM). Each model loads independently on first request and unloads after 5 minutes of inactivity.

| State | VRAM | Power | GPU Temp |
|-------|------|-------|----------|
| All idle | 3 MiB | 12W | ~35°C |
| TimesFM only | 1.1 GB | 55W | ~45°C |
| Qwen only | 2.2 GB | 55W | ~48°C |
| All three loaded | ~3.5 GB | 60W | ~55°C |

- **Idle timeout**: 5 minutes (configurable via `IDLE_TIMEOUT` env var)
- **Cold start**: 3-15s depending on model (first-ever start downloads TimesFM checkpoint ~800MB)
- The GPU fully powers down to P8 state when no models are loaded

### Check What's Loaded

```bash
curl http://localhost:8080/health
```

---

## Error Responses

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid JSON, missing fields) |
| 404 | Unknown endpoint |
| 502 | Backend error (model crashed) |
| 503 | Model failed to start (check container logs) |

Error body:

```json
{"error": "description of what went wrong"}
```

---

## Docker

```bash
# Start
docker compose up -d

# Stop
docker compose down

# Rebuild after code changes
docker compose down && docker compose build && docker compose up -d

# View logs
docker logs llm-inference-server

# GPU status
nvidia-smi
```
