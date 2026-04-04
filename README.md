# LLM Inference Server

Unified GPU inference server running **Qwen 3.5** (chat + vision), **Whisper** (audio transcription), and **TimesFM 2.5** (time-series forecasting) on a single Tesla P40.

One container, one port, three models. Each model loads on first request and unloads after idle timeout — GPU fully powers down when not in use.

## Quick Start

```bash
# Build (first time takes ~10 min for C++ compilation)
docker compose build

# Start
docker compose up -d

# Check health
curl http://localhost:8080/health
```

## Architecture

```
            :8080
              │
     ┌────────┴────────┐
     │   server.py      │  ← Pure Python, no GPU imports
     │   (router)       │     GPU stays at P8 when idle
     └──┬─────┬─────┬──┘
        │     │     │
   ┌────▼┐ ┌─▼───┐ ┌▼────────┐
   │qwen │ │whsp │ │timesfm  │  ← Each is a subprocess
   │9180 │ │9181 │ │9182     │     Killed after idle timeout
   └─────┘ └─────┘ └─────────┘     → CUDA context destroyed
                                    → GPU enters P8 (12W)
```

All models are managed as subprocesses. The main process imports zero GPU libraries, so the GPU fully powers down (P8 state, ~12W) when no models are loaded.

## Endpoints

| Endpoint | Model | Description |
|----------|-------|-------------|
| `POST /v1/chat/completions` | Qwen 3.5 0.8B | OpenAI-compatible chat (text + vision) |
| `POST /v1/audio/transcriptions` | Whisper base.en | OpenAI-compatible audio transcription |
| `POST /v1/forecast` | TimesFM 2.5 200M | Time-series forecasting with quantiles |
| `GET /health` | — | Shows which models are loaded |

See [API.md](API.md) for full documentation with request/response schemas.

## Usage

### Chat

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Audio Transcription

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "model=whisper"
```

### Time-Series Forecast

```bash
curl -X POST http://localhost:8080/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "time_series": [[10, 20, 15, 25, 20, 30, 25, 35]],
    "horizon": 5
  }'
```

### OpenAI SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## GPU Lifecycle

| State | VRAM | Power | P-State |
|-------|------|-------|---------|
| All idle | 3 MiB | 12W | P8 |
| TimesFM loaded | 1.1 GB | 55W | P0 |
| Qwen loaded | 2.2 GB | 55W | P0 |
| All three loaded | ~3.5 GB | 60W | P0 |
| After 5 min idle | → 3 MiB | → 12W | → P8 |

## Configuration

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `PORT` | 8080 | Server listen port |
| `IDLE_TIMEOUT` | 300 | Seconds before idle model unloads |
| `START_TIMEOUT` | 120 | Max seconds to wait for model startup |

Set in `docker-compose.yaml` under `environment`.

## Hardware

Optimized for:
- **GPU**: Tesla P40 (24 GB, compute 6.1)
- **CPU**: Xeon E5-2660 v2 (Ivy Bridge — no AVX2/FMA)
- **CUDA Driver**: 13.0+
- **PyTorch**: 2.4.1 (last version supporting sm_61)

## Models

Place model files in `./models/`:

```
models/
├── Qwen3.5-0.8B-Q5_K_M.gguf    # Qwen chat model
├── mmproj-F32.gguf               # Qwen vision projector
├── ggml-base.en-q5_1.bin         # Whisper English model
└── huggingface/                   # TimesFM (auto-downloaded on first use)
```

## Examples

See the [`examples/`](examples/) directory for integration examples in Python and shell.
