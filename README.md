# LLM Inference Server

Unified GPU inference server running **Qwen 3.5** (chat + vision), **Whisper** (audio transcription), and **TimesFM 2.5** (time-series forecasting) on a single Tesla P40.

One container. One port. Four models. Each model loads only when you need it and unloads itself after sitting idle — so the GPU drops back to ~12W when nothing is happening.

---

## What is this?

If you've landed here and aren't sure what this project does, here's the short version:

This runs several AI models on your machine in a single Docker container, accessible over a simple HTTP API. You send a request, the right model wakes up, does its job, and you get an answer back. No cloud. No API keys. No per-request fees.

**What the models do:**

| Model | What it's for |
|-------|--------------|
| Qwen 3.5 9B | General chat — answering questions, writing, reasoning |
| Qwen 3.5 0.8B | Lightweight chat with vision/audio input support |
| Whisper large-v3-turbo | Transcribes audio files to text |
| TimesFM 2.5 | Forecasts future values in a numeric time series |

**What you need to run it:**

- Docker and Docker Compose installed
- An NVIDIA GPU with at least 20 GB of VRAM (tested on Tesla P40, 24 GB)
- The NVIDIA Container Toolkit (`nvidia-docker2`) so Docker can access your GPU
- The model files (see [Getting the Models](#getting-the-models) below)

---

## Getting the Models

Model files are **not included** in this repo — they're too large. You need to download them separately and place them in the `models/` folder next to your `docker-compose.yaml`.

Create the folder if it doesn't exist:

```bash
mkdir -p models
```

Then download each model:

**Qwen 3.5 9B (chat)**
```bash
# ~9.5 GB
wget -P models/ https://huggingface.co/Qwen/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B.Q8_0.gguf
```

**Qwen 3.5 0.8B + vision projector**
```bash
# ~0.5 GB + ~0.3 GB
wget -P models/ https://huggingface.co/Qwen/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q5_K_M.gguf
wget -P models/ https://huggingface.co/Qwen/Qwen3.5-0.8B-GGUF/resolve/main/mmproj-F32.gguf
```

**Whisper large-v3-turbo**
```bash
# ~1.6 GB
wget -P models/ https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin
```

**TimesFM** is downloaded automatically on first use (cached to `models/huggingface/`).

Your `models/` folder should look like this when ready:

```
models/
├── Qwen3.5-9B.Q8_0.gguf
├── Qwen3.5-0.8B-Q5_K_M.gguf
├── mmproj-F32.gguf
└── ggml-large-v3-turbo-q8_0.bin
```

---

## Quick Start

```bash
# Build (first time takes ~15–20 min — compiles llama.cpp and whisper.cpp from source)
docker compose build

# Start in the background
docker compose up -d

# Confirm it's running (all models will show false until first request — that's normal)
curl http://localhost:8088/health
```

---

## Configuration

Create a `.env` file next to your `docker-compose.yaml` to override defaults:

```env
# .env

# How long (seconds) a model stays loaded after its last request before unloading
IDLE_TIMEOUT=300

# How long (seconds) to wait for a model to finish starting before giving up
START_TIMEOUT=120
```

These are already set to sensible defaults — you only need the `.env` if you want to change them.

---

## Example Files

### docker-compose.yaml

```yaml
services:
  llm-inference-server:
    build:
      context: .
      dockerfile: Dockerfile.combined
    image: llm-inference-server
    container_name: llm-inference-server
    ports:
      - "8088:8080"      # Host port 8088 maps to container port 8080
    volumes:
      - ./models:/models  # Your models folder is mounted here
    env_file:
      - .env
    environment:
      - IDLE_TIMEOUT=300
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```

### Dockerfile.combined (summary)

The build has two stages:

1. **Builder** — compiles `llama-server` (with TurboQuant KV cache support) and `whisper-server` from source, optimised for your CPU (Ivy Bridge flags — no AVX2/FMA)
2. **Final image** — copies the compiled binaries in, installs PyTorch 2.4.1, installs TimesFM, copies `server.py`

You do not need to modify the Dockerfile unless you're changing hardware targets (CUDA architecture, CPU flags).

---

## How It Works

```
  You → :8088 (host) → :8080 (container)
                           │
                      server.py
                      (pure Python router)
                           │
           ┌───────────────┼───────────────┐
           │               │               │
     llama-server    whisper-server   timesfm_worker.py
      :9180 qwen       :9182             :9183
      :9181 0.8B
```

`server.py` is the only process that stays running all the time. It listens for requests and launches the right model as a subprocess when needed. When a model hasn't been used for `IDLE_TIMEOUT` seconds, it's shut down and VRAM is freed.

Because `server.py` itself imports no GPU libraries, the GPU sits at P8 state (~12W) when all models are idle.

---

## API Endpoints

All endpoints are on `http://localhost:8088`.

| Method | Path | Model | Description |
|--------|------|-------|-------------|
| `POST` | `/v1/chat/completions` | Qwen 9B | OpenAI-compatible chat |
| `POST` | `/v1/transcribe` | Qwen 0.8B | Multimodal transcription |
| `POST` | `/v1/audio/transcriptions` | Whisper | Audio-to-text |
| `POST` | `/v1/forecast` | TimesFM | Time-series forecasting |
| `GET`  | `/health` | — | Shows which models are currently loaded |

See [API.md](API.md) for full request/response documentation.

### Quick examples

**Chat:**
```bash
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Explain quantum entanglement simply."}],
    "max_tokens": 500
  }'
```

**Transcribe audio:**
```bash
curl -X POST http://localhost:8088/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "model=whisper-1"
```

**Forecast:**
```bash
curl -X POST http://localhost:8088/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "time_series": [[10, 20, 15, 25, 20, 30, 25, 35]],
    "horizon": 5
  }'
```

**Using the OpenAI Python SDK:**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8088/v1", api_key="unused")

response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=500,
)
print(response.choices[0].message.content)
```

> **Note on `max_tokens`:** The Qwen models use chain-of-thought reasoning by default, which means they think silently before answering. Always set `max_tokens` to at least 300–500 or the model may run out of budget mid-thought and return an empty response.

---

## VRAM Usage

With all four models loaded simultaneously (~18.9 GB on a 24 GB P40):

| State | VRAM used | Power | GPU state |
|-------|-----------|-------|-----------|
| All idle | ~200 MiB | 12W | P8 |
| Qwen 9B only | ~10.5 GB | 55W | P0 |
| Qwen 0.8B only | ~1.5 GB | 55W | P0 |
| Whisper only | ~2.5 GB | 55W | P0 |
| TimesFM only | ~6.5 GB | 55W | P0 |
| All four loaded | ~18.9 GB | 60W | P0 |
| After idle timeout | → ~200 MiB | → 12W | → P8 |

In practice models rarely all load at once — the idle reaper frees each one independently.

---

## Hardware Target

This project is optimised for:

- **GPU**: Tesla P40 (24 GB, CUDA compute 6.1)
- **CPU**: Xeon E5-2660 v2 (Ivy Bridge — no AVX2/FMA/BMI2)
- **CUDA Driver**: 13.0+
- **PyTorch**: 2.4.1 (last version with Pascal / sm_61 support)

If you're on a different GPU, update `CMAKE_CUDA_ARCHITECTURES` in `Dockerfile.combined`. For a different CPU, update the `IVY_CFLAGS` env variable accordingly.

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Internal container port (change the host-side mapping in docker-compose instead) |
| `IDLE_TIMEOUT` | `300` | Seconds of inactivity before a model is unloaded |
| `START_TIMEOUT` | `120` | Seconds to wait for a model to become healthy on startup |

---

## Credits

- **[llama.cpp (TurboQuant fork)](https://github.com/TheTom/llama-cpp-turboquant)** — C++ LLM inference with turbo4 KV cache quantisation
- **[whisper.cpp](https://github.com/ggerganov/whisper.cpp)** — C++ inference for Whisper audio transcription
- **[TimesFM](https://github.com/google-research/timesfm)** — Google Research time-series foundation model
- **[Qwen 3.5](https://huggingface.co/Qwen)** — Alibaba's language/vision models
- **[OpenAI Whisper](https://github.com/openai/whisper)** — Original Whisper model weights
- **[PyTorch](https://pytorch.org/)** — ML framework powering TimesFM (v2.4.1 for Pascal GPU)

---

## Examples

See the [`examples/`](examples/) directory for ready-to-run Python and shell scripts.
