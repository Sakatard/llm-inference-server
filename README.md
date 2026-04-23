# LLM Inference Server

Unified GPU inference server running a routed **Qwen 3.6 27B** chat/coding model, selectable **qwen-planner** chat routing, **Qwen 3.5 0.8B** for multimodal transcription and vision-audio input, **Whisper** for audio transcription, and **TimesFM 2.5** for general-purpose numeric time-series forecasting on a single Tesla P40.

One container. One port. Multiple model routes. Each model loads only when you need it and unloads itself after a period of inactivity, so the GPU drops back toward idle power when nothing is happening. In the included `docker-compose.yaml`, that timeout is currently set to 20 minutes (`IDLE_TIMEOUT=1200`), which overrides the `server.py` application default of 5 minutes (`300` seconds). The longer compose timeout helps avoid repeated unload/reload churn for bigger chat and coding workloads.

---

## What is this?

If you've landed here and aren't sure what this project does, here's the short version:

This runs several AI models on your machine in a single Docker container, accessible over a simple HTTP API. You send a request, the server wakes the right model, does the work, and sends the result back. No cloud. No API keys. No per-request fees.

**What the models do:**

| Model | What it's for |
|-------|--------------|
| Qwen 3.6 27B | Default general chat and coding assistant model |
| qwen-planner (Qwen 3 1.7B) | Lightweight planner model selectable via chat `model` |
| Qwen 3.5 0.8B | Lightweight multimodal transcription / vision-audio route |
| Whisper large-v3-turbo | Transcribes audio files to text |
| TimesFM 2.5 | General-purpose numeric time-series forecasting |

**What you need to run it:**

- Docker and Docker Compose installed
- An NVIDIA GPU with at least 20 GB of VRAM (tested on Tesla P40, 24 GB)
- The NVIDIA Container Toolkit (`nvidia-docker2`) so Docker can access your GPU
- The model files listed below

---

## Getting the Models

Model files are **not included** in this repo. Put them in the `models/` folder next to your `docker-compose.yaml`.

Create the folder if it doesn't exist:

```bash
mkdir -p models
```

Place these files in `models/`:

- `Qwen3.6-27B-Q4_K_M.gguf` — default chat and coding model
- `Qwen3-1.7B-Q4_K_M.gguf` — planner model used when chat requests set `model` to `qwen-planner`
- `Qwen3.5-0.8B-Q5_K_M.gguf` — multimodal transcription / vision-audio route
- `mmproj-F32.gguf` — vision projector for the multimodal Qwen route
- `ggml-large-v3-turbo-q8_0.bin` — Whisper model

You can download the Qwen GGUF files from Hugging Face's Qwen releases and the Whisper file from `ggerganov/whisper.cpp`.

**TimesFM** is downloaded automatically on first use into Hugging Face's cache at `/models/huggingface` inside the container. Because `docker-compose.yaml` mounts `./models:/models` and `Dockerfile.combined` sets `HF_HOME=/models/huggingface`, those downloaded files persist on the host under `./models/huggingface/`.

Your `models/` folder should look like this when ready:

```text
models/
├── Qwen3.6-27B-Q4_K_M.gguf
├── Qwen3-1.7B-Q4_K_M.gguf
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

# Confirm it's running (models show false until first request — that's normal)
curl http://localhost:8088/health
```

---

## Configuration

Right now, the included `docker-compose.yaml` does **not** load a `.env` file. It passes only one setting directly to the container:

- `IDLE_TIMEOUT=1200`

Important distinction:
- `server.py` application default: `IDLE_TIMEOUT=300` seconds (5 minutes) if no environment variable is provided
- Current `docker-compose.yaml` override: `IDLE_TIMEOUT=1200` seconds (20 minutes)

So with the provided compose file, models stay loaded for **20 minutes / 1200 seconds** after their last request before unloading. This keeps the large chat/coding model warm longer so repeated coding sessions do not keep paying cold-start cost.

If you want to change that timeout, edit `docker-compose.yaml` directly:

```yaml
environment:
  - IDLE_TIMEOUT=1200
```

For example, to keep models loaded for 10 minutes instead of 20, change it to:

```yaml
environment:
  - IDLE_TIMEOUT=600
```

Then recreate the container so Docker uses the updated setting:

```bash
docker compose up -d --force-recreate
```

`server.py` also supports `START_TIMEOUT`, but the current `docker-compose.yaml` does not pass it through. If you want to use it, add it under `environment:` in `docker-compose.yaml` yourself.

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
    environment:
      - IDLE_TIMEOUT=1200
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

---

## How It Works

```text
  You → :8088 (host) → :8080 (container)
                           │
                      server.py
                    (pure Python router)
                     /    |     |    \
                    /     |     |     \
                qwen   whisper timesfm qwen-transcribe
                  |
           qwen-planner selectable
           via chat model routing
```

`server.py` is the only process that stays running all the time. It listens for requests and launches the right model as a subprocess when needed. In code, the application default is `IDLE_TIMEOUT=300` seconds unless the environment overrides it. In the provided compose setup, Docker injects `IDLE_TIMEOUT=1200`, so a model is shut down after 20 minutes of inactivity and VRAM is freed.

Chat requests are routed by the `model` field:

- `"qwen"` or omitted `model` → default Qwen 3.6 27B chat/coding route
- `"qwen-planner"` → lightweight planner chat route

Because `server.py` itself imports no GPU libraries, the GPU sits at P8 state when all models are idle.

---

## API Endpoints

All endpoints are on `http://localhost:8088`.

| Method | Path | Model | Description |
|--------|------|-------|-------------|
| `POST` | `/v1/chat/completions` | `qwen` by default, `qwen-planner` selectable | OpenAI-compatible chat |
| `POST` | `/v1/transcribe` | Qwen 3.5 0.8B | Multimodal transcription |
| `POST` | `/v1/audio/transcriptions` | Whisper | Audio-to-text; request is routed to the local Whisper service |
| `POST` | `/v1/forecast` | TimesFM | Numeric time-series forecasting |
| `GET`  | `/v1/models` | — | Lists available routed model IDs (`whisper` is the advertised Whisper ID) |
| `GET`  | `/health` | — | Shows which model routes are currently loaded |

See [API.md](API.md) for the full request and response details.

### Quick examples

**Default chat (`qwen`):**
```bash
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Explain quantum entanglement simply."}],
    "max_tokens": 500
  }'
```

**Planner chat (`qwen-planner`):**
```bash
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-planner",
    "messages": [{"role": "user", "content": "Give me a short step-by-step plan to migrate an API client."}],
    "max_tokens": 500
  }'
```

**Transcribe audio:**
```bash
curl -X POST http://localhost:8088/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "model=whisper"
```

For this endpoint, the proxy routes requests to the local Whisper service. `/v1/models` advertises the local model ID as `whisper`, and that is the safest value to send. The submitted `model` form field is not used to pick between multiple Whisper backends here.

**Forecast a numeric time series:**
```bash
curl -X POST http://localhost:8088/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "time_series": [[100, 102, 101, 105, 107, 109]],
    "horizon": 3
  }'
```

`time_series` is a list of one or more numeric series. The example above sends one series, but you can batch multiple series in the same request by adding more inner arrays.

**List available model IDs:**
```bash
curl http://localhost:8088/v1/models
```

**Python (OpenAI SDK):**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8088/v1", api_key="dummy")

response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=500,
)
print(response.choices[0].message.content)
```

> **Note on `max_tokens`:** The default `qwen` chat route enables reasoning, so it may think silently before answering. `qwen-planner` is still a chat route, but this README should not assume it uses the same reasoning mode by default. For the default `qwen` route, set `max_tokens` to at least 300–500 or the model may run out of budget and return an incomplete response.

> **Operational note:** TimesFM cold starts can be noticeably slower than a warm request. If you're building a downstream app, a simple cache-first/background-refresh pattern can keep forecast paths responsive.

---

## GPU Behavior

This project is designed to keep a single GPU useful without leaving every model loaded all the time.

- The Python router stays up continuously.
- Each model route loads on first use.
- Each route unloads independently after the configured idle timeout.
- App default: 5 minutes (`300` seconds) in `server.py`.
- Current compose override: 20 minutes (`1200` seconds).
- The big Qwen 3.6 chat/coding route is the heaviest path, which is why the longer timeout is helpful for repeated development work.

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
| `IDLE_TIMEOUT` | `300` app default; `1200` in current compose file | Seconds of inactivity before a model is unloaded |
| `START_TIMEOUT` | `120` | Seconds to wait for a model to become healthy on startup |

---

## Credits

- **[llama.cpp (TurboQuant fork)](https://github.com/TheTom/llama-cpp-turboquant)** — C++ LLM inference with turbo4 KV cache quantisation
- **[whisper.cpp](https://github.com/ggerganov/whisper.cpp)** — C++ inference for Whisper audio transcription
- **[TimesFM](https://github.com/google-research/timesfm)** — Google Research time-series foundation model
- **[Qwen](https://huggingface.co/Qwen)** — Alibaba's language and multimodal models
- **[OpenAI Whisper](https://github.com/openai/whisper)** — Original Whisper model weights
- **[PyTorch](https://pytorch.org/)** — ML framework powering TimesFM (v2.4.1 for Pascal GPU)

---

## Examples

See the [`examples/`](examples/) directory for ready-to-run Python and shell scripts.
