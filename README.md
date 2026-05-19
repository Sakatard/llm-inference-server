# LLM Inference Server

A single-container, idle-aware, OpenAI-compatible inference router for a
Tesla P40. Routes between **Qwen 3.6 27B** (MTP self-speculative decoding,
TurboQuant turbo4 KV cache), **Qwen 3.5 0.8B** (multimodal transcription),
**Whisper large-v3-turbo** (audio → text), and **TimesFM 2.5**
(numeric time-series forecasting).

One Python process owns lifecycle for all of them. Each model spins up only
when the relevant route is hit, and spins down again after a configurable
idle window so the GPU drops to P8 between bursts.

> **Models are interchangeable.** The Qwen / Whisper / TimesFM defaults
> below are this author's deployment — swap in any GGUF for the chat
> routes, any whisper.cpp-compatible `.bin` for the audio route, and edit
> `server.py` / the env knobs accordingly. The architecture (single
> Python router + on-demand subprocess models + idle reaper) is the
> point; the specific weights are not.

---

## What is this?

Production HTTP API on `:8088` routing OpenAI-shaped requests to the right
local model. Idle-aware, no API keys, no per-request fees.

| Model | Route | Notes |
|-------|-------|-------|
| Qwen 3.6 27B | `/v1/chat/completions` (default) | MTP self-speculative + turbo4 KV; ~23.7 tok/s decode on P40 |
| Qwen 3.5 0.8B + mmproj | `/v1/transcribe` | Multimodal vision/audio transcription |
| Whisper large-v3-turbo | `/v1/audio/transcriptions` | C++ inference via whisper.cpp |
| TimesFM 2.5 | `/v1/forecast` | PyTorch, cold-starts slower than warm path |

### Requirements

- Docker + Docker Compose with NVIDIA Container Toolkit
- NVIDIA GPU with ≥20 GB VRAM (designed for Tesla P40, sm_61, 24 GB)
- ~30 GB disk for the model files listed below
- ~50 GB build cache space (llama.cpp + whisper.cpp compile from source)

---

## Getting the Models

Model files are **not included**. Drop them into `./models/` alongside
`docker-compose.yaml`:

```bash
mkdir -p models
```

Required files:

| File | Used by |
|------|---------|
| `Qwen3.6-27B-IQ4_XS.gguf` *(MTP build, default)* OR `Qwen3.6-27B-Q4_K_M.gguf` *(non-MTP)* | qwen route |
| `Qwen3.5-0.8B-Q5_K_M.gguf` | qwen-transcribe route |
| `mmproj-F32.gguf` | vision projector for qwen-transcribe |
| `ggml-large-v3-turbo-q8_0.bin` | whisper route |

Default model selection follows `ENABLE_MTP`:

- `ENABLE_MTP=1` (recommended) → loads `Qwen3.6-27B-IQ4_XS.gguf` with
  `--spec-type draft-mtp` for self-speculative decoding (~10% throughput win)
- `ENABLE_MTP=0` (legacy) → loads `Qwen3.6-27B-Q4_K_M.gguf` with plain
  autoregressive decode

TimesFM is auto-downloaded on first use to `./models/huggingface/` (via the
container's `HF_HOME=/models/huggingface` setting + the `./models:/models`
bind mount).

```text
models/
├── Qwen3.6-27B-IQ4_XS.gguf       # MTP build (default path)
├── Qwen3.6-27B-Q4_K_M.gguf       # non-MTP fallback (optional)
├── Qwen3.5-0.8B-Q5_K_M.gguf
├── mmproj-F32.gguf
├── ggml-large-v3-turbo-q8_0.bin
└── huggingface/                  # auto-populated for TimesFM
```

---

## Quick Start

```bash
docker compose build           # first build ~15-25 min (compiles llama.cpp + applies patch series)
docker compose up -d
curl http://localhost:8088/health
```

To advance the llama.cpp tree, bump `LLAMA_UPSTREAM_SHA` (and re-apply or
regenerate patches if conflicts surface) in `Dockerfile` / `Dockerfile.combined`.

---

## Build Architecture

The Dockerfile pulls upstream `ggml-org/llama.cpp` at a pinned SHA, then
applies the patch series in `patches/llama-cpp/` to layer in the features
that aren't (yet) in mainline:

| Patch | Adds |
|-------|------|
| `0001-turboquant-base.patch` | TurboQuant turbo2/turbo3/turbo4 KV cache types + TQ4_1S weight quant + fattn-vec turbo template instances + Q pre-rotation in `llama-graph.cpp` (~26k LoC) |
| `0002` | Pre-rotated turbo type registration in the dflash qwen35 target graph |
| `0003` | Tree-op ggml extensions (`ggml_ssm_conv_tree`, `ggml_gated_delta_net_tree`) + lucebox integration shim |
| `0004` | `LLAMA_DFLASH=ON` build option + `--decode-engine` CLI flag |
| `0005, 0006, 0008, 0009, 0010` | `LlamaToDFlashTarget` bridge, server CMake hook, `llama_model_embed_input_tokens` public API, Pascal sm_61 CUDA fixes |

`0007` was a lucebox-LFS-symlink fixup; dropped because the fresh vendor
clone at the pinned SHA already has the symlink. MTP self-speculative
decoding now ships natively in upstream llama.cpp via [PR #22673](
https://github.com/ggml-org/llama.cpp/pull/22673) + fixes #23198/#23237,
so the prior MTP hunks in `0001` were dropped on the rebase.

Between `0001` and `0002`, the build clones
[`Luce-Org/lucebox-hub`](https://github.com/Luce-Org/lucebox-hub) at
`6fe0d9a0` into `vendor/lucebox-hub/`. Lucebox supplies the dFlash CUDA
kernels (FWHT + ternary draft, DDTree verify); patches 0002–0010 glue them
into llama.cpp without modifying lucebox itself.

There is **no separate llama.cpp fork** — all deltas live in this repo as
patches. Bumping upstream means rebasing `patches/llama-cpp/0001*` against
the new mainline.

---

## Configuration

`docker-compose.yaml` passes one knob today:

- `IDLE_TIMEOUT=1200` (20 min before idle models unload)

`server.py` exposes more via env. The most consequential ones:

| Env | Default | Effect |
|-----|---------|--------|
| `IDLE_TIMEOUT` | `300` (compose: `1200`) | Seconds of inactivity before a model unloads |
| `START_TIMEOUT` | `120` | Seconds to wait for a model's HTTP probe after spawn |
| `PORT` | `8080` | In-container port (change the `8088:8080` mapping for host) |
| `ENABLE_MTP` | `0` (compose: `1`) | When `1`, qwen route uses Qwen3.6 IQ4_XS + MTP draft (self-spec) |
| `MTP_MODEL_PATH` | `/models/Qwen3.6-27B-IQ4_XS.gguf` | MTP-preserving GGUF (base or fine-tune that retained `blk.N.nextn.*` tensors) |
| `MTP_DRAFT_N_MAX` | `2` | Max drafted tokens per step (1-16). P40 sweet spot is 2; ≥5 falls off cliff because Pascal lacks tensor cores so verify-batch cost exceeds draft savings. |
| `MTP_DRAFT_P_MIN` | `0.0` | Acceptance probability floor for drafts |
| `MTP_CACHE_TYPE` | `turbo4` | KV cache quant (`turbo4` / `q8_0` / `f16`) |
| `MTP_CTX` | `65536` (turbo4) / `32768` (other) | Context window — auto-shrinks for unquantised caches |
| `DECODE_ENGINE` | `dflash` | `legacy` or `dflash`. **Currently a no-op:** the dflash library links but its dispatch falls through to `llama_decode`. Real spec-decode wiring (`project_hidden_to_tokens` ggml graph + server-side dispatch) is still TBD. |

To change anything, either set under `environment:` in
`docker-compose.yaml` or pass at `docker compose run` time, then
`docker compose up -d --force-recreate`.

---

## Performance

Bench on Tesla P40 (24 GB), Qwen 3.6 27B-IQ4_XS, 200-token decoded reply
(3-sample average for the prod config; single-sample for earlier rows):

| Build | Decode tok/s | MTP accept | vs baseline |
|-------|--------------|------------|-------------|
| Non-MTP (Q4_K_M, turbo4) baseline | 18.48 | — | 1.00× |
| Old pin + MTP + dflash bridge (IQ4_XS, N=3, turbo4) | 20.03 | (claimed) | +8.4% |
| **New pin + native MTP (IQ4_XS, N=2, turbo4)** | **23.66** | **83.7%** | **+28.0%** |

The +28% jump comes almost entirely from upstream's native MTP
implementation in PR #22673 — our prior hand-rolled MTP hunks left
the spec-decode path under-tuned, and the upstream version draws much
higher acceptance on the same prompt distribution.

Draft-N × `MTP_DRAFT_P_MIN` sweep on P40 (3-sample average per cell, ~200-token
essay prompt; absolute tok/s lower than the headline row above because of a
larger prompt + different output distribution — apples-to-apples is within
the grid):

| N | `p_min=0.0` | `p_min=0.25` | `p_min=0.5` | `p_min=0.75` |
|---|-------------|--------------|-------------|--------------|
| **2** | 21.21 / 69.9% | 21.27 / 70.4% | **22.19 / 75.2%** ← winner | 22.04 / 74.6% |
| 3 | 19.99 / 63.7% | 20.72 / 66.8% | 20.08 / 64.0% | 19.67 / 62.1% |
| 4 | 20.06 / 55.8% | 20.79 / 58.7% | 21.06 / 59.6% | 20.82 / 58.8% |

Pascal lacks tensor cores; verify-batch cost dominates draft savings once
N ≥ 3. N=2 dominates across every `p_min`. `p_min=0.5` is the cleanest
choice at every N — early-rejecting weak drafts saves verify cost. The
production build now ships `MTP_DRAFT_N_MAX=2`, `MTP_DRAFT_P_MIN=0.5`.

Full dflash spec-decode dispatch + tree-mode CUDA kernels remain unwired
and are de-prioritised: the dflash bridge would need a real
`output_norm + lm_head` ggml graph (path B) plus a quantized-row dequant
on top of the current `llama_model_embed_input_tokens` API (which only
handles F32/F16/BF16, not the Q4_K/IQ4_XS used by production GGUFs). A
generic 0.6B drafter is also unlikely to beat MTP's in-model nextn
acceptance on the Qwen3.6 distribution. Going forward, optimisation focus
stays on MTP tuning + cache variants.

---

## How It Works

```text
  You → :8088 (host) → :8080 (container)
                           │
                       server.py
                  (pure-Python router, no GPU libs)
                  /     |       |        \
              qwen   transcribe whisper  timesfm
                │
        MTP draft-target self-spec
        (llama.cpp PR #22673 inside
         the patched fork build)
```

`server.py` imports no GPU libraries, so the GPU stays at P8 idle when no
model is loaded. Each model is spawned as a subprocess on its own port and
proxied. After `IDLE_TIMEOUT` seconds without a request, it's terminated
and VRAM is freed.

Chat routing on `/v1/chat/completions` is by request `model` field:

- `qwen` or omitted → Qwen 3.6 27B
- `qwen-transcribe` → multimodal Qwen 3.5 0.8B

---

## API Endpoints

All on `http://localhost:8088`.

| Method | Path | Backing | Notes |
|--------|------|---------|-------|
| `POST` | `/v1/chat/completions` | qwen / qwen-transcribe | OpenAI-shaped |
| `POST` | `/v1/transcribe` | qwen-transcribe | Multimodal (rewrites internally to chat-completions) |
| `POST` | `/v1/audio/transcriptions` | whisper | Whisper file upload |
| `POST` | `/v1/forecast` | timesfm | Numeric time-series JSON |
| `GET`  | `/v1/models` | — | Lists routed IDs |
| `GET`  | `/health` | — | Active model status |

Full schemas in [`API.md`](API.md).

### Quick examples

```bash
# Default chat (MTP self-spec when ENABLE_MTP=1)
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Explain quantum entanglement simply."}],"max_tokens":500}'

# Whisper transcription
curl -X POST http://localhost:8088/v1/audio/transcriptions \
  -F "file=@recording.wav" -F "model=whisper"

# TimesFM forecast
curl -X POST http://localhost:8088/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{"time_series":[[100,102,101,105,107,109]],"horizon":3}'
```

```python
# Python (OpenAI SDK)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8088/v1", api_key="dummy")
resp = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=500,
)
print(resp.choices[0].message.content)
```

> **`max_tokens` note:** the qwen route enables reasoning by default (silent
> CoT before the visible answer). Allow ≥ 300 tokens or the model may run
> out of budget mid-response.

---

## GPU Behavior

- `server.py` (Python, no CUDA libs) stays up continuously.
- Each model route loads on first use, gets its own UNIX subprocess + port.
- Each route unloads independently after `IDLE_TIMEOUT`.
- Provided compose timeout (20 min) is tuned for repeated coding sessions
  where the 17 GB qwen 27B model has a high reload cost. Drop it back to
  `300` if you'd rather optimise for VRAM availability between bursts.

---

## Hardware Target

| Component | Spec |
|-----------|------|
| GPU | Tesla P40 (24 GB, CUDA compute 6.1, no FA2, no tensor cores) |
| CPU | Xeon E5-2660 v2 (Ivy Bridge — no AVX2/FMA/BMI2) |
| CUDA Driver | 13.0+ (toolkit 12.8 in the image) |
| PyTorch | 2.4.1 (last release with Pascal sm_61 support) |

If you're on a newer GPU, bump `CMAKE_CUDA_ARCHITECTURES` in
`Dockerfile.combined`. For a CPU with AVX2/FMA, drop the `IVY_CFLAGS`
gating (or keep them — they cost negligible perf on newer cores).

---

## Project Layout

```text
.
├── server.py                       # router process (always-on)
├── timesfm_worker.py               # TimesFM subprocess entry
├── docker-compose.yaml             # production wiring (qwen IQ4_XS + MTP)
├── docker-compose.mtp.yaml         # MTP-only variant
├── Dockerfile                      # primary image
├── Dockerfile.combined             # qwen + whisper + timesfm in one build
├── Dockerfile.timesfm              # TimesFM-only image
├── patches/llama-cpp/              # 9-patch series (TurboQuant + dflash; MTP now upstream)
├── artifacts/                      # deployable binaries
├── examples/                       # ready-to-run client scripts
└── API.md                          # endpoint reference
```

---

## Credits

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — upstream C++ LLM inference
- **[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)** — turbo3/turbo4 KV cache quantisation (squashed into patch 0001)
- **[llama.cpp PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)** — MTP self-speculative decoding (now native upstream; previously squashed into our patch 0001)
- **[Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub)** — dFlash speculative decode + DDTree verify (vendored at SHA `6fe0d9a0`)
- **[whisper.cpp](https://github.com/ggerganov/whisper.cpp)** — Whisper inference
- **[TimesFM](https://github.com/google-research/timesfm)** — Google Research time-series foundation model
- **[Qwen](https://huggingface.co/Qwen)** — Alibaba's language and multimodal models
- **[OpenAI Whisper](https://github.com/openai/whisper)** — Whisper model weights
- **[PyTorch](https://pytorch.org/)** — ML framework (v2.4.1 for Pascal compatibility)

---

## Examples

See [`examples/`](examples/) for ready-to-run Python and shell scripts
hitting each route.
