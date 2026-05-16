# Phase 0b — Working Recipe (locked from PASS run, 2026-05-16)

After 14 iterations on Vast.ai (~$5 burned), this is the recipe that produced
`phase0b_report_PASS.json`. Apply these exact changes to the Phase 4 wrapper.

## Vast.ai offer filters

```
gpu_name=RTX_4090 num_gpus=1
gpu_ram>=24 reliability>0.99
verified=true rentable=true direct_port_count>=1
cuda_vers>=12.4
inet_down>=400 disk_bw>=500
geolocation in [US,CA,AU,GB,DE,NL,FR,SE,IE,NO,FI,SG,JP,IT,ES,CH,AT,BE]
```

**Why each filter:**
- `cuda_vers>=12.4` — torch 2.6+cu124 fails Error 804 on older host drivers.
- `inet_down>=400 disk_bw>=500` — first image pull is faster on well-provisioned hosts.
- `geolocation in [...]` — CN/east-Asia hosts hit pypi.org/files.pythonhosted.org timeouts.
- `verified=true` + `reliability>0.99` — stable hosts only.

## Base image

```python
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
```

NOT `unsloth/unsloth:latest`:
- That image is 13.7 GB (vs 6.9 GB pytorch:devel) → pull is 2× slower
- Critically, `unsloth/unsloth:latest` doesn't run sshd → Vast SSH proxy can't connect
- Vast auto-injects sshd into `pytorch/*` images (it tags them `pytorch_*/ssh`)

## SSH endpoint resolution: DIRECT, not proxy

Vast's `ssh_url(id)` returns the **proxy** URL (`ssh<N>.vast.ai:<port>`). The proxy depends on a reverse SSH tunnel from the container to Vast's edge — this tunnel often **fails to bind** on busy hosts, leaving the proxy endpoint dead even though the container is fully up.

**Always use the direct endpoint** from `show_instance()`:

```python
info = vast.show_instance(id=instance_id)
ssh_host = info["public_ipaddr"]              # e.g. 72.19.32.135
ssh_port = int(info["ports"]["22/tcp"][0]["HostPort"])   # e.g. 41108
```

Symptom of broken proxy: container log spams `Error: remote port forwarding failed for listen port <port>`. Auth keys are correct; the tunnel itself fails.

## SSH key setup

Vast auto-injects all keys registered against the account via env var `SSH_PUBLIC_KEY` (concatenated). To register a new key:

```
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"
```

The key applies to all FUTURE instances. Already-running instances need `vastai attach ssh <instance_id> "<pubkey_string>"`.

## Signal handler discipline

The orchestrator's signal handler **MUST NOT** auto-destroy the instance. SIGTERM should preserve the instance for inspection; only normal exit-with-PASS-report triggers destroy. This avoids burning $$$ when the orchestrator is killed mid-run for debugging.

```python
def sig(sig_num, _frame):
    print(f"[SIGNAL {sig_num}] orchestrator aborting — INSTANCE {instance_id} PRESERVED.")
    sys.exit(130)   # NO destroy() call
```

Finally block: destroy only if PASS report exists locally.

## Wait caps

- `status=running`: **30 min** (large images can take 15+ min to pull on average hosts)
- SSH ready: **10 min** (sshd inside container can take 3-5 min post-status=running)

## Dependency stack — THE KEY FINDING

After 8+ iterations, the working installation order:

```bash
# 1. Verify image torch (2.6.0+cu124 baked in)
python -c "import torch; assert torch.cuda.is_available()"

# 2. Install unsloth via official cu124-torch260 extra (lets Unsloth resolve)
pip install "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git"

# 3. CRITICAL DOWNGRADE: unsloth's own extra installs torchao 0.17 which needs
#    torch.utils._pytree.register_constant (only added in torch 2.7).
#    torch 2.7+cu124 wheel doesn't exist (cu124 caps at 2.6.0).
#    Workaround: downgrade torchao to <0.13 — unsloth-zoo complains about
#    "torchao>=0.13" constraint but ACTUALLY WORKS with 0.12.
pip install "torchao<0.13"
```

That last `pip install "torchao<0.13"` is the *load-bearing fix*. Without it:
```
AttributeError: module 'torch.utils._pytree' has no attribute 'register_constant'
```
With it, Unsloth imports cleanly + runs SFT + merge + GGUF round-trip.

The pip resolver WILL emit warnings about the dep conflict — those are non-fatal.

## phase0b_smoke.py fix

Original assertion `assert "<think>" not in r` was too strict — Qwen3.5's chat template renders `<think></think>` (empty pair) when `enable_thinking=False`. Relaxed to:

```python
m = re.search(r"<think>(.*?)</think>", r, re.DOTALL)
if m and m.group(1).strip():
    raise AssertionError(f"non-empty <think> content leaked: {m.group(1)[:80]!r}")
```

Only fails on NON-EMPTY think content (actual thinking leakage), not the empty marker pair.

## llama.cpp build deps

llama.cpp's `requirements/requirements-convert_hf_to_gguf.txt` (and other req files) list `torch` — which pip would re-install, downgrading our CUDA torch to a CPU wheel. Filter:

```bash
for f in /workspace/llama.cpp/requirements/*.txt; do
    sed -i '/^torch/d; /^torchvision/d; /^torchaudio/d' "$f" 2>/dev/null || true
done
pip install -r /workspace/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
```

In-place sed (not redirect to `/tmp/`) because the req files contain relative `-r` cross-references.

## PASS metrics (the locked-in baseline)

| Step | Time | VRAM |
|---|---|---|
| Load Qwen3.5-9B 4-bit | 20.42s | (post-load) |
| 1 SFT step (rank 8 LoRA, bs=1, seq 4096) | 23.08s | 8320 MiB peak |
| Save PEFT adapter | 0.89s | — |
| Save merged fp16 (~18 GB) | 80.64s | — |
| Convert merged → fp16 GGUF | 35.99s | — |
| Quantize fp16 → Q4_K_M | 129.22s | — |
| **Final Q4_K_M GGUF** | **5,629,108,768 bytes = 5.24 GiB** | — |

End-to-end Phase 4 estimate (200-example train, 3 epochs, rank 8):
- Train: ~3-5 min (small dataset, small rank)
- Merge + quantize: ~4 min
- Plus model download (first time only): ~2.5 min

Total per training iteration: ~10-15 min wall, ~$0.05-0.10 cost.

## Phase 0b cost retrospective

~$5 across 14 iterations to debug. Per-iteration breakdown:
- Cheap hosts (China): $0.22/hr — but PyPI timeouts forced abandonment
- US/CA hosts: $0.27-0.40/hr — reliable network
- Sweden hosts: $0.33/hr — fastest network, sometimes flaky availability

**Going into Phase 4**, baseline cost per training run = ~$0.10. The infrastructure debugging is amortized.

## Next: update `vast_run_phase0b.py` (now `vast_run.py` for Phase 4)

Promote this recipe to the production wrapper that Phase 4 uses. Same script with parameters for dataset size / rank / epochs.
