"""Vast.ai orchestrator for Phase 4 — QLoRA over the MTP-preserved
Qwen3.6-27B checkpoint on a single 4090. Same lifecycle as
vast_run_phase0f.py (search → create → wait running → scp inputs →
remote setup + train → scp outputs back → destroy).

Inputs (must exist locally before running):
  finetune/datasets/phase4.train.jsonl
  finetune/datasets/phase4.holdout.jsonl
  finetune/phase4_train.py

Outputs (pulled back to finetune/REVIEWS/phase4_<ts>/):
  train_metrics.json
  holdout_preds.jsonl
  adapter/                  (PEFT weights, ~200 MB)
  merged/                   (only when PHASE4_FETCH_MERGED=1; ~54 GB)

Run: VAST_API_KEY=... python3 finetune/vast_run_phase4.py
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

MAX_DOLLARS_PER_HOUR = 1.10
MIN_DOLLARS_PER_HOUR = 0.40  # skip dead $0.28 provider (IP 66.183.63.178, offer ~26349244-class — intended=stopped)
MAX_WALLCLOCK_MIN = 240
MIN_GPU_RAM_GB = 24
MIN_RELIABILITY = 0.95  # loosened from 0.99 — weekend 4090 supply thin
SSH_READY_TIMEOUT_S = 600
SETUP_TIMEOUT_S = 3600
TRAIN_TIMEOUT_S = 7200
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
DISK_GB = 200  # ~54 GB base + ~10 GB merged + caches

REPO_DIR = Path(__file__).parent
TRAIN_SCRIPT = REPO_DIR / "phase4_train.py"
DATASETS_DIR = REPO_DIR / "datasets"
TRAIN_JSONL = DATASETS_DIR / "phase4.train.jsonl"
HOLDOUT_JSONL = DATASETS_DIR / "phase4.holdout.jsonl"
LOCAL_REPORT_DIR = REPO_DIR / "REVIEWS"
LOCAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)


def fail(msg: str, code: int = 1):
    print(f"\n[FAIL] {msg}", flush=True)
    sys.exit(code)


def banner(msg: str):
    print(f"\n=== {msg} ===", flush=True)


REMOTE_SCRIPT = """
set -euo pipefail
cd /workspace

echo "[setup] image torch + cuda check"
python3 -c 'import torch; print("torch:", torch.__version__, "cuda_avail:", torch.cuda.is_available(), "GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'

echo "[setup] apt deps"
apt-get update -qq && apt-get install -y -qq git aria2 2>&1 | tail -3

echo "[setup] pip install unsloth (no composite extra; let pip resolve compatible torchao)"
pip install --quiet --upgrade pip
pip install --quiet "unsloth==2026.5.2" 2>&1 | tail -3
pip install --quiet hf_transfer safetensors 2>&1 | tail -3

echo "[setup] verify all imports up front (catches Unsloth/torch ABI breakage early)"
python3 -c 'import torch; assert torch.cuda.is_available(); print("torch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))'
python3 -c 'import unsloth, transformers, peft, trl, datasets, bitsandbytes; print("unsloth:", unsloth.__version__, "transformers:", transformers.__version__, "trl:", trl.__version__)' \
  > /workspace/phase4_imports.log 2>&1
IRC=$?
cat /workspace/phase4_imports.log
if [ $IRC -ne 0 ]; then
    echo "[setup] IMPORT SMOKE FAIL rc=$IRC — aborting before train"
    exit 11
fi

# hf_transfer C extension has no retry/timeout — single stalled shard hangs.
# Default huggingface_hub auto-retries; re-enable only after upstream fix.
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=60
export TOKENIZERS_PARALLELISM=false

echo "[train] phase4_train.py (log → /workspace/phase4_train.log; redirect not piped so file is created even on early crash)"
python3 -u /workspace/phase4_train.py --work-dir /workspace > /workspace/phase4_train.log 2>&1
TRAIN_RC=$?
echo "[train] exit=$TRAIN_RC"
echo "=== last 80 lines of phase4_train.log ==="
tail -80 /workspace/phase4_train.log 2>&1
ls -la /workspace/output/ 2>&1 | head -20

if [ $TRAIN_RC -eq 0 ] && [ -d /workspace/output/merged ]; then
    echo "[gguf] convert + quantize merged → IQ4_XS-Q8nextn"
    bash /workspace/convert_to_gguf.sh 2>&1 | tee -a /workspace/phase4_train.log
    ls -lh /workspace/output/gguf/ 2>&1 | head -10
else
    echo "[gguf] skip — train rc=$TRAIN_RC or merged dir missing"
fi
"""


def main():
    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        fail("VAST_API_KEY env var not set")
    for p in (TRAIN_SCRIPT, TRAIN_JSONL, HOLDOUT_JSONL):
        if not p.exists():
            fail(f"missing required file: {p}")

    from vastai import VastAI  # type: ignore
    vast = VastAI(api_key=api_key)

    banner(f"Search 4090 offers (≥{MIN_GPU_RAM_GB} GB VRAM, ≥{DISK_GB} GB disk, ≤${MAX_DOLLARS_PER_HOUR}/hr)")
    query = (
        f"gpu_name=RTX_4090 num_gpus=1 "
        f"gpu_ram>={MIN_GPU_RAM_GB} reliability>{MIN_RELIABILITY} "
        f"verified=true rentable=true direct_port_count>=1 "
        f"cuda_vers>=12.4 "
        f"disk_space>={DISK_GB} "
        f"inet_down>=600 disk_bw>=500 "
        f"geolocation in [US,CA,AU,GB,DE,NL,FR,SE,IE,NO,FI,SG,JP,IT,ES,CH,AT,BE]"
    )
    offers = vast.search_offers(query=query, order="dph_total", limit="20")
    if not offers:
        fail("no offers matching policy")

    chosen = None
    for off in offers:
        dph = float(off.get("dph_total") or off.get("dph") or 999)
        if MIN_DOLLARS_PER_HOUR <= dph <= MAX_DOLLARS_PER_HOUR:
            chosen = off; chosen["_dph"] = dph; break
    if chosen is None:
        cheapest = min(offers, key=lambda o: float(o.get("dph_total", 999)))
        fail(f"no offer in [${MIN_DOLLARS_PER_HOUR}, ${MAX_DOLLARS_PER_HOUR}]/hr; cheapest is ${cheapest.get('dph_total')}/hr")

    offer_id = chosen["id"]
    print(f"chosen offer={offer_id}  ${chosen['_dph']}/hr  "
          f"gpu={chosen.get('gpu_name')}  vram={chosen.get('gpu_ram')} GB  "
          f"disk={chosen.get('disk_space','?')} GB  "
          f"inet_down={chosen.get('inet_down','?')} Mbps  "
          f"reliability={chosen.get('reliability2')}  host={chosen.get('hostname','?')}")

    banner(f"Create instance offer_id={offer_id}")
    create_resp = vast.create_instance(
        id=int(offer_id), image=IMAGE, disk=DISK_GB,
        runtype="ssh_direc ssh_proxy",
        onstart_cmd="nvidia-smi && sleep infinity",
        label="phase4-qlora-mtp-preserved",
    )
    instance_id = create_resp.get("new_contract") or create_resp.get("contract") or create_resp.get("instance_id")
    if not instance_id:
        fail(f"create_instance returned no instance_id: {create_resp}")
    instance_id = int(instance_id)
    print(f"instance_id={instance_id}")

    destroyed = False
    def destroy():
        nonlocal destroyed
        if destroyed: return
        banner(f"Destroy instance {instance_id}")
        try:
            vast.destroy_instance(id=instance_id); destroyed = True
        except Exception as e:
            print(f"[WARN] destroy failed: {e}; retry once")
            time.sleep(5)
            try:
                vast.destroy_instance(id=instance_id); destroyed = True
            except Exception as e2:
                print(f"[ERROR] second destroy failed: {e2}. Manual: vastai destroy instance {instance_id}")

    def sig(sig_num, _frame):
        print(f"\n[SIGNAL {sig_num}] aborting — INSTANCE {instance_id} PRESERVED. Destroy manually.")
        sys.exit(130)
    signal.signal(signal.SIGINT, sig)
    signal.signal(signal.SIGTERM, sig)

    wall_start = time.time()
    try:
        banner("Wait for status=running (30 min cap)")
        for _ in range(120):
            info = vast.show_instance(id=instance_id)
            status = info.get("actual_status") or info.get("status") or ""
            print(f"  [t+{int(time.time()-wall_start)}s] status={status}", flush=True)
            if status == "running": break
            if status in {"exited", "unknown", "offline", "error"}:
                fail(f"instance in bad status: {status}")
            time.sleep(15)
        else:
            fail("instance did not reach running in 30 min")

        banner("Resolve direct SSH endpoint")
        info = vast.show_instance(id=instance_id)
        ssh_user = "root"
        ssh_host = info.get("public_ipaddr")
        ports_22 = info.get("ports", {}).get("22/tcp") or []
        if not ssh_host or not ports_22:
            fail(f"no public_ipaddr or no port 22 map: {info}")
        ssh_port = int(ports_22[0]["HostPort"])
        print(f"  direct SSH: {ssh_user}@{ssh_host}:{ssh_port}")

        ssh_opts = [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            "-p", str(ssh_port),
        ]
        scp_opts = [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(ssh_port),
        ]

        banner("Wait for SSH")
        ssh_ready = False
        for _ in range(SSH_READY_TIMEOUT_S // 10):
            res = subprocess.run(
                ["ssh", *ssh_opts, f"{ssh_user}@{ssh_host}", "echo ready"],
                capture_output=True, text=True, timeout=15,
            )
            if res.returncode == 0 and "ready" in res.stdout:
                ssh_ready = True
                print(f"  ssh ready at t+{int(time.time()-wall_start)}s")
                break
            time.sleep(10)
        if not ssh_ready:
            fail("ssh never became ready")

        banner("scp phase4_train.py + train.jsonl + holdout.jsonl + convert_to_gguf.sh + patches/")
        subprocess.run(["scp", *scp_opts, str(TRAIN_SCRIPT),
                        f"{ssh_user}@{ssh_host}:/workspace/phase4_train.py"], check=True)
        subprocess.run(["scp", *scp_opts, str(REPO_DIR / "convert_to_gguf.sh"),
                        f"{ssh_user}@{ssh_host}:/workspace/convert_to_gguf.sh"], check=True)
        # MTP-aware convert_hf_to_gguf.py lives behind the TurboQuant+MTP base
        # patch; convert_to_gguf.sh applies it before running the converter.
        # mkdir parent first because scp -r doesn't auto-create the parent dir.
        subprocess.run(["ssh", *ssh_opts, f"{ssh_user}@{ssh_host}",
                        "mkdir -p /workspace/patches"], check=True)
        subprocess.run(["scp", *scp_opts, "-r", str(REPO_DIR.parent / "patches" / "llama-cpp"),
                        f"{ssh_user}@{ssh_host}:/workspace/patches/llama-cpp"], check=True)
        subprocess.run(["scp", *scp_opts, str(TRAIN_JSONL),
                        f"{ssh_user}@{ssh_host}:/workspace/train.jsonl"], check=True)
        subprocess.run(["scp", *scp_opts, str(HOLDOUT_JSONL),
                        f"{ssh_user}@{ssh_host}:/workspace/holdout.jsonl"], check=True)

        banner("Remote setup + train (slow: 54 GB base safetensors download)")
        remote_cmd_file = REPO_DIR / "_remote_runner_phase4.sh"
        remote_cmd_file.write_text(REMOTE_SCRIPT)
        subprocess.run(["scp", *scp_opts, str(remote_cmd_file),
                        f"{ssh_user}@{ssh_host}:/workspace/_remote_runner.sh"], check=True)
        remote_cmd_file.unlink()

        proc = subprocess.run(
            ["ssh", *ssh_opts, f"{ssh_user}@{ssh_host}", "bash /workspace/_remote_runner.sh"],
            timeout=SETUP_TIMEOUT_S + TRAIN_TIMEOUT_S,
        )
        if proc.returncode != 0:
            print(f"\n[WARN] remote train exited code {proc.returncode} — fetching whatever exists")

        banner("Fetch outputs")
        ts = int(time.time())
        out_local = LOCAL_REPORT_DIR / f"phase4_{ts}"
        out_local.mkdir(parents=True, exist_ok=True)
        # metrics + preds + log (always)
        for src in ("/workspace/output/train_metrics.json",
                    "/workspace/output/holdout_preds.jsonl",
                    "/workspace/phase4_train.log"):
            subprocess.run(["scp", *scp_opts,
                            f"{ssh_user}@{ssh_host}:{src}", str(out_local / Path(src).name)])
        # adapter (always — small)
        subprocess.run(["scp", *scp_opts, "-r",
                        f"{ssh_user}@{ssh_host}:/workspace/output/adapter",
                        str(out_local / "adapter")])
        # Quantized GGUF (always pull when present — drop-in for ./models/)
        gguf_remote_glob = "/workspace/output/gguf/qwen-trader-IQ4_XS-Q8nextn.gguf"
        models_dir = REPO_DIR.parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        gguf_local = models_dir / f"Qwen3.6-27B-Trader-IQ4_XS_{ts}.gguf"
        banner(f"Fetch quantized GGUF → {gguf_local.name}")
        scp_rc = subprocess.run(
            ["scp", *scp_opts, f"{ssh_user}@{ssh_host}:{gguf_remote_glob}", str(gguf_local)],
        ).returncode
        if scp_rc == 0 and gguf_local.exists():
            sz_gb = gguf_local.stat().st_size / 1e9
            print(f"  GGUF transferred ({sz_gb:.2f} GB)")
        else:
            print(f"  [WARN] GGUF scp rc={scp_rc} — convert may have failed; check log")

        # merged (opt-in only — large)
        if os.environ.get("PHASE4_FETCH_MERGED") == "1":
            banner("Fetch merged bf16 checkpoint (~54 GB)")
            subprocess.run(["scp", *scp_opts, "-r",
                            f"{ssh_user}@{ssh_host}:/workspace/output/merged",
                            str(out_local / "merged")])

        metrics_path = out_local / "train_metrics.json"
        if metrics_path.exists():
            banner("train_metrics.json")
            print(metrics_path.read_text())
            print(f"\n  saved → {out_local}")
        else:
            print(f"\n[WARN] no train_metrics.json under {out_local}; check the log")

    finally:
        elapsed_min = (time.time() - wall_start) / 60
        print(f"\n[total wall: {elapsed_min:.1f} min]")
        if os.environ.get("PRESERVE_ON_FAIL", "1") == "1" and not destroyed:
            # Always destroy after Phase 4 (no PASS gate; cost discipline).
            destroy()
        else:
            destroy()


if __name__ == "__main__":
    main()
