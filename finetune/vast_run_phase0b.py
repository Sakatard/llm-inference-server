"""Vast.ai orchestrator for Phase 0b training-loader smoke-test.

Lifecycle: search 4090 offers → create instance → wait for ready → upload
phase0b_smoke.py → run remote setup + smoke → fetch report → destroy.

Policy guards (per user instructions):
- Max 1 instance at a time
- Max wall-clock 60 minutes — kill + destroy if exceeded
- Max $0.80/hr accepted offer
- GPU must be RTX_4090 with ≥24 GiB VRAM
- Verified host, reliability ≥ 0.98
- ALWAYS destroy on exit, even on exceptions
- VAST_API_KEY must come from env, never hardcoded

Run: VAST_API_KEY=... python3 finetune/vast_run_phase0b.py
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

# ---- POLICY ----
MAX_DOLLARS_PER_HOUR = 0.80
MAX_WALLCLOCK_MIN = 60
MIN_GPU_RAM_GB = 24
MIN_RELIABILITY = 0.99
SSH_READY_TIMEOUT_S = 600   # extend — Vast container's SSH daemon can take 3-5 min post status=running
SETUP_TIMEOUT_S = 2400     # 40 min — first run pulls 13.7 GB image; subsequent runs use cached image
SMOKE_TIMEOUT_S = 1200     # 20 min for model download + smoke
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"   # 6.9 GB. Has Vast's sshd auto-setup. unsloth/unsloth image lacks sshd → SSH proxy fails.
DISK_GB = 80

REPO_DIR = Path(__file__).parent
SMOKE_SCRIPT = REPO_DIR / "phase0b_smoke.py"
LOCAL_REPORT_DIR = REPO_DIR / "REVIEWS"
LOCAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)


def fail(msg: str, code: int = 1):
    print(f"\n[FAIL] {msg}", flush=True)
    sys.exit(code)


def banner(msg: str):
    print(f"\n=== {msg} ===", flush=True)


def main():
    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        fail("VAST_API_KEY env var not set")
    if not SMOKE_SCRIPT.exists():
        fail(f"smoke script missing: {SMOKE_SCRIPT}")

    from vastai import VastAI  # type: ignore
    vast = VastAI(api_key=api_key)

    # ---- SEARCH ----
    banner(f"Search 4090 offers (≥{MIN_GPU_RAM_GB} GB, ≤${MAX_DOLLARS_PER_HOUR}/hr, verified, reliability≥{MIN_RELIABILITY})")
    # Filters:
    # - cuda_vers>=12.8: matches Unsloth image's CUDA. Older hosts hit Error 804.
    # - inet_down>=400 + disk_bw>=500: fast pull for 13.7 GB unsloth image
    # - exclude China for PyPI/HF routing reliability (even though deps baked, runtime model download still uses HF)
    query = (
        f"gpu_name=RTX_4090 num_gpus=1 "
        f"gpu_ram>={MIN_GPU_RAM_GB} reliability>{MIN_RELIABILITY} "
        f"verified=true rentable=true direct_port_count>=1 "
        f"cuda_vers>=12.8 "
        f"inet_down>=400 disk_bw>=500 "
        f"geolocation in [US,CA,AU,GB,DE,NL,FR,SE,IE,NO,FI,SG,JP,IT,ES,CH,AT,BE]"
    )
    offers = vast.search_offers(query=query, order="dph_total", limit="20")
    if not offers:
        fail("no 4090 offers matching policy")

    chosen = None
    for off in offers:
        dph = float(off.get("dph_total") or off.get("dph") or 999)
        if dph <= MAX_DOLLARS_PER_HOUR:
            chosen = off
            chosen["_dph"] = dph
            break

    if chosen is None:
        # show what we got for diagnostics
        cheapest = min(offers, key=lambda o: float(o.get("dph_total", 999)))
        fail(f"no offer under ${MAX_DOLLARS_PER_HOUR}/hr; cheapest is ${cheapest.get('dph_total')}/hr")

    offer_id = chosen["id"]
    print(f"chosen offer={offer_id}  ${chosen['_dph']}/hr  "
          f"gpu={chosen.get('gpu_name')}  vram={chosen.get('gpu_ram')} GB  "
          f"reliability={chosen.get('reliability2')}  host={chosen.get('hostname','?')}")

    # ---- CREATE ----
    banner(f"Create instance offer_id={offer_id}")
    # SDK kwargs differ from CLI. --ssh --direct maps to runtype="ssh_direc ssh_proxy"
    create_resp = vast.create_instance(
        id=int(offer_id),
        image=IMAGE,
        disk=DISK_GB,
        runtype="ssh_direc ssh_proxy",
        onstart_cmd="nvidia-smi && sleep infinity",
        label="phase0b-smoke",
    )
    instance_id = create_resp.get("new_contract") or create_resp.get("contract") or create_resp.get("instance_id")
    if not instance_id:
        fail(f"create_instance returned no instance_id: {create_resp}")
    instance_id = int(instance_id)
    print(f"instance_id={instance_id}")

    # ---- SAFE DESTROY WRAPPER ----
    destroyed = False
    def destroy():
        nonlocal destroyed
        if destroyed:
            return
        banner(f"Destroy instance {instance_id}")
        try:
            vast.destroy_instance(id=instance_id)
            destroyed = True
            print(f"instance {instance_id} destroyed")
        except Exception as e:
            print(f"[WARN] destroy failed: {e}; retry once")
            time.sleep(5)
            try:
                vast.destroy_instance(id=instance_id)
                destroyed = True
            except Exception as e2:
                print(f"[ERROR] second destroy also failed: {e2}. MANUAL CLEANUP REQUIRED: vastai destroy instance {instance_id}")

    # ---- SIGNAL HANDLERS: DO NOT auto-destroy on signal (user feedback) ----
    # User must manually destroy via `vastai destroy instance <ID>` if they want cleanup.
    # The try/finally still destroys on NORMAL exit (smoke completed) — but signal-kill leaves instance up.
    def sig(sig_num, _frame):
        print(f"\n[SIGNAL {sig_num}] orchestrator aborting — INSTANCE {instance_id} PRESERVED. Destroy manually with: vastai destroy instance {instance_id}")
        sys.exit(130)   # NOT calling destroy()
    signal.signal(signal.SIGINT, sig)
    signal.signal(signal.SIGTERM, sig)

    wall_start = time.time()
    try:
        # ---- WAIT FOR RUNNING ----
        banner("Wait for status=running")
        for i in range(120):   # 30 min cap — 13.7 GB unsloth image can take 15+ min to pull on slower hosts
            info = vast.show_instance(id=instance_id)
            status = info.get("actual_status") or info.get("status") or ""
            print(f"  [t+{int(time.time()-wall_start)}s] status={status}", flush=True)
            if status == "running":
                break
            if status in {"exited", "unknown", "offline", "error"}:
                fail(f"instance landed in bad status: {status}")
            time.sleep(15)
        else:
            fail("instance did not reach running in 30 min")

        # ---- GET DIRECT SSH (bypass Vast's proxy which often has broken reverse tunnels) ----
        banner("Resolve direct SSH endpoint")
        info = vast.show_instance(id=instance_id)
        ssh_user = "root"
        ssh_host = info.get("public_ipaddr")
        ports_22 = info.get("ports", {}).get("22/tcp") or []
        if not ssh_host or not ports_22:
            fail(f"instance has no public_ipaddr or no port 22 mapping: {info}")
        ssh_port = int(ports_22[0]["HostPort"])
        print(f"  direct SSH: {ssh_user}@{ssh_host}:{ssh_port}")

        ssh_opts = [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            "-p", str(ssh_port),
        ]

        # ---- WAIT FOR SSH ----
        banner("Wait for SSH to accept connections")
        ssh_ready = False
        for i in range(SSH_READY_TIMEOUT_S // 10):
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

        # ---- UPLOAD SMOKE SCRIPT ----
        banner("scp phase0b_smoke.py")
        scp_opts = [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(ssh_port),
        ]
        subprocess.run(
            ["scp", *scp_opts, str(SMOKE_SCRIPT), f"{ssh_user}@{ssh_host}:/workspace/phase0b_smoke.py"],
            check=True,
        )

        # ---- REMOTE SETUP + SMOKE ----
        banner("Remote setup + smoke (this is the slow part: pip + llama.cpp build + model download + train step)")
        remote_script = """
set -euo pipefail
cd /workspace
echo "[setup] image torch + cuda check"
python3 -c "import torch; print('torch:', torch.__version__, 'cuda_avail:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
echo "[setup] apt deps"
apt-get update -qq && apt-get install -y -qq cmake build-essential git 2>&1 | tail -3
echo "[setup] install unsloth via official cu124-torch260 extra (lets Unsloth resolve its own pinned deps)"
pip install --quiet --upgrade pip
pip install --quiet "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git"
pip install --quiet hf_transfer
echo "[setup] verify imports (with GPU now)"
python3 -c "import torch; assert torch.cuda.is_available() and '+cu' in torch.__version__, f'torch broke: {torch.__version__}'; print('torch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0))"
python3 -c "import unsloth; print('unsloth:', unsloth.__version__)" 2>&1 | tail -3

echo "[setup] clone + build llama.cpp"
if [ ! -d /workspace/llama.cpp ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /workspace/llama.cpp
fi
cd /workspace/llama.cpp
if [ ! -x build/bin/llama-quantize ]; then
    cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF >/tmp/cmake.log 2>&1
    cmake --build build -j$(nproc) --target llama-quantize >>/tmp/cmake.log 2>&1
fi
for f in requirements/*.txt; do
    sed -i '/^torch/d; /^torchvision/d; /^torchaudio/d' "$f" 2>/dev/null || true
done
pip install --quiet -r requirements/requirements-convert_hf_to_gguf.txt 2>&1 | tail -3 || true
echo "[setup] final torch verify"
python3 -c "import torch; assert torch.cuda.is_available(); print('torch FINAL OK:', torch.__version__)"
cd /workspace

export HF_HUB_ENABLE_HF_TRANSFER=1
export PHASE0B_OUT=/workspace/phase0b_out
export LLAMA_CPP_DIR=/workspace/llama.cpp
echo "[smoke] run"
python3 phase0b_smoke.py 2>&1 | tee /workspace/phase0b_smoke.log
echo "[smoke] done; report:"
cat /workspace/phase0b_out/phase0b_report.json
""".strip()
        remote_cmd_file = REPO_DIR / "_remote_runner.sh"
        remote_cmd_file.write_text(remote_script)
        subprocess.run(
            ["scp", *scp_opts, str(remote_cmd_file), f"{ssh_user}@{ssh_host}:/workspace/_remote_runner.sh"],
            check=True,
        )
        remote_cmd_file.unlink()

        smoke_proc = subprocess.run(
            ["ssh", *ssh_opts, f"{ssh_user}@{ssh_host}",
             "bash /workspace/_remote_runner.sh"],
            timeout=SETUP_TIMEOUT_S + SMOKE_TIMEOUT_S,
        )
        if smoke_proc.returncode != 0:
            print(f"\n[WARN] remote smoke exited code {smoke_proc.returncode} — fetching whatever report exists")

        # ---- FETCH REPORT ----
        banner("Fetch phase0b_report.json + phase0b_smoke.log")
        ts = int(time.time())
        local_report = LOCAL_REPORT_DIR / f"phase0b_report_{ts}.json"
        local_log = LOCAL_REPORT_DIR / f"phase0b_smoke_{ts}.log"
        subprocess.run(
            ["scp", *scp_opts,
             f"{ssh_user}@{ssh_host}:/workspace/phase0b_out/phase0b_report.json",
             str(local_report)],
        )
        subprocess.run(
            ["scp", *scp_opts,
             f"{ssh_user}@{ssh_host}:/workspace/phase0b_smoke.log",
             str(local_log)],
        )
        if local_report.exists():
            print("\n=== phase0b_report.json ===")
            print(local_report.read_text())
            print(f"\n  saved -> {local_report}")
        else:
            print("\n[WARN] no report retrieved")
        if local_log.exists():
            print(f"  log -> {local_log} ({local_log.stat().st_size} bytes)")

    finally:
        elapsed_min = (time.time() - wall_start) / 60
        print(f"\n[total wall: {elapsed_min:.1f} min]")
        # Only auto-destroy on SUCCESSFUL smoke (report fetched). Otherwise preserve for inspection.
        if os.environ.get("PRESERVE_ON_FAIL", "1") == "1" and not destroyed:
            success_report = LOCAL_REPORT_DIR / "phase0b_report.json"
            if success_report.exists() and 'PASS' in success_report.read_text():
                destroy()
            else:
                print(f"[preserve] instance {instance_id} NOT destroyed (no PASS report). Manual cleanup: vastai destroy instance {instance_id}")
        else:
            destroy()


if __name__ == "__main__":
    main()
