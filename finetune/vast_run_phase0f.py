"""Vast.ai orchestrator for Phase 0f — Qwen3.6-27B QLoRA with MTP head training.

Same lifecycle as vast_run_phase0e.py with these changes:
- f-string bug FIXED (use Python string template + .replace, NOT f-string for remote_script)
- Pushes BOTH phase0f_smoke.py AND qwen35_mtp_modeling.py to remote
- Includes --tensor-type nextn=q8_0 in quantize step inside the python script

Run: VAST_API_KEY=... python3 finetune/vast_run_phase0f.py
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

MAX_DOLLARS_PER_HOUR = 0.90
MAX_WALLCLOCK_MIN = 180
MIN_GPU_RAM_GB = 24
MIN_RELIABILITY = 0.99
SSH_READY_TIMEOUT_S = 600
SETUP_TIMEOUT_S = 3600
SMOKE_TIMEOUT_S = 3600
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
DISK_GB = 300
# Historical: Phase 0f was run against Sakatard/llama-cpp-turboquant fork @ c85252627.
# Fork removed; for re-run, transfer ../patches/llama-cpp/ to /workspace and apply
# 0001-turboquant-mtp-base.patch on top of upstream after clone+checkout.
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_SHA = "253ba110bcd372207ca7b0bb56f1ea10d60d53fd"

REPO_DIR = Path(__file__).parent
SMOKE_SCRIPT = REPO_DIR / "phase0f_smoke.py"
MTP_MODELING = REPO_DIR / "qwen35_mtp_modeling.py"
LOCAL_REPORT_DIR = REPO_DIR / "REVIEWS"
LOCAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)


def fail(msg: str, code: int = 1):
    print(f"\n[FAIL] {msg}", flush=True)
    sys.exit(code)


def banner(msg: str):
    print(f"\n=== {msg} ===", flush=True)


# NOTE: this is a plain triple-quoted string (NOT f-string). Variables injected via .replace()
# below to avoid Python interpreting `{torch.__version__}` etc inside python -c blocks.
REMOTE_SCRIPT_TEMPLATE = """
set -euo pipefail
LLAMA_CPP_REPO=__LLAMA_CPP_REPO__
LLAMA_CPP_SHA=__LLAMA_CPP_SHA__

cd /workspace
echo "[setup] image torch + cuda check"
python3 -c 'import torch; print("torch:", torch.__version__, "cuda_avail:", torch.cuda.is_available(), "GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
echo "[setup] apt deps"
apt-get update -qq && apt-get install -y -qq cmake build-essential git aria2 2>&1 | tail -3
echo "[setup] install unsloth via cu124-torch260 extra"
pip install --quiet --upgrade pip
pip install --quiet "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git"
pip install --quiet "torchao<0.13"
pip install --quiet hf_transfer safetensors
echo "[setup] verify imports"
python3 -c 'import torch; assert torch.cuda.is_available() and "+cu" in torch.__version__; print("torch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))'
python3 -c 'import unsloth; print("unsloth:", unsloth.__version__)' 2>&1 | tail -3

echo "[setup] clone + checkout pinned Sakatard llama.cpp fork at $LLAMA_CPP_SHA"
if [ ! -d /workspace/llama.cpp ]; then
    git clone "$LLAMA_CPP_REPO" /workspace/llama.cpp
fi
cd /workspace/llama.cpp
git fetch --depth 50 origin || git fetch origin
git checkout "$LLAMA_CPP_SHA"
echo "[setup] llama.cpp head: $(git rev-parse HEAD)"
if [ ! -x build/bin/llama-quantize ]; then
    cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF >/tmp/cmake.log 2>&1
    cmake --build build -j$(nproc) --target llama-quantize >>/tmp/cmake.log 2>&1
fi
for f in requirements/*.txt; do
    sed -i '/^torch/d; /^torchvision/d; /^torchaudio/d' "$f" 2>/dev/null || true
done
pip install --quiet -r requirements/requirements-convert_hf_to_gguf.txt 2>&1 | tail -3 || true
echo "[setup] final torch verify"
python3 -c 'import torch; assert torch.cuda.is_available(); print("torch FINAL OK:", torch.__version__)'
cd /workspace

export HF_HUB_ENABLE_HF_TRANSFER=1
export PHASE0F_OUT=/workspace/phase0f_out
export LLAMA_CPP_DIR=/workspace/llama.cpp
echo "[smoke] run (56 GB safetensors download will dominate first ~10-15 min)"
python3 phase0f_smoke.py 2>&1 | tee /workspace/phase0f_smoke.log
echo "[smoke] done; report:"
cat /workspace/phase0f_out/phase0f_report.json
"""


def main():
    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        fail("VAST_API_KEY env var not set")
    for p in (SMOKE_SCRIPT, MTP_MODELING):
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
        if dph <= MAX_DOLLARS_PER_HOUR:
            chosen = off; chosen["_dph"] = dph; break
    if chosen is None:
        cheapest = min(offers, key=lambda o: float(o.get("dph_total", 999)))
        fail(f"no offer under ${MAX_DOLLARS_PER_HOUR}/hr; cheapest is ${cheapest.get('dph_total')}/hr")

    offer_id = chosen["id"]
    print(f"chosen offer={offer_id}  ${chosen['_dph']}/hr  "
          f"gpu={chosen.get('gpu_name')}  vram={chosen.get('gpu_ram')} GB  "
          f"disk={chosen.get('disk_space','?')} GB  "
          f"inet_down={chosen.get('inet_down','?')} Mbps  "
          f"reliability={chosen.get('reliability2')}  host={chosen.get('hostname','?')}")

    banner(f"Create instance offer_id={offer_id}")
    create_resp = vast.create_instance(
        id=int(offer_id),
        image=IMAGE,
        disk=DISK_GB,
        runtype="ssh_direc ssh_proxy",
        onstart_cmd="nvidia-smi && sleep infinity",
        label="phase0f-mtp-train-smoke",
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
            print(f"instance {instance_id} destroyed")
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

        banner("scp phase0f_smoke.py + qwen35_mtp_modeling.py")
        subprocess.run(
            ["scp", *scp_opts, str(SMOKE_SCRIPT), f"{ssh_user}@{ssh_host}:/workspace/phase0f_smoke.py"],
            check=True,
        )
        subprocess.run(
            ["scp", *scp_opts, str(MTP_MODELING), f"{ssh_user}@{ssh_host}:/workspace/qwen35_mtp_modeling.py"],
            check=True,
        )

        banner("Remote setup + smoke (slow: 56 GB safetensors download dominates)")
        # Substitute template variables — plain string replace, no f-string interpretation
        remote_script = (
            REMOTE_SCRIPT_TEMPLATE
            .replace("__LLAMA_CPP_REPO__", LLAMA_CPP_REPO)
            .replace("__LLAMA_CPP_SHA__", LLAMA_CPP_SHA)
        )
        remote_cmd_file = REPO_DIR / "_remote_runner_0f.sh"
        remote_cmd_file.write_text(remote_script)
        subprocess.run(
            ["scp", *scp_opts, str(remote_cmd_file), f"{ssh_user}@{ssh_host}:/workspace/_remote_runner.sh"],
            check=True,
        )
        remote_cmd_file.unlink()

        smoke_proc = subprocess.run(
            ["ssh", *ssh_opts, f"{ssh_user}@{ssh_host}", "bash /workspace/_remote_runner.sh"],
            timeout=SETUP_TIMEOUT_S + SMOKE_TIMEOUT_S,
        )
        if smoke_proc.returncode != 0:
            print(f"\n[WARN] remote smoke exited code {smoke_proc.returncode} — fetching report")

        banner("Fetch phase0f_report.json + phase0f_smoke.log")
        ts = int(time.time())
        local_report = LOCAL_REPORT_DIR / f"phase0f_report_{ts}.json"
        local_log = LOCAL_REPORT_DIR / f"phase0f_smoke_{ts}.log"
        subprocess.run(["scp", *scp_opts, f"{ssh_user}@{ssh_host}:/workspace/phase0f_out/phase0f_report.json", str(local_report)])
        subprocess.run(["scp", *scp_opts, f"{ssh_user}@{ssh_host}:/workspace/phase0f_smoke.log", str(local_log)])

        # Fetch the final quantized GGUF (~14-17 GB). Skip if not PASS to save bandwidth.
        # PASS check: report has result == "PASS" OR all gate booleans True
        is_pass = False
        if local_report.exists():
            try:
                rep = json.loads(local_report.read_text())
                is_pass = rep.get("result") == "PASS" or all(rep.get("gates", {}).values())
            except Exception:
                pass
        if is_pass or os.environ.get("PHASE0F_ALWAYS_FETCH_GGUF") == "1":
            banner("Fetch final IQ4_XS-Q8nextn GGUF (~14-17 GB)")
            models_dir = REPO_DIR.parent / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            local_gguf = models_dir / f"Qwen3.6-27B-MTP-FT-IQ4_XS-Q8nextn_{ts}.gguf"
            # find the actual output filename (IQ4_XS_Q8nextn OR Q4_K_M_Q8nextn depending on PHASE0F_QUANT)
            find_cmd = subprocess.run(
                ["ssh", *ssh_opts, f"{ssh_user}@{ssh_host}", "ls /workspace/phase0f_out/model_*_Q8nextn.gguf 2>/dev/null | head -1"],
                capture_output=True, text=True, timeout=30,
            )
            remote_gguf = find_cmd.stdout.strip()
            if remote_gguf:
                print(f"  remote: {remote_gguf}")
                print(f"  local:  {local_gguf}")
                t_scp = time.time()
                subprocess.run(["scp", *scp_opts, f"{ssh_user}@{ssh_host}:{remote_gguf}", str(local_gguf)], check=False)
                if local_gguf.exists():
                    print(f"  GGUF transferred ({local_gguf.stat().st_size / 1e9:.2f} GB in {time.time()-t_scp:.1f}s)")
                else:
                    print("  [WARN] GGUF scp failed; check Vast disk + network")
            else:
                print("  [WARN] no model_*_Q8nextn.gguf found on remote; skipping GGUF fetch")
        else:
            print("\n[skip] not PASS; skipping GGUF fetch (saves 14-17 GB bandwidth)")
            print("       to force: PHASE0F_ALWAYS_FETCH_GGUF=1 python3 vast_run_phase0f.py")

        if local_report.exists():
            print("\n=== phase0f_report.json ===")
            print(local_report.read_text())
            print(f"\n  saved -> {local_report}")
        else:
            print("\n[WARN] no report retrieved")
        if local_log.exists():
            print(f"  log -> {local_log} ({local_log.stat().st_size} bytes)")

    finally:
        elapsed_min = (time.time() - wall_start) / 60
        print(f"\n[total wall: {elapsed_min:.1f} min]")
        if os.environ.get("PRESERVE_ON_FAIL", "1") == "1" and not destroyed:
            recent = sorted(LOCAL_REPORT_DIR.glob("phase0f_report_*.json"))
            if recent and '"PASS"' in recent[-1].read_text():
                destroy()
            else:
                print(f"[preserve] instance {instance_id} NOT destroyed (no PASS report). Manual: vastai destroy instance {instance_id}")
        else:
            destroy()


if __name__ == "__main__":
    main()
