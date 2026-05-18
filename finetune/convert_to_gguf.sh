#!/bin/bash
# Convert a merged bf16 HF checkpoint to GGUF + IQ4_XS quant, preserving MTP heads.
#
# Designed to run ON THE VAST INSTANCE (right after phase4_train.py finishes the
# bf16 merge). Outputs land in /workspace/output/gguf/ for SCP back.
#
# Why on Vast: the merge writes a ~54 GB bf16 dir that's faster to convert there
# than to pull cross-continent and convert on the local host.
#
# Pre: /workspace/output/merged/ exists with model.safetensors + config.json
# Post: /workspace/output/gguf/qwen-trader-IQ4_XS.gguf
#
# Usage (from local host, after train finishes):
#   scp -P 13264 finetune/convert_to_gguf.sh root@79.160.189.79:/workspace/convert_to_gguf.sh
#   ssh -p 13264 root@79.160.189.79 "bash /workspace/convert_to_gguf.sh"
#   # then SCP the produced gguf back via finetune/phase4_pull.py

set -euo pipefail

MERGED_DIR=${1:-/workspace/output/merged}
OUT_DIR=${2:-/workspace/output/gguf}
LLAMA_DIR=${LLAMA_DIR:-/workspace/llama.cpp}
UPSTREAM_SHA=${UPSTREAM_SHA:-a135ec0baa1bcf7eb0437c9fd04920f87cf33ace}
PATCHES_DIR=${PATCHES_DIR:-/workspace/patches/llama-cpp}

mkdir -p "$OUT_DIR"

echo "[gguf] $(date +%T) ensure llama.cpp checkout at $UPSTREAM_SHA + apply MTP patches"
if [ ! -d "$LLAMA_DIR" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"
git fetch --depth 50 origin || true
git checkout "$UPSTREAM_SHA"
git reset --hard "$UPSTREAM_SHA"

# Upstream (pin a135ec0baa1b) now ships native MTP support via PR #22673 +
# fixes #23198/#23237. convert_hf_to_gguf.py handles mtp.* → blk.N.nextn.*
# remapping and emits nextn_predict_layers metadata natively. No patch needed.
# (Keep 0001-turboquant-base.patch in the tree but don't apply it here — it's
# only needed for the runtime container, not the converter.)

echo "[gguf] $(date +%T) install convert deps"
for f in requirements/requirements-convert_hf_to_gguf.txt; do
    if [ -f "$f" ]; then
        sed -i '/^torch/d; /^torchvision/d; /^torchaudio/d' "$f" || true
        pip install --quiet -r "$f" 2>&1 | tail -3 || true
    fi
done

echo "[gguf] $(date +%T) build llama-quantize (CUDA)"
if [ ! -x build/bin/llama-quantize ]; then
    cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF > /tmp/cmake.log 2>&1
    cmake --build build -j$(nproc) --target llama-quantize >> /tmp/cmake.log 2>&1
fi

BF16_GGUF="$OUT_DIR/qwen-trader-bf16.gguf"
IQ4_GGUF="$OUT_DIR/qwen-trader-IQ4_XS-Q8nextn.gguf"

echo "[gguf] $(date +%T) convert merged HF → bf16 GGUF (preserves nextn / MTP)"
# `convert_hf_to_gguf.py` automatically retains nextn_predict_layers metadata
# when the source config has it. Verify post-convert that mtp.* tensors landed.
python3 convert_hf_to_gguf.py "$MERGED_DIR" \
    --outtype bf16 \
    --outfile "$BF16_GGUF"

echo "[gguf] $(date +%T) verify MTP tensors in bf16 GGUF (gguf py module via pip)"
pip install --quiet gguf 2>&1 | tail -2 || true
python3 - <<PYEOF
import sys
try:
    import gguf
except ImportError:
    print("[gguf] gguf module not available; skipping MTP verify"); sys.exit(0)
r = gguf.GGUFReader('$BF16_GGUF')
mtp = [t.name for t in r.tensors if 'mtp' in t.name.lower() or 'nextn' in t.name.lower()]
print(f'[gguf] mtp/nextn tensor count: {len(mtp)}')
print(f'[gguf] first 5: {mtp[:5]}')
if not mtp:
    sys.exit('FAIL: no mtp/nextn tensors in bf16 GGUF — convert patch ineffective')
PYEOF

echo "[gguf] $(date +%T) quantize → IQ4_XS w/ Q8_0 nextn (matches prod IQ4_XS path)"
build/bin/llama-quantize \
    --tensor-type "nextn=Q8_0" \
    "$BF16_GGUF" "$IQ4_GGUF" IQ4_XS

echo "[gguf] $(date +%T) done"
ls -lh "$BF16_GGUF" "$IQ4_GGUF"
