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
UPSTREAM_SHA=${UPSTREAM_SHA:-253ba110bcd372207ca7b0bb56f1ea10d60d53fd}

mkdir -p "$OUT_DIR"

echo "[gguf] $(date +%T) ensure llama.cpp checkout at $UPSTREAM_SHA"
if [ ! -d "$LLAMA_DIR" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"
git fetch --depth 50 origin || true
git checkout "$UPSTREAM_SHA"

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

echo "[gguf] $(date +%T) verify MTP tensors in bf16 GGUF"
build/bin/llama-quantize --help 2>&1 | head -2
build/bin/llama-quantize --quantize-output-tensor "$BF16_GGUF" 2>/dev/null || true
# Cheap check: count of mtp tensors via gguf-dump
python3 -c "
import gguf
r = gguf.GGUFReader('$BF16_GGUF')
mtp_tensors = [t.name for t in r.tensors if t.name.startswith('mtp.') or '.mtp.' in t.name]
print(f'[gguf] mtp tensor count: {len(mtp_tensors)}')
if not mtp_tensors:
    raise SystemExit('FAIL: no mtp.* tensors in bf16 GGUF')
"

echo "[gguf] $(date +%T) quantize → IQ4_XS w/ Q8_0 nextn (matches prod IQ4_XS path)"
build/bin/llama-quantize \
    --tensor-type "nextn=Q8_0" \
    "$BF16_GGUF" "$IQ4_GGUF" IQ4_XS

echo "[gguf] $(date +%T) done"
ls -lh "$BF16_GGUF" "$IQ4_GGUF"
