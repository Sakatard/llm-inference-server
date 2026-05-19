#!/usr/bin/env bash
# Cache-type sweep on P40 at fixed N=2, p_min=0.5 winner config.
# Re-creates container per cache type, runs 3 generations, prints CSV row.
#
# Usage: ./scripts/cache_sweep.sh [CACHE_LIST] [RUNS]
#   defaults: CACHE="turbo4 q8_0 f16"   RUNS=3
# Each cache type uses its server.py default ctx (65536 turbo4, 32768 others).

set -euo pipefail
cd "$(dirname "$0")/.."

CACHE_LIST="${1:-turbo4 q8_0 f16}"
RUNS="${2:-3}"

PROMPT='Write a detailed 200-word essay about the history of the Roman Empire, covering its rise, peak, and fall.'
MAX_TOKENS=200
ENDPOINT="http://localhost:8088/v1/chat/completions"

COMPOSE_BAK="$(mktemp)"
cp docker-compose.yaml "$COMPOSE_BAK"
trap 'cp "$COMPOSE_BAK" docker-compose.yaml; rm -f "$COMPOSE_BAK"; echo "restored compose"' EXIT

wait_ready() {
  for _ in $(seq 1 90); do
    if curl -sf -m 3 http://localhost:8088/v1/models >/dev/null 2>&1; then
      # also wait until qwen route is actually ready (model loaded)
      for _ in $(seq 1 60); do
        local h
        h=$(curl -s -m 3 http://localhost:8088/health 2>/dev/null)
        if echo "$h" | jq -e '.active.qwen == true' >/dev/null 2>&1; then return 0; fi
        sleep 2
      done
      return 0
    fi
    sleep 2
  done
  echo "container never came up" >&2
  return 1
}

run_one() {
  local CACHE=$1
  sed -i -E "s/(MTP_CACHE_TYPE=)[A-Za-z0-9_]+/\1${CACHE}/" docker-compose.yaml
  rtk docker compose down >/dev/null 2>&1 || true
  rtk docker compose up -d --force-recreate >/dev/null 2>&1
  wait_ready || return 1
  # warmup — also forces model load on qwen route
  curl -s -X POST "$ENDPOINT" -H "Content-Type: application/json" \
    -d "$(jq -n --arg p "$PROMPT" '{model:"qwen",messages:[{role:"user",content:$p}],max_tokens:50,stream:false}')" \
    --max-time 180 >/dev/null

  local sum_tps=0 sum_acc=0 sum_draft=0 sum_prompt_tps=0
  for r in $(seq 1 "$RUNS"); do
    local resp tps draft acc_n acc prompt_tps
    resp=$(curl -s -X POST "$ENDPOINT" -H "Content-Type: application/json" \
      -d "$(jq -n --arg p "$PROMPT" --argjson m "$MAX_TOKENS" '{model:"qwen",messages:[{role:"user",content:$p}],max_tokens:$m,stream:false}')" \
      --max-time 180)
    tps=$(echo "$resp" | jq -r '.timings.predicted_per_second // 0')
    draft=$(echo "$resp" | jq -r '.timings.draft_n // 0')
    acc_n=$(echo "$resp" | jq -r '.timings.draft_n_accepted // 0')
    prompt_tps=$(echo "$resp" | jq -r '.timings.prompt_per_second // 0')
    acc=$(awk -v a="$acc_n" -v d="$draft" 'BEGIN{ if(d>0) printf "%.4f", a/d; else print "0" }')
    sum_tps=$(awk -v s="$sum_tps" -v v="$tps" 'BEGIN{printf "%.4f", s+v}')
    sum_acc=$(awk -v s="$sum_acc" -v v="$acc" 'BEGIN{printf "%.4f", s+v}')
    sum_draft=$(awk -v s="$sum_draft" -v v="$draft" 'BEGIN{printf "%.4f", s+v}')
    sum_prompt_tps=$(awk -v s="$sum_prompt_tps" -v v="$prompt_tps" 'BEGIN{printf "%.4f", s+v}')
    echo "    run=$r tps=$tps prompt_tps=$prompt_tps draft_n=$draft accept=$acc" >&2
  done
  local avg_tps avg_acc avg_draft avg_prompt
  avg_tps=$(awk -v s="$sum_tps" -v r="$RUNS" 'BEGIN{printf "%.2f", s/r}')
  avg_acc=$(awk -v s="$sum_acc" -v r="$RUNS" 'BEGIN{printf "%.4f", s/r}')
  avg_draft=$(awk -v s="$sum_draft" -v r="$RUNS" 'BEGIN{printf "%.1f", s/r}')
  avg_prompt=$(awk -v s="$sum_prompt_tps" -v r="$RUNS" 'BEGIN{printf "%.1f", s/r}')
  # Capture VRAM from nvidia-smi
  local vram
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
  echo "${CACHE},${avg_tps},${avg_acc},${avg_draft},${avg_prompt},${vram}"
}

echo "cache,tok_per_s,accept,draft_n_avg,prompt_tps,vram_mib"
for CACHE in $CACHE_LIST; do
  echo "=== cache=$CACHE ===" >&2
  run_one "$CACHE" || echo "${CACHE},FAIL,FAIL,FAIL,FAIL,FAIL"
done
