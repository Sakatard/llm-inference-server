#!/usr/bin/env bash
# MTP draft-N × p_min sweep on P40.
# Re-creates container per combo, runs 3 generations, prints CSV row.
#
# Usage: ./scripts/mtp_sweep.sh [N_LIST] [P_LIST] [RUNS]
#   defaults: N="2 3 4"   P="0.0 0.25 0.5 0.75"   RUNS=3

set -euo pipefail
cd "$(dirname "$0")/.."

N_LIST="${1:-2 3 4}"
P_LIST="${2:-0.0 0.25 0.5 0.75}"
RUNS="${3:-3}"

PROMPT='Write a detailed 200-word essay about the history of the Roman Empire, covering its rise, peak, and fall.'
MAX_TOKENS=200
ENDPOINT="http://localhost:8088/v1/chat/completions"

COMPOSE_BAK="$(mktemp)"
cp docker-compose.yaml "$COMPOSE_BAK"
trap 'cp "$COMPOSE_BAK" docker-compose.yaml; rm -f "$COMPOSE_BAK"; echo "restored compose"' EXIT

wait_ready() {
  for _ in $(seq 1 60); do
    if curl -sf -m 3 http://localhost:8088/v1/models >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  echo "container never came up" >&2
  return 1
}

run_one() {
  local N=$1 P=$2
  sed -i -E "s/(MTP_DRAFT_N_MAX=)[0-9]+/\1${N}/" docker-compose.yaml
  if grep -q "MTP_DRAFT_P_MIN" docker-compose.yaml; then
    sed -i -E "s/(MTP_DRAFT_P_MIN=)[0-9.]+/\1${P}/" docker-compose.yaml
  else
    sed -i -E "/MTP_DRAFT_N_MAX=/a\      - MTP_DRAFT_P_MIN=${P}" docker-compose.yaml
  fi
  rtk docker compose down >/dev/null 2>&1 || true
  rtk docker compose up -d --force-recreate >/dev/null 2>&1
  wait_ready || return 1
  # warmup
  curl -s -X POST "$ENDPOINT" -H "Content-Type: application/json" \
    -d "$(jq -n --arg p "$PROMPT" '{model:"qwen",messages:[{role:"user",content:$p}],max_tokens:50,stream:false}')" \
    --max-time 60 >/dev/null

  local sum_tps=0 sum_acc=0 sum_draft=0
  for r in $(seq 1 "$RUNS"); do
    local resp
    resp=$(curl -s -X POST "$ENDPOINT" -H "Content-Type: application/json" \
      -d "$(jq -n --arg p "$PROMPT" --argjson m "$MAX_TOKENS" '{model:"qwen",messages:[{role:"user",content:$p}],max_tokens:$m,stream:false}')" \
      --max-time 120)
    local tps acc draft acc_n
    tps=$(echo "$resp" | jq -r '.timings.predicted_per_second // 0')
    draft=$(echo "$resp" | jq -r '.timings.draft_n // 0')
    acc_n=$(echo "$resp" | jq -r '.timings.draft_n_accepted // 0')
    acc=$(awk -v a="$acc_n" -v d="$draft" 'BEGIN{ if(d>0) printf "%.4f", a/d; else print "0" }')
    sum_tps=$(awk -v s="$sum_tps" -v v="$tps" 'BEGIN{printf "%.4f", s+v}')
    sum_acc=$(awk -v s="$sum_acc" -v v="$acc" 'BEGIN{printf "%.4f", s+v}')
    sum_draft=$(awk -v s="$sum_draft" -v v="$draft" 'BEGIN{printf "%.4f", s+v}')
    echo "    run=$r tps=$tps draft_n=$draft accept=$acc" >&2
  done
  local avg_tps avg_acc avg_draft
  avg_tps=$(awk -v s="$sum_tps" -v r="$RUNS" 'BEGIN{printf "%.2f", s/r}')
  avg_acc=$(awk -v s="$sum_acc" -v r="$RUNS" 'BEGIN{printf "%.4f", s/r}')
  avg_draft=$(awk -v s="$sum_draft" -v r="$RUNS" 'BEGIN{printf "%.1f", s/r}')
  echo "${N},${P},${avg_tps},${avg_acc},${avg_draft}"
}

echo "N,p_min,tok_per_s,accept,draft_n_avg"
for N in $N_LIST; do
  for P in $P_LIST; do
    echo "=== N=$N p_min=$P ===" >&2
    run_one "$N" "$P"
  done
done
