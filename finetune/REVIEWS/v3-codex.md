Reading additional input from stdin...
OpenAI Codex v0.129.0 (research preview)
--------
workdir: /home/xel/containers/llm-inference-server
model: gpt-5.5
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: xhigh
reasoning summaries: none
session id: 019e2cb0-ab1f-7ff1-a0f9-8028f05db1fa
--------
user
Re-review fine-tune spec v3. Verdict: PROCEED or BLOCKER. This is review round 4. You previously voted BLOCKER on v0, v1, v2. v3 ships:

v2 → v3 fixes (5 items, all from your v2 reviews):
1. Phase -1 wording fixed — SII-WANGZJ no longer claims to "solve orderbook reconstruction entirely"; now explicitly: trade-flow + market-metadata only, no CLOB depth. (codex v2)
2. §5 row filter v3 — drops only high-confidence wrong rows (probability > 0.85). Keeps low-conf wrong rows AND correct rows at weight 1.0. Resolves the v2 reviewer conflict: codex wanted "don't bias toward easy markets" (keep wrong rows) vs gemini wanted "no poison pills" (drop full-gradient wrong-label training). Compromise threshold removes egregious teacher errors only. (gemini v2 + codex v0)
3. §7 chat template — `enable_thinking=False` LOCKED everywhere, MANDATORY language, pre-flight assertion before training. No copy-paste footgun. (codex v2)
4. §12 locked decisions reworded — "audit only + drop high-conf wrong; all emitted rows weight 1.0." No more "downweight" language that contradicts §5. (codex v2)
5. CC-BY-SA legal gate — explicit sub-step in Phase -1. trentmkelly is CC-BY-SA-4.0; spec now requires explicit user decision (accept ShareAlike / reject / eval-only). Default if no decision = reject + fall back to scraping. (gemini v2)

Verdict: PROCEED or BLOCKER. If BLOCKER, list ONLY new killer issues or remaining textual inconsistencies. Don't re-litigate fixes that are addressed. Under 300 words.

Spec below.

---

# Polymarket / Crypto Trade-Decision Fine-Tune — Specification v3

**Status:** v3 revision. v2 was BLOCKER on 3 codex textual contradictions + 2 gemini design issues (poison pills + CC-BY-SA). v3 cleans up the contradictions, adds drop-high-conf-wrong row filter (replaces broken loss weighting), and adds explicit CC-BY-SA legal gate. Pending fourth review round before lock.
**Owner:** inference-server-engineer (project agent)
**Training base model:** `unsloth/Qwen3.5-9B` (Unsloth's Hugging Face repo of the post-trained variant; same weights, with Unsloth tooling preconfigured)
**Inference base GGUF:** `unsloth/Qwen3.5-9B-GGUF` → `Qwen3.5-9B-Q4_K_M.gguf` (Unsloth's K-quants; we deploy the merged LoRA on top)
**Source lock:** All Qwen models pulled from `unsloth/*` on HF, not `Qwen/*` direct, per project convention. Same weights, but Unsloth's variants include tokenizer fixes and prepared 4-bit/MTP variants.
**Training:** Unsloth QLoRA on rented Vast.ai 4090 24 GiB
**Deploy target:** P40 (24 GiB, Pascal sm_61) as new `qwen-trader` service
**Schemas:** [`finetune/schemas/polymarket_decision.schema.json`](./schemas/polymarket_decision.schema.json) + [`finetune/schemas/crypto_decision.schema.json`](./schemas/crypto_decision.schema.json) — two domain-specific contracts, NOT one schema with nullable extension.

Cross-model review v0 findings: `finetune/REVIEWS/v0-gemini.md` + `finetune/REVIEWS/v0-codex.md` (to be archived from this session).

---

## 0b. Why v3 exists (delta from v2)

v2 was BLOCKER per both reviewers. v3 fixes:

| Issue (source) | v3 fix |
|---|---|
| Phase -1 still claimed SII-WANGZJ "solves orderbook reconstruction entirely" — contradicts §0a (codex v2) | Phase -1 reworded to: "primary trade-flow + market-metadata source. Does NOT provide full CLOB depth." |
| §12 locked decisions still said "downweight + audit instead" — contradicts §5 v2 (codex v2) | §12 reworded to: "audit only; all rows that pass filter get weight 1.0; high-confidence-wrong rows are filtered out before training." |
| §7 `apply_chat_template` example missing `enable_thinking=False` — copy-paste footgun (codex v2) | §7 Format row updated to include `enable_thinking=False` everywhere. |
| All-rows-weight-1.0 = poison pills with full gradient on wrong answers (gemini v2) | **Drop high-confidence wrong rows (conf > 0.85).** Keep low-conf wrong rows AND correct rows at weight 1.0. Compromise between codex's "don't bias toward easy" and gemini's "don't train on confident garbage." Threshold 0.85 is stricter than v1's 0.75 — removes only egregious teacher errors, not all uncertainty. |
| CC-BY-SA 4.0 copyleft risk on trentmkelly (gemini v2) | Phase -1 adds explicit legal-gate sub-step. If user does not accept ShareAlike risk on the LoRA adapter, fall back to scraping trentmkelly's source (Polymarket directly). Documented as a gate, not assumed permissible. |

## 0a. Why v2 existed (delta from v1)

v1 was a **BLOCKER** per both reviewers. v2 addresses every v1 finding:

| v1 Decision | v1 Problem | v2 Fix |
|---|---|---|
| Per-sample loss weight=0.5 for high-conf wrong rows | SFTTrainer doesn't support per-sample weighting natively; needs custom DataCollator + compute_loss override → scope creep | **Drop downweighting. All rows weight 1.0.** Drop only malformed rows. Audit log retained for post-hoc analysis. |
| "SII-WANGZJ solves orderbook reconstruction entirely" | SII-WANGZJ has trades + raw OrderFilled events + market metadata, but NOT full CLOB add/cancel/depth history | **SII-WANGZJ = trade-price + market-metadata source.** Phase 3 stays mandatory. Crypto orderbook comes from trentmkelly native snapshots. Polymarket microstructure in v0 = derived from trade flow only (mid-price from last trades, spread approx from bid-ask trade interleave). Accept reduced microstructure quality for non-crypto in v0. |
| Decision-time selector `{open+24h, midpoint, close-7d}` global | Invalid for short markets (5/15-min crypto markets, hour-scale Polymarket events) | **Per-dataset rules.** See §4 v2. trentmkelly uses native per-step snapshots; SII-WANGZJ Polymarket uses percentile-of-active-duration (25/50/75%); markets <2hr active use random in-window sample. |
| `tokenizer.apply_chat_template(tokenize=False)` | Qwen3.5 defaults to thinking mode; `<think>` open tag with JSON grammar = corrupted output | **Lock `enable_thinking=False`** in dataset rendering AND llama.cpp serving (`--jinja` with template kwargs override or pre-rendered prompts). Document the failure mode. |
| `FastLanguageModel.from_pretrained` as Unsloth loader | Qwen3.5-9B is multimodal (vision-language). Unsloth's text loader may not handle vision encoder. Native pipeline_tag = `image-text-to-text`. | **Phase 0 verifies correct loader API.** Try `FastModel` (multimodal) → `FastLanguageModel` (text-only) → raw `AutoModelForImageTextToText`. Document which works. Fallback: `unsloth/Qwen3-8B` (text-only Qwen3 family). |

Minor v2 fixes (not BLOCKER-level):
- **Teacher-runner pin scope** expanded: pin committee code SHA + model IDs (gpt-5.5 referee, deepseek-v3.2 theses) + prompt template strings + sampling temps + OpenRouter provider versions.
- **Schema wording**: "PolymarketDecision uses same controlled-vocab enums as RefereeOutput; it is owned by our project, not polymarket-agents." Decouple the contract from teacher internals at the documentation level.

## 0. Why v1 existed (delta from v0)

v0 was a **BLOCKER** per both reviewers. v1 addresses every finding:

| v0 Decision | v0 Problem | v1 Fix |
|---|---|---|
| Base = Qwen3.5-7B-Instruct | Doesn't exist publicly (Qwen3.5 sizes: 0.8/2/4/9/27B) | Base = `Qwen/Qwen3.5-9B` |
| Teacher = import polymarket-agents live committee | Coupling trap; teacher flaws baked in | Frozen `teacher-runner` contract; offline label generation; reuse connectors only via stable APIs |
| 50/500/2000 volume curve | 500 examples → schema-only learning, no judgment | Smoke 200 → learning-curve [250, 500, 1000, 2000]; holdout 200 time-stratified |
| `RefereeOutput` + nullable `crypto_extension` | Schema debt; null-spam; teacher internals leaked into product | Two schemas: `PolymarketDecision`, `CryptoDecision`. Shared core + domain-specific fields. |
| Pascal LoRA deploy unproven | 80-150 tok/s estimate is fantasy on sm_61 | **Mandatory Pascal benchmark gate (Phase 0)** before training run. Measure base + dummy LoRA on actual P40 at 16k ctx. |
| Runtime `--lora` flag | Pascal does adapter math at fp16, slow | Merge LoRA into fp16 base before GGUF quantization. Single Q4_K_M output. |
| Discard rows where committee disagrees with outcome | Outcome-oracle selection bias; destroys calibration; deletes hard markets | Keep all rows. Downweight high-confidence wrong teacher samples (loss multiplier 0.5). Send broken/ambiguous rows to audit. |
| Rank 32 alpha 64 | Memorization-prone on 500 examples | Start rank 8 alpha 16 dropout 0.1. Rank-search [8, 16, 32] only after smoke train confirms pipeline works. |
| Manual ChatML JSONL | Tokenization mismatch footgun (Qwen has thinking/no-thinking template branches) | Render with `tokenizer.apply_chat_template(tokenize=False)` at dataset-build time. Store prompt hashes. Serve with matching `--jinja` template. |
| Volume-weighted midpoint as decision time | Target leakage (uses future volume to pick timestamp) | Ex-ante deterministic timestamps: random sample from {open+24h, mid-active-period, last-active-week} seeded by market_id. News window strictly BEFORE decision time. |
| 8k seq len + bs2 | OOM risk on 4090 24 GiB | 4k seq len v0. Truncate news section first if exceeded. |
| Fine-tune optional/skipped baseline | May not need FT — base + JSON grammar may suffice | **Baseline gate (Phase 1)**: eval `Qwen3.5-9B + JSON grammar + system prompt` on 50 examples. If schema_validity ≥ 95% AND accuracy ≥ committee on this set, fine-tuning is style-only or skipped entirely. |

## 1. Phased gates

Each phase has a verification gate. Cannot proceed without passing.

### Phase -1 — Existing HF dataset inventory (1 day)

**Critical reuse opportunity discovered during planning.** Multiple HF datasets already cover most of our raw-data needs. Inventory + decide reuse strategy BEFORE building any scraping infra.

**Confirmed candidates:**

| Dataset | Size | Coverage | Use case for us |
|---|---|---|---|
| `SII-WANGZJ/Polymarket_data` | 1.9B records / 538k markets / 163 GB / MIT | Trade events (OrderFilled blockchain logs) + trades-with-market-linkage + market metadata + derived per-user quants. **Does NOT contain full CLOB add/cancel/depth-snapshot history.** | **Primary trade-flow + market-metadata source.** Microstructure features for non-crypto Polymarket markets in v0 are derived from trade flow only (mid-price from recent trades, spread approximated from bid-ask trade interleave). Full orderbook depth NOT available from this dataset. |
| `trentmkelly/polymarket_crypto_derivatives` | 13k dl / Feb-Mar 2026 / cc-by-sa-4.0 | High-freq orderbook for BTC/ETH/SOL/XRP up-down markets, 5/15-min intervals, 100ms decision-snapshot cadence. steps.parquet + events.parquet + book_levels.parquet. | **Direct crypto-decision raw source** for crypto-trader scope. Decision snapshots already aligned to market-resolution windows. |
| `puneeth/crypto-trading-r1-sft` | 2437 examples | Crypto SFT format (messages/role/content + label/label_text/date/price). ML Intern auto-generated. | **Supplementary training data** — needs schema conversion to our two-decision contract, but adds free volume. Likely classification-grade not reasoning-grade — quality assessment in Phase -1. |
| `CK0607/closed-polymarket-2025H1` | 10k-100k records / 2025-07 | Closed Polymarket events H1 2025 | **Backup raw source** for the resolution-outcome ground-truth column. |
| `2084Collective/prediction-markets-historical-v5` | 1M-10M records / 2025-11 | Multi-platform prediction markets historical | **Diversity supplement** for non-Polymarket prediction-market markets if scope expands. |

**Phase -1 tasks:**
1. Pull dataset cards + first 100 rows of each via `datasets.load_dataset(..., split='train', streaming=True)`.
2. Document: exact schema, license, time-coverage span, joinability with GDELT news on `market_id` or `timestamp`.
3. Decide per-dataset: **raw source** / **labels** / **supplementary** / **skip**.
4. **License legal-gate (gemini v2 fix):**
   - `SII-WANGZJ` is MIT — clean, no issues.
   - `trentmkelly` is **CC-BY-SA-4.0** (copyleft / ShareAlike). Fine-tuning on CC-BY-SA data may impose ShareAlike on the resulting LoRA adapter. This is **legally ambiguous** for ML training in many jurisdictions and is a strict-ban for many corporate policies.
   - **Sub-gate:** explicit user/owner decision REQUIRED before using trentmkelly. Options:
     - (a) Accept ShareAlike risk on the qwen-trader adapter (must then publish adapter under cc-by-sa-4.0 if redistributed).
     - (b) Reject trentmkelly. Fall back to scraping crypto orderbook directly from Polymarket CLOB API or a different licensed source.
     - (c) Use trentmkelly only for evaluation (not training), keeping the trained adapter clean.
   - **Default if no decision: reject (option b).** Scraping is more work but legally clean.
   - Other dataset licenses (puneeth: not explicitly stated; CK0607: not explicitly stated; 2084Collective: not explicitly stated) require similar review before use.

**Gate:** at minimum `SII-WANGZJ/Polymarket_data` confirmed loadable + market-metadata layer joinable with our chosen news connectors. AND license gate decision recorded per dataset. If neither SII-WANGZJ nor a permissive crypto source is usable, fall back to Polymarket Gamma API scraping (Phase 3 fallback path).

**Cost savings if reuse works:** Phase 3 (data archival verification) for the trade-flow portion becomes a 1-hour test instead of a 1-day data engineering exercise. Full orderbook reconstruction (crypto only) still requires either trentmkelly (license-gated) OR direct Polymarket CLOB scraping (clean but slower).

### Phase 0 — Pascal deployment benchmark (1-2 hours)
1. Download `unsloth/Qwen3.5-9B-GGUF` → `Qwen3.5-9B-Q4_K_M.gguf` directly (skip the fp16 + quantize step for benchmark — same final shape as merged-LoRA-then-quantize output)
2. Optionally create a dummy rank-8 LoRA later via Unsloth (`FastLanguageModel.get_peft_model`) to test the merge+quantize pipeline end-to-end before Phase 4
3. Deploy temporarily on P40 at 16k ctx, no MTP
4. Benchmark: PONG sanity + 200-token essay generation, capture decode tok/s + VRAM peak

**Gate:** ≥ 25 tok/s decode on P40 at 16k ctx with base Q4_K_M (no LoRA). If < 25 tok/s, reconsider scale: try `unsloth/Qwen3.5-4B-GGUF` (smaller) OR accept slower service.

### Phase 1 — Baseline-without-FT (1 day)
1. Hold out 50 markets (random sample from filter set — see Phase 3)
2. Run `Qwen/Qwen3.5-9B` (Q4_K_M, no LoRA) + locked system prompt + llama.cpp JSON grammar constraint against each held-out market
3. Score: schema_validity, outcome_accuracy, calibration (Brier score)

**Gate:** if schema_validity ≥ 95% AND outcome_accuracy ≥ committee_accuracy_on_holdout, **fine-tuning is NOT NEEDED** for v0 scope. Pivot to: better system prompt + grammar tuning only. Skip Phases 3-6.

If gate fails (likely), proceed to Phase 3.

### Phase 2 — Frozen teacher-runner contract (2 days)
1. Define `TeacherContextBundle` Pydantic model — versioned, exact shape the committee consumes
2. Define `TeacherLabel` Pydantic model — versioned, matches the new domain-specific decision schemas
3. Write `finetune/teacher_runner/run.py` — single entrypoint that takes a `TeacherContextBundle`, runs the polymarket-agents committee (via subprocess or HTTP), returns `TeacherLabel`
4. Pin committee code to a specific commit SHA in `finetune/teacher_runner/PINNED_SHA.txt`
5. Smoke-test on 5 historical markets — verify output validates

**Gate:** 5/5 smoke tests produce schema-valid labels. Both decision schemas exercised (3 polymarket + 2 crypto).

### Phase 3 — Data archival verification (1 day)
**Codex flagged this risk: Gamma closed-market data may NOT reconstruct historical orderbooks/news.**

1. Pull 20 resolved markets from various dates (recent: 30 days ago; mid: 90 days; old: 180 days)
2. For each: verify we can recover orderbook depth at the chosen decision moment AND retrieve news from the pre-decision window
3. Document loss rates: % markets with full reconstruction, % degraded, % unreconstructable

**Gate:** ≥ 70% full-reconstruction rate within the 180-day lookback window. If < 70%, reduce lookback window or accept partial-context training.

### Phase 4 — Smoke train (200 examples, 1 day)
1. Run Phase 2 teacher-runner on 200 markets meeting Phase 3 reconstruction criteria
2. Format JSONL with `tokenizer.apply_chat_template`
3. Train Qwen3.5-9B QLoRA rank 8 alpha 16 dropout 0.1, seq 4k, 3 epochs on Vast.ai 4090
4. Eval on 50-market holdout (same as Phase 1 baseline)

**Gate:** delta vs Phase 1 baseline — schema_validity ↑ OR outcome_accuracy ↑ OR Brier ↓. If no improvement, the dataset/teacher is broken — debug before scaling.

### Phase 5 — Learning curve (1 day each at 250, 500, 1000, 2000)
1. Re-train at increasing dataset sizes using same train/eval split
2. Plot: schema_validity, outcome_accuracy, Brier vs N

**Gate:** schema_validity plateaus ≥ 98%; outcome_accuracy converges. Pick the smallest N at plateau.

### Phase 6 — Rank search (optional, after Phase 5 converges)
Try rank [8, 16, 32] at chosen N. Pick best by holdout metric.

### Phase 7 — Deploy
1. Merge final LoRA into fp16 base
2. Quantize to Q4_K_M
3. Add `qwen-trader` service to server.py (NO `--lora` runtime flag; merged GGUF only)
4. Production smoke test on 10 live markets

## 2. Base model selection

| Candidate | Pros | Cons |
|---|---|---|
| `unsloth/Qwen3.5-9B` (training) + `unsloth/Qwen3.5-9B-GGUF` (inference) | Same family as prod 27B; Unsloth-supported; preconfigured tokenizer fixes; MTP variant exists | Multimodal-tagged (we don't use vision); larger VRAM |
| `unsloth/Qwen3.5-4B` + `Qwen3.5-4B-GGUF` | Smaller, faster, less VRAM | Lower reasoning ceiling |
| `unsloth/Qwen3-8B` | Older Qwen3 family but proven | Different tokenizer family from prod 27B |

**Choice: Unsloth Qwen3.5-9B.** Same tokenizer/family as prod, has MTP support for future, fits 4090 QLoRA, fits P40 inference at Q4_K_M. Locked to Unsloth's variant — `unsloth/Qwen3.5-9B` for training (safetensors w/ Unsloth tokenizer fixes), `unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf` for inference.

Fallback if Phase 0 fails: `unsloth/Qwen3.5-4B` + corresponding GGUF.

## 3. Two-schema contract (replaces v0 single-schema with nullable extension)

### `polymarket_decision.schema.json` v0

```jsonc
{
  "schema_version": "polymarket_decision_v0",
  "summary": "...",
  "global_confidence": 0.0-1.0,
  "choice_assessments": [
    {
      "label": "YES|NO|<outcome_name>",
      "probability": 0.0-1.0,
      "confidence": 0.0-1.0,
      "evidence": { "reliability", "diversity", "recency", "overall_strength" },
      "data_coverage": 0.0-1.0,
      "contradictions": [...],
      "key_points": [...]
    }
  ],
  "uncertainty_flags": [...controlled vocab...],
  "trade_blockers": [...controlled vocab...],
  "rationale": "..."
}
```

Mirrors polymarket-agents `RefereeOutput` exactly. No extension fields.

### `crypto_decision.schema.json` v0

```jsonc
{
  "schema_version": "crypto_decision_v0",
  "symbol": "BTC/USDT",
  "interval": "1h|4h|1d",
  "summary": "...",
  "global_confidence": 0.0-1.0,
  "choice_assessments": [
    {"label": "LONG|SHORT|HOLD", "probability": 0.0-1.0, "confidence": 0.0-1.0, "evidence": {...}, "data_coverage": 0.0-1.0, "contradictions": [...], "key_points": [...]}
  ],
  "uncertainty_flags": [...controlled vocab...],
  "trade_blockers": [...controlled vocab...],
  "rationale": "...",
  "timefm_score": -1.0 to 1.0,
  "news_score": -1.0 to 1.0
}
```

`symbol`, `interval`, `timefm_score`, `news_score` are required (not nullable). Crypto-trader's `Signal` dataclass maps cleanly:
- `Signal.symbol` ← `symbol`
- `Signal.direction` ← `argmax(choice_assessments where label != HOLD).label.lower()`
- `Signal.confidence` ← `global_confidence`
- `Signal.timefm_score` ← `timefm_score`
- `Signal.news_score` ← `news_score`

System prompts are domain-locked: trader receives different system prompt for polymarket vs crypto. Routes via `/v1/trade/polymarket` and `/v1/trade/crypto`.

## 4. Decision-moment selection v2 (per-dataset, codex v1 fix)

The global `{open+24h, midpoint, close-7d}` selector is invalid for short markets. v2 uses per-dataset rules:

### 4a. trentmkelly crypto derivatives (5/15-min up-down markets)

Use the dataset's **native per-step decision snapshots** (already aligned by the dataset author to be ex-ante decisions). Pick ONE step per market via seeded random:

```python
import hashlib
steps = load_steps_parquet(market_id)
seed = int.from_bytes(hashlib.sha256(market_id.encode()).digest()[:8], 'big')
chosen_step = steps[seed % len(steps)]
decision_time = chosen_step['ts']
```

News window: `[decision_time - 24h, decision_time]` (shorter window — these are minute-scale markets, week-old news isn't predictive).

### 4b. SII-WANGZJ / Polymarket general markets

Compute the **active duration window** (first trade timestamp → resolution event), then sample at one of three percentiles:

```python
active_start = first_trade_ts(market_id)
active_end   = resolution_ts(market_id)
duration     = active_end - active_start

if duration < 2 * 60 * 60:   # < 2 hours
    seed = sha256(...).digest()[0]
    decision_time = active_start + (seed / 255.0) * duration   # random in-window
else:
    percentiles = [0.25, 0.50, 0.75]
    decision_time = active_start + percentiles[seed % 3] * duration
```

News window: `[decision_time - 7d, decision_time]` for multi-day markets, `[decision_time - 24h, decision_time]` for short markets.

### 4c. Universal guards

- News window strictly past — drop articles published ≥ decision_time.
- All timestamps UTC.
- Decision_time must fall within a period where the market had ≥10 trades in the trailing 1 hour (avoid dead-window decision times).
- Document chosen decision_time in dataset row metadata for audit.

## 4-OLD (v1 deprecated — kept for review trail)
Original global selector: `{open+24h, midpoint, close-7d}`. Replaced by per-dataset rules above.

## 5. Teacher labeling — v3 compromise filter (replaces v0 full discard, v1 weighting, v2 keep-all)

v1 proposed per-sample weights but SFTTrainer doesn't support them natively. v2 kept all rows at weight 1.0 — gemini correctly flagged this as poisoning the dataset with full-gradient training on confident wrong labels.

v3 compromise: **drop only the high-confidence wrong rows** (poison pills). Keep low-confidence wrong rows (legitimate uncertainty signal). Keep all correct rows.

```python
HIGH_CONF_WRONG_THRESHOLD = 0.85   # stricter than v1's 0.75 — only filter egregious teacher errors

for market m in dataset:
    bundle = build_context_bundle(m, decision_time)
    label = teacher_runner.run(bundle)
    actual = m.resolved_outcome

    if label is None or not validates(label, schema):
        skip(m)  # literal broken — drop
        continue

    teacher_picked = argmax(label.choice_assessments, key=probability)
    teacher_wrong = teacher_picked.label != actual

    if teacher_wrong and teacher_picked.probability > HIGH_CONF_WRONG_THRESHOLD:
        # Drop: training on confident garbage poisons the model
        log_to_audit("high_conf_wrong", m, label)
        skip(m)
        continue

    if teacher_wrong:
        log_to_audit("low_conf_wrong", m, label)   # kept but flagged
    row.weight = 1.0
    emit(row)
```

Why this works (resolves v2 reviewer conflict):
- Codex v0 worried: "outcome-oracle filtering biases toward easy markets." → v3 doesn't filter ALL wrong rows, only the egregiously-confident-wrong ones. Hard markets where teacher said "0.55 wrong" stay in the dataset.
- Gemini v2 worried: "weight=1.0 on confident wrong = poison pill." → v3 removes the poison pills explicitly.
- SFTTrainer compatibility: no custom DataCollator needed. All emitted rows get default weight 1.0.

Expected drop rate at threshold 0.85: ~5-15% of dataset (committee is mostly right; very-high-confidence-wrong is rare). Document actual drop rate post-Phase-2.

## 6. Volume curve (replaces v0 50/500/2000)

| Phase | N train | N holdout | Purpose |
|---|---|---|---|
| Smoke | 200 | 50 | Verify pipeline end-to-end produces a measurable delta over baseline |
| Learning curve point 1 | 250 | 200 | Establish baseline |
| Point 2 | 500 | 200 | Find scaling slope |
| Point 3 | 1000 | 200 | Find plateau region |
| Point 4 | 2000 | 200 | Confirm plateau or detect overfitting |

Holdout = 200 markets, time-stratified (last 10% by resolution date) AND category-stratified (≤25% any single category).

## 7. Training stack (replaces v0)

| Layer | v1 Choice |
|---|---|
| Base | `unsloth/Qwen3.5-9B` (Unsloth's post-trained variant — NOT `-Base`). **Multimodal** (vision-language, native pipeline_tag `image-text-to-text`); text-only fine-tune still works via standard SFT but loader API varies. |
| 4-bit loader | **TBD pending Phase 0 verification.** Candidates in order of preference: `FastModel.from_pretrained` (Unsloth multimodal API) → `FastLanguageModel.from_pretrained` (text-only) → raw `AutoModelForImageTextToText` + manual PEFT. Phase 0 decides. Fallback if all fail: `unsloth/Qwen3-8B` (text-only Qwen3 family). |
| Chat template | **MANDATORY:** `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)`. **The `enable_thinking=False` kwarg is LOCKED in every render path.** Pre-flight check before training: `assert "<think>" not in any rendered_text`. |
| Format | Each emitted row's "text" field MUST be the full string from the above call with `enable_thinking=False`. No exceptions. Dataset-build script asserts this at write time. |
| Method | QLoRA |
| Initial rank | 8 (alpha 16, dropout 0.1) |
| Rank search | 8/16/32 in Phase 6 only |
| Target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| Optimizer | adamw_8bit, lr 2e-4, cosine, warmup 0.03 |
| Sequence length | **4096** (v0 had 8192 — OOM risk) |
| Batch | grad_accum=8, per_device=2 (effective 16) |
| Epochs | 3 smoke; tuned per learning-curve point |
| Hardware | Vast.ai RTX 4090 24 GiB |
| Format | `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)` — pre-rendered text field, not OpenAI ChatML JSON |

## 8. Inference at deploy time (replaces v0 runtime --lora; v2 locks no-thinking)

After Phase 7 merge:
1. Save merged model via the loader API decided in Phase 0 — likely `unsloth.FastModel.save_pretrained_merged(...)` or `FastLanguageModel.save_pretrained_merged(...)` → fp16 merged base+LoRA
2. `convert_hf_to_gguf.py` → fp16 GGUF
3. `llama-quantize ... Q4_K_M` → final GGUF
4. Deploy:
   ```python
   qwen_trader = Service("qwen-trader", [
       "llama-server",
       "-m", "/models/qwen-trader-v0-Q4_K_M.gguf",   # merged, no --lora flag
       "-c", "16384",
       "--cache-type-k", "q8_0",
       "--cache-type-v", "q8_0",
       "-ngl", "99",
       "--host", "127.0.0.1",
       "--port", "9186",
       "--no-mmap",
       "--no-webui",
       "--jinja",
       "--reasoning", "none",                         # disable thinking — critical: <think> tags break JSON grammar
       "--grammar-file", "/app/grammars/trade_decision.gbnf",
   ], port=9186)
   ```
5. **Caller MUST send `chat_template_kwargs: {"enable_thinking": false}` in every request body OR llama-server must have a server-side override.** Verify in Phase 7 production smoke test that no `<think>` tag appears in the model output stream.

6. Routes: `/v1/trade/polymarket` + `/v1/trade/crypto` per system-prompt domain.

JSON grammar file enforces schema validity at sampling time. Generated from the JSON schema via `json-schema-to-grammar.py`. With `<think>` disabled, the model emits the JSON object directly — no thinking-trace preamble that would violate the grammar.

## 9. Expected metrics (per Pascal reality check)

Per reviewers, original 80-150 tok/s estimate was fantasy. Realistic on P40 Q4_K_M no-MTP:
- ~30-60 tok/s decode (Pascal, no FA2, no tensor cores)
- ~150-300 ms latency for short answers
- ~5 GB VRAM steady-state

If Phase 0 measures <30 tok/s, the realistic next-step is Qwen3.5-4B (smaller weights → less memory bandwidth → faster).

## 10. Cost (revised after Phase -1 reuse)

| Item | Cost |
|---|---|
| Existing HF datasets (SII-WANGZJ + trentmkelly + puneeth + CK0607) | $0 (download bandwidth only — ~170 GB if SII-WANGZJ full) |
| OpenRouter LLM calls during teacher labeling (2000 markets × ~3 committee calls × ~3k tokens each) | $30-60 |
| News/connector calls (GDELT free, Google RSS free, NewsAPI optional) | $0-50 |
| Vast.ai 4090 — 6 train runs (smoke + 4 learning-curve + rank search) × 1-2 hr each | $4-10 |
| Phase 0 + 1 setup runs on P40 (local) | $0 |
| **Total v0+v1** | **$35-120** (was $35-120 in v0 too — savings come from time, not $$$) |

**Time saved by HF dataset reuse:** Phase 3 (data archival) drops from 1 day → 1 hour. Total elapsed time 1-2 weeks → ~1 week.

## 11. Task ownership

Each phase has a single owner-task in TaskCreate. No phase starts before prior gate passes.

## 12. Locked decisions

- One base model family: **Unsloth Qwen3.5-9B** (`unsloth/Qwen3.5-9B` for training, `unsloth/Qwen3.5-9B-GGUF` Q4_K_M for inference).
- Two schemas: PolymarketDecision + CryptoDecision (separate, not nullable extension).
- **Existing HF datasets first**: `SII-WANGZJ/Polymarket_data` (MIT, trade-flow + metadata only — no full CLOB depth). `trentmkelly/polymarket_crypto_derivatives` available but **license-gated** (CC-BY-SA-4.0 — copyleft risk; user decision required before use).
- **Row filtering (v3):** audit-only logging for low-confidence wrong rows; **drop high-confidence wrong rows (probability > 0.85)** to avoid poison pills. All emitted rows get default weight 1.0 — no per-sample weighting (SFTTrainer doesn't support it).
- Ex-ante decision moments; strictly-past news windows. Per-dataset selectors (§4 v2).
- **`enable_thinking=False` LOCKED** in every chat-template render path (training + inference).
- Merged LoRA at deploy (no runtime `--lora`).
- Apply chat template at dataset render; never hand-write ChatML.
- JSON grammar constraint at inference (don't spend LoRA rank on schema).
- Pascal benchmark MANDATORY before training run.
- Baseline-without-FT MANDATORY before training run.
- Loader API decided by Phase 0 (multimodal Qwen3.5-9B may need `FastModel`, not `FastLanguageModel`).

## 13. Open questions still requiring decision (for v1 lock review)

- **Hindsight teacher** (gemini's suggestion): for the ~30% of markets where the committee was wrong, should we generate a SECOND teacher label using a frontier model + actual outcome ("write the inevitable rationale")? This adds a second teacher pipeline but provides explicit calibration for hard cases. **Default: skip in v0, add in v1 if calibration is the weak metric.**
- **JSON grammar choice**: dynamic strict GBNF from JSON schema vs hand-tuned grammar that prioritizes key fields? **Default: auto-generate from schema; iterate if too restrictive.**
- **Multi-turn?** Trading agent may issue follow-up queries. **Default: single-turn v0. Multi-turn is v2.**

---

**Next gate:** SPEC v1 cross-model review (gemini + codex). After consensus PROCEED, begin Phase 0 (Pascal benchmark).
codex
BLOCKER.

Remaining textual inconsistencies:

1. **§7 still has a copy-paste footgun.** The final `Format` row says:
   `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)`
   and omits `enable_thinking=False`, directly contradicting the locked `Chat template` row and §12. Delete the duplicate row or add the kwarg there too.

2. **Phase 0 does not actually verify the training loader API.** §0a/§7/§12 say Phase 0 decides `FastModel` vs `FastLanguageModel` vs `AutoModelForImageTextToText`, but Phase 0’s concrete gate only benchmarks the GGUF on P40. The dummy LoRA/loader path is “optionally” tested “later,” which means training can start before the multimodal Qwen3.5 loader path is proven. Make loader smoke-test mandatory in Phase 0: load 4-bit, attach rank-8 LoRA, run one forward/backward or tiny SFT step, save/merge if feasible.

Fix those two and I’d vote PROCEED.
tokens used
11,221
BLOCKER.

Remaining textual inconsistencies:

1. **§7 still has a copy-paste footgun.** The final `Format` row says:
   `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)`
   and omits `enable_thinking=False`, directly contradicting the locked `Chat template` row and §12. Delete the duplicate row or add the kwarg there too.

2. **Phase 0 does not actually verify the training loader API.** §0a/§7/§12 say Phase 0 decides `FastModel` vs `FastLanguageModel` vs `AutoModelForImageTextToText`, but Phase 0’s concrete gate only benchmarks the GGUF on P40. The dummy LoRA/loader path is “optionally” tested “later,” which means training can start before the multimodal Qwen3.5 loader path is proven. Make loader smoke-test mandatory in Phase 0: load 4-bit, attach rank-8 LoRA, run one forward/backward or tiny SFT step, save/merge if feasible.

Fix those two and I’d vote PROCEED.
