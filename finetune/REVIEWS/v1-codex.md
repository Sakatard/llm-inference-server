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
session id: 019e2c9e-59f2-7493-be79-6beeb10bf416
--------
user
Review this fine-tuning spec critically. Be blunt. You are reviewing a fine-tuning specification for a real production-bound LLM project. Be blunt, no glazing. Find architectural mistakes, scope creep, and bad assumptions. Specifically attack:

1. The "reuse existing polymarket-agents committee pipeline as the teacher" decision — is there a cleaner path? Are we creating a dependency nightmare?
2. The 50→500→2000 volume curve — too aggressive? not enough? Should pilot be bigger?
3. The schema reuse — `RefereeOutput` clone with optional `crypto_extension`. Should `crypto_extension` be a separate schema/model entirely?
4. The Pascal-deploy assumption — Qwen3.5-7B + LoRA on sm_61 P40 with `--lora` flag at runtime. Will llama.cpp's LoRA path break on Pascal? Any sharp edges?
5. The "discard rows where committee decision contradicts actual outcome" filter — does this bias the dataset toward easy markets? Should we keep some hard cases?
6. The Unsloth Pascal incompatibility — fine, we train on rented Ampere. But: are there bnb-4bit / FA2-baked LoRA adapter artifacts that would fail at inference on Pascal? Any LoRA-format mismatches?
7. Is rank-32 LoRA right for a 500-example dataset, or way overkill? Should we start lower?
8. The "ChatML JSONL" format — Qwen3.5 uses its own chat template (`<|im_start|>system\n...<|im_end|>`). Are we sure ChatML-Qwen vs OpenAI-ChatML format won't cause silent tokenization mismatch? This is a known footgun.
9. Volume-weighted midpoint as decision moment — does this leak information? Markets that traded high volume at midpoint may correlate with resolved outcome (insider info, news leakage). Pick differently?
10. Anything else stupid we haven't seen.

The spec is below. Read it carefully. Return blunt findings + alternative recommendations under each numbered question, plus a final "blocker / proceed" verdict. Under 800 words.

---

# Polymarket / Crypto Trade-Decision Fine-Tune — Specification v0

**Status:** draft, pending cross-model review (gemini + codex per CLAUDE.md gate 2)
**Owner:** inference-server-engineer (project agent)
**Target model:** `Qwen3.5-7B-Instruct` (Unsloth 4-bit base for QLoRA)
**Deploy target:** P40 (24 GiB, Pascal sm_61) as new `qwen-trader` service on llm-inference-server
**Schema:** [`finetune/schemas/trade_decision.schema.json`](./schemas/trade_decision.schema.json) — mirrors polymarket-agents `RefereeOutput` (committee/contracts.py)

---

## 1. Why this size, why this model

| Decision | Choice | Reason |
|---|---|---|
| Base model size | 7B (not 27B) | Per "start small" advice — 5-10× faster iteration on data + LoRA config. 7B QLoRA fits on a $0.30/hr rented 4090, full run in ~30-90 min. |
| Family | Qwen3.5-7B-Instruct | Same tokenizer family as our prod Qwen3.6-27B. Smooth Unsloth notebook support. Future option to scale to Qwen3.6-27B with minimal data changes (Qwen3.6 has no 7B variant). |
| Instruction tuning | Already instruct-tuned base | Don't waste data teaching basic chat — focus on the schema-conformance + reasoning-quality task. |
| 4-bit base | bnb-4bit Unsloth variant | Cuts training VRAM ~4×. Output adapter merged back to fp16 then quantized to Q4_K_M for deployment. |

We do NOT fine-tune Qwen3.6-27B in v0. After 7B pipeline proves itself, we may scale up. Skipping the scale-up entirely is also a valid outcome — a purpose-built 7B can beat a generic 27B on a narrow task.

## 2. Teacher labels — reuse polymarket-agents committee

**Critical decision: do NOT build a new teacher pipeline.** The polymarket-agents project at `~/containers/polymarket-agents` already has a fully-functional committee labeller emitting `RefereeOutput` JSON.

Existing committee pipeline (`agents/research/committee/*`):
- `yes_thesis.py` — DeepSeek v3.2 (OpenRouter), bull-case thesis
- `no_thesis.py` — DeepSeek v3.2 (OpenRouter), bear-case thesis
- `microstructure.py` — deterministic Python orderbook math
- `referee.py` — **gpt-5.5 xhigh reasoning** synthesizes everything into `RefereeOutput`

The referee output IS our fine-tune target. Compressing this multi-LLM-call ensemble into one forward pass on a 7B model.

### Dataset construction flow (one-pass)

```
For each historical resolved market m:
    1. Reconstruct the "decision moment" context for m:
       - Market metadata (question, rules, parsed_rules)
       - Snapshot orderbook + price at chosen decision time
       - News articles from a ±3 day window around decision time
         (GDELT relay + Google News RSS — reuse polymarket-agents connectors verbatim)
       - Microstructure report (deterministic, computable from snapshot)
    2. Run the existing committee on m's reconstructed context:
       - Get RefereeOutput (this is the "teacher label")
    3. Cross-check committee's argmax(probability) vs m's actual resolved outcome
       - PASS → keep example
       - FAIL → discard (don't teach the model decisions that turned out wrong)
    4. Emit ChatML JSONL row:
       - system: trader system prompt (locked, version-pinned)
       - user: serialized context (event + rules + news + market state)
       - assistant: RefereeOutput JSON (the teacher label)
```

This re-uses **every existing connector, rate-limiter, cache, and prompt** in polymarket-agents. We are not building a new news pipeline. We are not building a new LLM client. We are not building a new schema.

**What we ARE building:**
- A historical-market enumerator (extends `scripts/btc15m_backtest/market_enum.py` patterns)
- A "rewind context" function that reconstructs what the committee would have seen at decision time
- A JSONL formatter
- A quality-filter pass

## 3. Decision-moment selection

For each resolved market, we need ONE canonical decision moment. Candidates:
- (A) Market open + 24h (early signal, before crowd forms)
- (B) Market midpoint by volume (typical decision time for active traders)
- (C) 48h before resolution (close-to-truth, but easier to predict)

**Choice: (B) volume-weighted midpoint.** Closest to real production usage where the bot decides while there's still tradeable liquidity. Avoids the trivial "predict on the last day" case.

## 4. Volume target

| Phase | Target |
|---|---|
| Pilot | 50 examples — sanity check pipeline end-to-end, no training run |
| v0 train | 500 examples — first usable LoRA, expect modest quality |
| v1 train | 2,000 examples — quality plateau target |
| v2 (optional) | 5,000+ — only if v1 evals show ROI for more data |

500 high-quality examples is enough for a 7B QLoRA to learn schema-conformance + style. The reasoning capability is mostly in the base model — we're shaping its output, not teaching reasoning from scratch.

## 5. Scope — which historical markets

Pull resolved markets from `gamma-api.polymarket.com/markets?closed=true` (same endpoint as `scripts/btc15m_backtest/market_enum.py`).

Filters:
- **Resolved with a definitive winner** (not ambiguous / refunded)
- **Volume > $10k USDC** (skip dead markets — no real liquidity = noisy training signal)
- **Resolution date ≥ 2025-01-01** (recent enough that GDELT + Google RSS have good coverage)
- **Categories**: politics, sports, crypto, current-events. Skip novelty markets where rules are weird (celebrity, meme).
- **Diversity quota**: ≤30% from any single category in v0 (avoids overfitting one domain).

Sampling: random within filter, weighted by `volume * recency`. ~500-2000 markets total.

## 6. Eval harness

Hold-out 50 markets from the pilot-50 set. Never see during training.

Three metrics on hold-out:
1. **Schema validity** — does the model emit valid JSON parseable by the Pydantic validator? Target: ≥98%.
2. **Outcome accuracy** — does `argmax(probability)` match the resolved outcome? Target: ≥ committee's own accuracy on this set (we're cloning the committee, not improving it).
3. **Calibration delta** — Brier score vs committee's Brier on same set. Target: within 10% of committee.

Schema validity is the killer metric. If the model can't emit clean JSON we have nothing.

## 7. Training stack

| Layer | Choice |
|---|---|
| Framework | Unsloth (PEFT + custom FA2 fork, Pascal-incompatible — runs on rented GPU only) |
| Method | QLoRA, rank 32, alpha 64, dropout 0.05 |
| Target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` (no MTP — Qwen3.5 has no MTP heads, simplifies v0) |
| Optimizer | adamw_8bit, lr 2e-4, cosine schedule, warmup 0.03 |
| Sequence length | 8,192 (context is large — news articles + rules + microstructure) |
| Batch size | gradient_accumulation=8, per_device=2 (effective 16) |
| Epochs | 3 — overfit risk at 500 examples is real, monitor train+eval loss |
| Hardware | Vast.ai RTX 4090 (24 GiB) — ~$0.30/hr, ~2 hr per run |

## 8. Deployment back to P40

After train + eval pass:
1. `python3 llama.cpp/convert_lora_to_gguf.py <adapter_dir> --outfile qwen-trader-v0.gguf`
2. Add new service to `server.py`:
   ```python
   qwen_trader = Service("qwen-trader", [
       "llama-server",
       "-m", "/models/Qwen3.5-7B-Q4_K_M.gguf",
       "--lora", "/models/qwen-trader-v0.gguf",
       "-c", "16384",                  # smaller ctx, agents don't need 128k
       "--cache-type-k", "q8_0",
       "--cache-type-v", "q8_0",
       "-ngl", "99",
       "--host", "127.0.0.1",
       "--port", "9186",
       "--no-mmap",
       "--no-webui",
       "--jinja",
   ], port=9186)
   ```
3. New route `/v1/trade` → qwen-trader.
4. Expected VRAM: ~5-6 GiB. Fits cleanly alongside the 27B (3 GiB headroom remains for one more concurrent service).
5. Expected throughput: ~80-150 tok/s decode (smaller model + smaller ctx).

## 9. What we do NOT do in v0

- No DPO / GRPO / PPO. SFT only. RL is v2.
- No multi-modal (no charts, no audio).
- No retrieval-time embedding. The committee context already includes news article bodies — embed-and-retrieve is wasted for our scope.
- No tool-use fine-tuning. The model outputs JSON, our adapter consumes JSON. No function calling.
- No on-the-fly news fetching by the FT model. News context is baked into the prompt by the caller (same shape as committee input).
- No fine-tuning the MTP heads (Qwen3.5 doesn't have them; on Qwen3.6 it's a v2 problem).

## 10. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Teacher (committee) is wrong on training data | Cross-check vs actual outcome → discard wrong-decision rows. Step 3 of dataset flow. |
| Committee labels are too verbose for 7B to learn | Cap `rationale` field length during dataset prep. Truncate teacher's rationale to first 1500 chars if longer. |
| 7B can't match 27B's reasoning depth | Accept it. 7B FT'd on narrow task can match generic 27B *on that task*. If not, fall back to scaling to Qwen3.6-27B on same pipeline. |
| Polymarket connectors hit rate limits during dataset build | Reuse existing GDELT + Google RSS rate limiters (Postgres-backed circuit breakers). Run dataset build overnight. |
| Article-extraction failures (paywall, bot wall) | Existing `extract.py` has anti-bot filter. Drop rows where article coverage < threshold. |
| Adapter file too large for runtime swap | Rank 32 → ~250 MB. Acceptable. If >500 MB, drop rank to 16. |
| Schema drift between FT and consumer | Version-pin: `TradeDecisionV0`. v1 schema lives in a new file. Consumer code branches on schema version field if needed. |

## 11. Cost estimate

| Item | Cost |
|---|---|
| OpenRouter LLM calls during dataset build (2000 markets × 3 committee calls × ~3k tokens each) | ~$30-60 |
| GDELT + Google RSS | $0 (reuse existing infra) |
| Vast.ai 4090 — 3-5 training runs × 2 hr | ~$3-6 |
| **Total v0 + v1** | **~$40-100** |

Time estimate: 1-2 weeks elapsed, ~2-4 days of focused work.

## 12. Open questions (decide before locking spec)

- [ ] Is reusing polymarket-agents committee as teacher acceptable given the v2 plan is RL (DPO)? RL needs preference pairs which the committee provides naturally (yes_thesis vs no_thesis disagreement). Yes, fine.
- [ ] System prompt content — locked to v0 or per-domain (polymarket vs crypto split)? **Decision: one system prompt for both, switch behavior via context content not prompt.** Simpler, generalizes better.
- [ ] Hold-out methodology — random split or time-based (train on pre-2025-09, eval on post)? **Decision: time-based.** More realistic, catches drift.
- [ ] Should `crypto_extension` be present in polymarket training rows (null) or absent? **Decision: present-but-null** for schema consistency. Pydantic union types are fragile.

## 13. Open-loop: cross-model review

Per CLAUDE.md gate 2 (spec review before commit), this spec MUST be reviewed by gemini + codex in parallel before we cut any code. Specifically asking them to attack:

1. The "reuse committee as teacher" assumption — is there a cleaner path?
2. The 500→2000 volume curve — too aggressive? not enough?
3. The schema reuse decision — should `crypto_extension` be a separate schema instead?
4. The Pascal-deploy assumption — any reason a Qwen3.5-7B + LoRA would misbehave on sm_61?
5. The "discard wrong-decision rows" filter — does this bias the dataset toward "easy" markets?

After review + revisions, this spec becomes the locked source-of-truth. Implementation begins from here.

---

**Files written so far:**
- `finetune/SPEC.md` (this file)
- `finetune/schemas/trade_decision.schema.json` (v0 schema)
- `finetune/examples/polymarket_election.json` (reference example)
- `finetune/examples/crypto_btc.json` (reference example)

**Next step before code:** cross-model review (gemini + codex parallel) → revise spec → lock → begin implementation.
web search: 
web search: llama.cpp LoRA GGUF --lora convert_lora_to_gguf qwen LoRA adapter Pascal sm_61 CUDA support
web search: 
web search: site:github.com/ggml-org/llama.cpp CUDA compute capability Pascal sm_61 support LoRA
web search: 
web search: https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/quickstart.md
web search: 
web search: Qwen3.5 official chat template Qwen3.5-7B-Instruct Hugging Face
web search: 
web search: https://huggingface.co/Qwen/Qwen3.5-9B
web search: 
web search: Qwen3.5-7B-Instruct Hugging Face
codex
I checked current Qwen, llama.cpp, and Unsloth docs. Blunt version:

1. **Reuse committee as teacher**
Finding: Reusing the whole `polymarket-agents` committee is a coupling trap: prompts, vendors, caches, rate limits, schemas, and historical reconstruction all become hidden training dependencies.  
Recommendation: Extract a frozen `teacher-runner` contract: versioned context bundle in, versioned label out. Reuse connectors only behind stable APIs. Do not make the fine-tune pipeline import the live agent stack.

2. **50→500→2000 volume curve**
Finding: 50 “no training run” proves almost nothing. 500 may teach schema/style, not robust trading judgment.  
Recommendation: Do a 200-300 example smoke train first, then learning curves at 250/500/1000/2000. Holdout should be at least 200 and time-based/category-stratified.

3. **Schema reuse / `crypto_extension`**
Finding: Cloning `RefereeOutput` plus nullable `crypto_extension` is schema debt. You are leaking teacher internals into product API design.  
Recommendation: Define `TradeDecisionV0` as the product contract. Use a stable core plus a discriminated domain payload, or separate `PolymarketDecision` / `CryptoDecision`. Nullable extension fields teach null-spam.

4. **Pascal deploy**
Finding: This is not proven. llama.cpp supports GGUF/LoRA generally, but P40/Pascal is a sharp edge. Pascal lacks modern tensor cores; dynamic LoRA can add slow fp16-ish adapter work; flash-attention fallback has current Pascal failure reports at long KV depths. Your 80-150 tok/s estimate is probably fantasy until benchmarked.  
Recommendation: Prefer merge LoRA into fp16 base, then quantize final GGUF, instead of runtime `--lora`, unless hot-swapping matters. Before locking spec, run base + dummy LoRA on the actual P40 at 16k ctx.

5. **Discard teacher-wrong rows**
Finding: Bad filter. It biases toward easy markets, destroys calibration, and uses outcome as a selection oracle. A 45/55 teacher “wrong” call is not bad data.  
Recommendation: Keep hard cases. Drop only broken/ambiguous rows. Downweight high-confidence wrong teacher outputs or send them to audit. Evaluate on unfiltered realistic markets.

6. **Unsloth artifacts**
Finding: Training on Ampere is fine; adapters are not hardware-bound. The risks are export-format risks: bnb merged artifacts, added tokens, embedding/lm_head adapters, tokenizer/template drift, or converter/runtime version mismatch.  
Recommendation: Save clean PEFT safetensors, no new tokens, no embedding/lm_head LoRA. Convert with the same llama.cpp commit used in prod. Better: merge to fp16, convert, quantize.

7. **Rank-32 LoRA**
Finding: Rank 32 over all linear modules for 500 examples is likely overkill and memorization-prone.  
Recommendation: Start r=8 or r=16, alpha=2r, dropout 0.05-0.1. Run r=8/16/32 only after the smoke train. Use JSON grammar for schema validity instead of spending rank on formatting.

8. **ChatML JSONL**
Finding: This is a known footgun. “ChatML JSONL” is not enough. Qwen docs explicitly require model-specific chat templates via `apply_chat_template`; Qwen3.5 also has thinking/non-thinking template behavior.  
Recommendation: Render training rows with the exact tokenizer/template from the exact checkpoint. Store rendered prompt hashes. Serve with the same llama.cpp `--jinja` template or explicit template file. Do not hand-roll OpenAI-style ChatML.

9. **Volume-weighted midpoint**
Finding: Leaky selection. If midpoint is computed from final lifetime volume, you used future market activity to choose the decision time. Also high-volume midpoint often occurs after news/insider information.  
Recommendation: Use ex-ante timestamps: open+24h, first liquidity threshold crossing, or random active-time samples before resolution. Use only news available before decision time. The `±3 day` news window is also future leakage.

10. **Other stupid bits**
Finding: `Qwen3.5-7B-Instruct` appears not to exist publicly; current Qwen3.5 public sizes include 4B, 9B, 27B, etc. That alone can invalidate the stack. Also Gamma closed-market data probably won’t reconstruct historical orderbooks/news without separate archival data.  
Recommendation: Pick a real target, likely Qwen3.5-9B or Qwen2.5-7B, and prove data reconstruction before training. Add a baseline: base model + prompt + JSON grammar. Fine-tuning may not be needed for schema validity.

**Verdict: Blocker.** Do not implement this spec as written. Fix model identity, data leakage, teacher isolation, schema contract, and P40 deployment proof first.

Sources checked: Qwen chat-template docs, Qwen3.5 model card, llama.cpp GGUF/LoRA docs, Unsloth GGUF export docs.
tokens used
56,988
I checked current Qwen, llama.cpp, and Unsloth docs. Blunt version:

1. **Reuse committee as teacher**
Finding: Reusing the whole `polymarket-agents` committee is a coupling trap: prompts, vendors, caches, rate limits, schemas, and historical reconstruction all become hidden training dependencies.  
Recommendation: Extract a frozen `teacher-runner` contract: versioned context bundle in, versioned label out. Reuse connectors only behind stable APIs. Do not make the fine-tune pipeline import the live agent stack.

2. **50→500→2000 volume curve**
Finding: 50 “no training run” proves almost nothing. 500 may teach schema/style, not robust trading judgment.  
Recommendation: Do a 200-300 example smoke train first, then learning curves at 250/500/1000/2000. Holdout should be at least 200 and time-based/category-stratified.

3. **Schema reuse / `crypto_extension`**
Finding: Cloning `RefereeOutput` plus nullable `crypto_extension` is schema debt. You are leaking teacher internals into product API design.  
Recommendation: Define `TradeDecisionV0` as the product contract. Use a stable core plus a discriminated domain payload, or separate `PolymarketDecision` / `CryptoDecision`. Nullable extension fields teach null-spam.

4. **Pascal deploy**
Finding: This is not proven. llama.cpp supports GGUF/LoRA generally, but P40/Pascal is a sharp edge. Pascal lacks modern tensor cores; dynamic LoRA can add slow fp16-ish adapter work; flash-attention fallback has current Pascal failure reports at long KV depths. Your 80-150 tok/s estimate is probably fantasy until benchmarked.  
Recommendation: Prefer merge LoRA into fp16 base, then quantize final GGUF, instead of runtime `--lora`, unless hot-swapping matters. Before locking spec, run base + dummy LoRA on the actual P40 at 16k ctx.

5. **Discard teacher-wrong rows**
Finding: Bad filter. It biases toward easy markets, destroys calibration, and uses outcome as a selection oracle. A 45/55 teacher “wrong” call is not bad data.  
Recommendation: Keep hard cases. Drop only broken/ambiguous rows. Downweight high-confidence wrong teacher outputs or send them to audit. Evaluate on unfiltered realistic markets.

6. **Unsloth artifacts**
Finding: Training on Ampere is fine; adapters are not hardware-bound. The risks are export-format risks: bnb merged artifacts, added tokens, embedding/lm_head adapters, tokenizer/template drift, or converter/runtime version mismatch.  
Recommendation: Save clean PEFT safetensors, no new tokens, no embedding/lm_head LoRA. Convert with the same llama.cpp commit used in prod. Better: merge to fp16, convert, quantize.

7. **Rank-32 LoRA**
Finding: Rank 32 over all linear modules for 500 examples is likely overkill and memorization-prone.  
Recommendation: Start r=8 or r=16, alpha=2r, dropout 0.05-0.1. Run r=8/16/32 only after the smoke train. Use JSON grammar for schema validity instead of spending rank on formatting.

8. **ChatML JSONL**
Finding: This is a known footgun. “ChatML JSONL” is not enough. Qwen docs explicitly require model-specific chat templates via `apply_chat_template`; Qwen3.5 also has thinking/non-thinking template behavior.  
Recommendation: Render training rows with the exact tokenizer/template from the exact checkpoint. Store rendered prompt hashes. Serve with the same llama.cpp `--jinja` template or explicit template file. Do not hand-roll OpenAI-style ChatML.

9. **Volume-weighted midpoint**
Finding: Leaky selection. If midpoint is computed from final lifetime volume, you used future market activity to choose the decision time. Also high-volume midpoint often occurs after news/insider information.  
Recommendation: Use ex-ante timestamps: open+24h, first liquidity threshold crossing, or random active-time samples before resolution. Use only news available before decision time. The `±3 day` news window is also future leakage.

10. **Other stupid bits**
Finding: `Qwen3.5-7B-Instruct` appears not to exist publicly; current Qwen3.5 public sizes include 4B, 9B, 27B, etc. That alone can invalidate the stack. Also Gamma closed-market data probably won’t reconstruct historical orderbooks/news without separate archival data.  
Recommendation: Pick a real target, likely Qwen3.5-9B or Qwen2.5-7B, and prove data reconstruction before training. Add a baseline: base model + prompt + JSON grammar. Fine-tuning may not be needed for schema validity.

**Verdict: Blocker.** Do not implement this spec as written. Fix model identity, data leakage, teacher isolation, schema contract, and P40 deployment proof first.

Sources checked: Qwen chat-template docs, Qwen3.5 model card, llama.cpp GGUF/LoRA docs, Unsloth GGUF export docs.
