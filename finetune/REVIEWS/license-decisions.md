# License Decisions — Phase -1 Legal Gate

## trentmkelly/polymarket_crypto_derivatives (CC-BY-SA-4.0)

**Decision (2026-05-16, owner approved):** ACCEPT ShareAlike on the qwen-trader LoRA adapter.

**Implications:**
- Dataset USABLE as a training source.
- The trained qwen-trader LoRA adapter inherits CC-BY-SA-4.0 obligations IF it is redistributed.
- For local-only deployment on the llm-inference-server (single P40 host, single user), redistribution doesn't occur, so practical impact is nil.
- If the adapter is ever shared publicly, privately to third parties, uploaded to HuggingFace, or used in a hosted multi-tenant service, it MUST be published under CC-BY-SA-4.0 with attribution to trentmkelly.
- The cryptotrader/polymarket-agents code consuming this adapter is NOT affected (CC-BY-SA applies only to the model weights, not to inference code calling them).

**Mitigation steps required in pipeline:**
- Embed trentmkelly attribution in adapter metadata (GGUF kv field).
- Add LICENSE-ATTRIBUTION.md to the finetune/ directory once training completes.

## SII-WANGZJ/Polymarket_data (MIT)

Clean. No restrictions on adapter.

## Other datasets (puneeth, CK0607, 2084Collective)

License unstated on HF cards. Phase -1 sub-step: confirm each dataset's license before incorporating. Default = reject if license unstated.
