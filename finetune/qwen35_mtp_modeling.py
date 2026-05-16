"""Qwen3.5/3.6 MTP (Multi-Token Prediction) head — PyTorch implementation.

Reverse-engineered from llama.cpp Sakatard/llama-cpp-turboquant @ c85252627
file `src/models/qwen35.cpp:487-626` (graph_mtp). Upstream transformers ships ZERO
MTP code — only ignore-patterns dropping mtp.* tensors at load. This module
restores MTP as a trainable PyTorch nn.Module.

Training loss (per DeepSeek-V3 §2.2, single-D case):
    main_loss = CE(trunk_logits, labels[1:])
    mtp_loss  = CE(mtp_logits,  labels[2:])
    total     = main_loss + ALPHA * mtp_loss

Cross-model review fixes (round 2):
  - Do NOT store shared embed_tokens/lm_head as module attrs (nn.Module __setattr__
    registers them as children, contaminating state_dict). Pass via forward args.
  - Use forward_pre_hook on existing norm instead of replacing it (replacement
    unregisters norm.weight from state_dict and merged save).
  - Patch model.forward AFTER get_peft_model (PEFT wraps/replaces forward).
  - RoPE positions: use position_ids[:, 1:] not [:, :-1] (MTP input is x_{t+1}).
  - Explicit causal mask for MTP (eager attention path doesn't infer causality).
  - Whitelist exactly the 15 expected mtp keys at injection.
"""
from __future__ import annotations

import json
import os
import types
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

ALPHA = float(os.environ.get("PHASE0F_MTP_ALPHA", "0.3"))

# Expected 15 mtp.* tensor keys per Unsloth/Qwen3.6-27B safetensors index
EXPECTED_MTP_KEYS = {
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
}
assert len(EXPECTED_MTP_KEYS) == 15


def _find_full_attention_layer_idx(config) -> int:
    tc = getattr(config, "text_config", config)
    for i, t in enumerate(tc.layer_types):
        if t == "full_attention":
            return i
    raise RuntimeError("No full_attention layer in text_config.layer_types")


class Qwen3_5MTPBlock(nn.Module):
    """MTP block: 4 norms + 1 projection + 1 Qwen3_5DecoderLayer (full_attention).

    Shared embed_tokens and lm_head are NOT stored as module attrs (would pollute
    state_dict). They are passed as forward args by the wrapper.
    """

    def __init__(self, config):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer, Qwen3_5RMSNorm,
        )
        tc = getattr(config, "text_config", config)
        hidden = tc.hidden_size
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(hidden, eps=tc.rms_norm_eps)
        self.pre_fc_norm_hidden    = Qwen3_5RMSNorm(hidden, eps=tc.rms_norm_eps)
        self.fc   = nn.Linear(2 * hidden, hidden, bias=False)
        self.norm = Qwen3_5RMSNorm(hidden, eps=tc.rms_norm_eps)
        full_idx = _find_full_attention_layer_idx(config)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(tc, full_idx)])

    def forward(
        self,
        prev_hidden: torch.Tensor,                                # [B, T-1, H]  pre-norm trunk hidden at pos 0..T-2
        next_tokens: torch.Tensor,                                 # [B, T-1]    token ids at pos 1..T-1
        position_embeddings: tuple[torch.Tensor, torch.Tensor],    # rope cos/sin sliced to pos 1..T-1
        shared_embed_tokens: nn.Embedding,
        shared_lm_head: nn.Linear,
        attention_mask: Optional[torch.Tensor] = None,             # causal mask for T-1 length
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tok_embd = shared_embed_tokens(next_tokens)                # [B, T-1, H]
        h_norm = self.pre_fc_norm_hidden(prev_hidden)
        e_norm = self.pre_fc_norm_embedding(tok_embd)
        concat = torch.cat([e_norm, h_norm], dim=-1)               # [B, T-1, 2H]
        inpSA  = self.fc(concat)                                   # [B, T-1, H]
        cur = self.layers[0](
            inpSA,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        cur = self.norm(cur)
        return shared_lm_head(cur)


def _load_mtp_state_from_safetensors(repo_cache_dir: Path, mtp_module: Qwen3_5MTPBlock) -> int:
    idx_path = repo_cache_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        cands = list(repo_cache_dir.rglob("model.safetensors.index.json"))
        if not cands:
            raise FileNotFoundError(f"safetensors index not found under {repo_cache_dir}")
        idx_path = cands[0]
    idx = json.loads(idx_path.read_text())
    weight_map = idx["weight_map"]

    by_shard: dict[str, list[str]] = {}
    for k, shard in weight_map.items():
        if k.startswith("mtp."):
            by_shard.setdefault(shard, []).append(k)

    mtp_sd: dict[str, torch.Tensor] = {}
    shard_root = idx_path.parent
    for shard_name, keys in by_shard.items():
        shard_path = shard_root / shard_name
        if not shard_path.exists():
            cands = list(shard_root.parent.rglob(shard_name))
            if not cands:
                raise FileNotFoundError(f"shard not found: {shard_name}")
            shard_path = cands[0]
        with safe_open(str(shard_path), framework="pt") as f:
            for k in keys:
                rel = k[len("mtp."):]
                mtp_sd[rel] = f.get_tensor(k)

    missing, unexpected = mtp_module.load_state_dict(mtp_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected MTP keys: {unexpected}")
    # missing keys allowed — Qwen3_5DecoderLayer has its own params we don't preload
    # (those come from safetensors via mtp.layers.0.* keys, all 9 of them)
    return len(mtp_sd)


def _resolve_hf_cache_dir(repo: str) -> Path:
    cache_root = Path(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"))
    repo_dirname = "models--" + repo.replace("/", "--")
    repo_dir = cache_root / "hub" / repo_dirname / "snapshots"
    if not repo_dir.exists():
        repo_dir = cache_root / "hub" / repo_dirname
    snapshots = sorted([p for p in repo_dir.glob("*") if p.is_dir()])
    if not snapshots:
        raise FileNotFoundError(f"no snapshots under {repo_dir}")
    return snapshots[-1]


def attach_mtp_head(model, base_repo: str, dtype: torch.dtype = torch.bfloat16) -> tuple[Qwen3_5MTPBlock, int]:
    """Instantiate MTP block, load weights from HF cache, attach as model.mtp.

    Returns:
        (mtp_module, n_keys_loaded). Caller must verify n_keys_loaded == 15.
    """
    mtp = Qwen3_5MTPBlock(model.config).to(dtype=dtype)
    cache_dir = _resolve_hf_cache_dir(base_repo)
    n_loaded = _load_mtp_state_from_safetensors(cache_dir, mtp)
    print(f"[MTP] loaded {n_loaded} mtp.* tensors from {cache_dir}")

    target_device = next(model.parameters()).device
    mtp = mtp.to(device=target_device)
    model.mtp = mtp
    return mtp, n_loaded


def patch_forward_with_mtp_loss(model, alpha: float = ALPHA) -> None:
    """Add LM+α·MTP combined loss to model.forward.

    Implementation:
      - forward_pre_hook on language_model.norm to capture pre-norm hidden state
        (does NOT replace the norm module — preserves norm.weight in state_dict)
      - Wrap model.forward to compute MTP loss after main forward
      - Pass shared embed_tokens + lm_head as forward args to MTP (avoids
        registering them as module attrs which would pollute state_dict)

    Call AFTER get_peft_model() since PEFT may replace forward.
    """
    # Resolve the wrapped model (could be PeftModel)
    base = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    text_model = base.model.language_model
    embed_tokens = text_model.embed_tokens
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.language_model.lm_head

    captured: dict = {}

    def _capture_pre_norm(_module, inputs):
        captured["pre_norm"] = inputs[0]
        # Return None → input passes through unchanged to real norm
        return None

    # Register hook on real norm (does not replace the module → norm.weight stays registered)
    handle = text_model.norm.register_forward_pre_hook(_capture_pre_norm)
    model._mtp_norm_hook_handle = handle  # keep ref so it isn't garbage-collected

    # Get reference to whatever forward is current (post-PEFT-wrap)
    original_forward = model.forward

    def _build_causal_mask(seqlen: int, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Standard causal mask: 0 for attend, -inf for masked."""
        mask = torch.full((seqlen, seqlen), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, seqlen, seqlen)

    def patched_forward(self, *args, **kwargs):
        labels = kwargs.pop("labels", None)
        try:
            out = original_forward(*args, labels=labels, **kwargs)
            if labels is None:
                return out

            pre_norm = captured.get("pre_norm")
            if pre_norm is None:
                return out

            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is None or input_ids.shape[1] < 3:
                return out

            B, T = input_ids.shape
            device = input_ids.device

            # Position ids for MTP attention: positions 1..T-1 (MTP input is x_{t+1})
            position_ids = kwargs.get("position_ids")
            if position_ids is None:
                position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            mtp_pos_ids = position_ids[:, 1:].contiguous()        # [B, T-1]

            # RoPE for positions 1..T-1
            rotary = text_model.rotary_emb
            # Provide a dummy hidden of right shape so rotary_emb infers seq dim correctly.
            # rotary_emb usually only needs position_ids; pass a small placeholder.
            pos_embeds = rotary(pre_norm, mtp_pos_ids if mtp_pos_ids.ndim >= 2 else mtp_pos_ids.unsqueeze(0))
            # pos_embeds may be (cos, sin) tuple or single tensor
            if isinstance(pos_embeds, tuple):
                cos, sin = pos_embeds
            else:
                cos, sin = pos_embeds[0], pos_embeds[1]
            # cos/sin shape: [B, T-1, rotary_dim]. Slice explicitly on seq dim.
            # If rotary returned full [B, T, ...] (some impls), slice explicitly:
            if cos.shape[-2] == T:
                cos = cos[:, 1:, :].contiguous()
                sin = sin[:, 1:, :].contiguous()

            # MTP input
            prev_hidden = pre_norm[:, :-1, :].contiguous()        # [B, T-1, H] hidden at pos 0..T-2
            next_tokens = input_ids[:, 1:].contiguous()           # [B, T-1]   tokens at pos 1..T-1

            # Explicit causal mask for T-1 (eager attention safety)
            attn_mask = _build_causal_mask(T - 1, B, device, pre_norm.dtype)

            mtp_logits = base.mtp(
                prev_hidden=prev_hidden,
                next_tokens=next_tokens,
                position_embeddings=(cos, sin),
                shared_embed_tokens=embed_tokens,
                shared_lm_head=lm_head,
                attention_mask=attn_mask,
                position_ids=mtp_pos_ids,
            )

            # MTP loss: predict token[t+2] for t in 0..T-3 → mtp_logits[:, :-1] against labels[:, 2:]
            # Compute loss in CHUNKS to avoid materializing full [B*T, V] fp32 tensor (~2GB at V=248k).
            if mtp_logits.shape[1] < 2:
                return out
            mtp_logits_for_loss = mtp_logits[:, :-1, :]            # [B, T-2, V]
            mtp_target = labels[:, 2:]                             # [B, T-2]
            B_, T_minus_2, V_ = mtp_logits_for_loss.shape
            chunk = 256
            total_ce = mtp_logits_for_loss.new_zeros((), dtype=torch.float32)
            total_n = 0
            for i in range(0, T_minus_2, chunk):
                lg = mtp_logits_for_loss[:, i:i+chunk, :].reshape(-1, V_)
                tg = mtp_target[:, i:i+chunk].reshape(-1)
                mask = tg != -100
                if not mask.any():
                    continue
                ce = F.cross_entropy(lg[mask].float(), tg[mask], reduction="sum")
                total_ce = total_ce + ce
                total_n += int(mask.sum().item())
            mtp_loss = total_ce / max(total_n, 1)

            # Combine
            if out.loss is not None:
                out.lm_loss = out.loss.detach()
                out.loss = out.loss + alpha * mtp_loss
                out.mtp_loss = mtp_loss.detach()
            return out
        finally:
            captured.clear()   # always clear to prevent memory pinning

    model.forward = types.MethodType(patched_forward, model)


def lora_target_modules_for_mtp_training(model) -> list[str]:
    """Trunk + MTP decoder block Linears for LoRA. Excludes MTP fc/norms (modules_to_save those)."""
    import re
    pat = re.compile(
        r"^(model\.)?model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"
        r"|^(model\.)?mtp\.layers\.0\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"
    )
    return [name for name, mod in model.named_modules() if pat.match(name) and isinstance(mod, nn.Linear)]


def lora_modules_to_save_for_mtp_training() -> list[str]:
    return ["mtp.fc", "mtp.norm", "mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_hidden"]


def save_mtp_to_merged_safetensors(merged_dir: Path, model) -> int:
    """After save_pretrained_merged, ensure all 15 mtp.* tensors are in merged dir.

    Two-phase approach:
      1. SCAN merged dir for existing mtp.* keys (Unsloth may have written them).
      2. If any of the 15 expected keys are missing, extract them from a freshly-merged
         model via peft.merge_and_unload() (strips LoRA wrappers + modules_to_save
         wrappers, returning clean base Linear/RMSNorm modules with original key paths).

    Returns total count of mtp tensors in merged dir (must equal 15 to pass gate).
    """
    from safetensors.torch import save_file

    # Phase 1: scan merged dir
    existing_mtp_in_dir: set[str] = set()
    shard_paths: dict[str, Path] = {}
    idx_path = merged_dir / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        for k, shard in idx.get("weight_map", {}).items():
            if k.startswith("mtp."):
                existing_mtp_in_dir.add(k)
                shard_paths[k] = merged_dir / shard
    else:
        # No index → scan single safetensors
        try:
            for sf in merged_dir.glob("*.safetensors"):
                with safe_open(str(sf), framework="pt") as f:
                    for k in f.keys():
                        if k.startswith("mtp."):
                            existing_mtp_in_dir.add(k)
                            shard_paths[k] = sf
        except Exception as e:
            print(f"[mtp-inject] scan failed: {e}")

    missing = EXPECTED_MTP_KEYS - existing_mtp_in_dir
    extras_in_dir = existing_mtp_in_dir - EXPECTED_MTP_KEYS
    print(f"[mtp-inject] merged dir scan: {len(existing_mtp_in_dir)} mtp.* present, {len(missing)} missing, {len(extras_in_dir)} extras")
    if extras_in_dir:
        print(f"[mtp-inject] extras (unexpected keys): {sorted(extras_in_dir)[:5]}")

    if not missing:
        print(f"[mtp-inject] all 15 mtp tensors already in merged dir — no injection needed")
        return 15

    # Phase 2: get clean MTP state via merge_and_unload
    print(f"[mtp-inject] need to inject {len(missing)} keys: {sorted(missing)[:5]}{'...' if len(missing)>5 else ''}")
    base_model = None
    if hasattr(model, "merge_and_unload"):
        try:
            base_model = model.merge_and_unload(progressbar=False)
            print(f"[mtp-inject] merge_and_unload OK, type: {type(base_model).__name__}")
        except Exception as e:
            print(f"[mtp-inject] merge_and_unload failed: {e}; falling back to wrapped model")
    if base_model is None:
        base_model = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model

    mtp = base_model.mtp if hasattr(base_model, "mtp") else None
    if mtp is None:
        raise RuntimeError("Cannot locate model.mtp after merge_and_unload — MTP injection cannot proceed")

    # Walk mtp.state_dict(), filter to EXPECTED keys with "mtp." prefix
    mtp_sd: dict[str, torch.Tensor] = {}
    for k, v in mtp.state_dict().items():
        full_key = f"mtp.{k}"
        if full_key in EXPECTED_MTP_KEYS and full_key in missing:
            mtp_sd[full_key] = v.detach().to(torch.bfloat16).contiguous().cpu()

    # If walking the standard state_dict still doesn't yield everything, try direct submodule walk
    still_missing = missing - set(mtp_sd.keys())
    if still_missing:
        print(f"[mtp-inject] state_dict walk got {len(mtp_sd)}/{len(missing)}; trying direct submodule walk for {len(still_missing)} remaining")
        for name, submod in mtp.named_modules():
            full = f"mtp.{name}.weight" if name else "mtp.weight"
            if full in still_missing and hasattr(submod, "weight") and submod.weight is not None:
                w = submod.weight
                # Handle PEFT base_layer
                if hasattr(submod, "base_layer") and hasattr(submod.base_layer, "weight"):
                    w = submod.base_layer.weight
                mtp_sd[full] = w.detach().to(torch.bfloat16).contiguous().cpu()

    still_missing = missing - set(mtp_sd.keys())
    if still_missing:
        raise RuntimeError(f"Cannot extract MTP weights for keys: {still_missing}")

    print(f"[mtp-inject] extracted {len(mtp_sd)} missing keys, writing new shard")
    out_path = merged_dir / "model-mtp.safetensors"
    save_file(mtp_sd, str(out_path))

    # Update index
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        wm = idx.get("weight_map", {})
        for k in mtp_sd:
            wm[k] = "model-mtp.safetensors"
        idx["weight_map"] = wm
        total = idx.get("metadata", {}).get("total_size", 0)
        for t in mtp_sd.values():
            total += t.numel() * t.element_size()
        idx.setdefault("metadata", {})["total_size"] = total
        idx_path.write_text(json.dumps(idx, indent=2))

    final_count = len(existing_mtp_in_dir) + len(mtp_sd)
    return final_count
