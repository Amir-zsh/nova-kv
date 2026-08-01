"""Memory-capacity model for the throughput benchmark.

Predicts sglang's KV pool size and the largest batch that fits, so cells can be
planned before measuring and the engine's own accounting can be gated against an
independent calculation. That gate matters: a double-count in
``pool_configurator.py`` sized the vq2 pool at 80/112 of its budget for months,
and nothing downstream noticed because no one recomputed the bytes.

Validated on Qwen3-8B (H100, mem_frac 0.85): predicting the KV arenas from these
numbers reproduces the engine's "KV Cache is allocated" line to 0.01 GiB for both
the bf16 arm (52.66 GiB) and the quantized arms (int2 52.66, vq2 38.02).
"""
from __future__ import annotations

import json
from pathlib import Path

GiB = 1024 ** 3

# Unified mixed-KV page geometry: a page holds 1 HP token or N_Q quant tokens.
N_Q = 8                 # = 4 * bfloat16.itemsize
HP_BYTES = 2            # bfloat16
SCALE_BYTES = 4         # float32 (scale, zero) pairs
PREFIX_TOKENS = 64      # SGLANG_MIXED_KV_PREFIX_TOKENS as serve_oscar.sh sets it
RECENT_TOKENS = 256     # SGLANG_MIXED_KV_RECENT_TOKENS as serve_oscar.sh sets it


def model_geometry(model_id: str, hf_home: str | None = None) -> dict:
    """(layers, kv_heads, head_dim, vocab) from the HF config in the local cache."""
    root = Path(hf_home or Path.home() / ".cache/huggingface/hub")
    tag = "models--" + model_id.replace("/", "--")
    cfgs = sorted((root / tag).glob("snapshots/*/config.json"))
    if not cfgs:
        raise FileNotFoundError(f"no cached HF config for {model_id} under {root}")
    c = json.loads(cfgs[-1].read_text())
    head_dim = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    return {
        "model_id": model_id,
        "layers": c["num_hidden_layers"],
        "kv_heads": c["num_key_value_heads"],
        "q_heads": c["num_attention_heads"],
        "head_dim": head_dim,
        "v_head_dim": head_dim,
        "vocab": c["vocab_size"],
        # None on dense models. The engine uses this minus one (sglang's
        # get_attention_sliding_window_size convention); the difference is one
        # token per request, and erring high only makes the SWA bound stricter.
        "sliding_window": c.get("sliding_window"),
    }


def bytes_per_head(arm: str, geom: dict) -> int:
    """Bytes per (quant token, kv head) across K and V, including scales.

    bf16 stores K and V dense. The quantized arms store packed int2 K+V --
    expressed by the engine as a shared page cost ``(D + Dv) * HP_BYTES // N_Q``
    -- plus one (scale, zero) fp32 pair per group for each of K and V.

    vq2 is *identical* to int2 here. Its uint8 VQ index arena replaces the int2
    K arena rather than adding to it, and at G=4 both cost head_dim/4 bytes.
    """
    d, dv = geom["head_dim"], geom["v_head_dim"]
    if arm == "bf16":
        return (d + dv) * HP_BYTES
    k_groups = v_groups = 1                       # --kv-cache-quant-group-size 128
    return (d + dv) * HP_BYTES // N_Q + 2 * SCALE_BYTES * (k_groups + v_groups)


def cell_size(arm: str, geom: dict, tp: int = 1) -> int:
    """Bytes per token of KV pool, per GPU."""
    heads = max(1, geom["kv_heads"] // tp)
    return heads * geom["layers"] * bytes_per_head(arm, geom)


def hp_arena_bytes(geom: dict, max_req_slots: int, hp_prefix_slots: int,
                   tp: int = 1) -> int:
    """Bytes the mixed-KV HP arena reserves *outside* the profiled budget.

    Every HP slot costs a full bf16 KV token. The default hp-prefix sizing is
    ``max_running_requests * PREFIX_TOKENS * 16``, so raising max_running_requests
    to admit a large batch silently reserves tens of GB -- which is why the
    driver pins SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS instead of inheriting it.
    """
    heads = max(1, geom["kv_heads"] // tp)
    # Mirrors unified_kv_pool.compute_recent_ring_size. The +N_Q (not N_Q-1)
    # reserves one transient slot: decode installs the new full->SWA mapping
    # before the flush plan releases the oldest N_Q, so a ring sized to the
    # steady-state maximum loses a mapping on the first wrap (fix p2-3).
    ring = RECENT_TOKENS + N_Q
    slots = hp_prefix_slots + max_req_slots * ring
    per_slot = geom["layers"] * heads * (geom["head_dim"] + geom["v_head_dim"]) * HP_BYTES
    return slots * per_slot


def hp_prefix_slots_for(max_reqs: int) -> int:
    """Enough HP-prefix slots to retain every cached prompt's bf16 prefix window.

    With the radix cache on, this tier holds PREFIX_TOKENS per retained prefix;
    undersize it and the warm pass quietly stops taking. 4x headroom, floored so
    small batches still get a usable cache.
    """
    return max(4096, 2 * max_reqs * PREFIX_TOKENS)


def swa_tokens_for_retention(ctx: int, out: int, batch: int) -> int:
    """SWA slots needed to keep ``batch`` cached prefixes matchable, FLAG OFF.

    One slot per full-tier token, for every token of every retained prefix. See
    ``b_max`` for the measurements. A tier smaller than this does not degrade
    reuse gracefully -- it drops it to zero: the stock SWA evictor tombstones a
    prefix's internal nodes root-first (harmless, mid-path tombstones still
    match) but DELETES its leaf -- the trailing window the matcher requires --
    so the match collapses entirely. Root cause of the bs_max caching cliff;
    fixed behind SGLANG_SWA_KEEP_PREFIX_TAIL, see swa_per_retained_prefix.
    """
    return batch * (ctx + out)


def swa_per_retained_prefix(ctx: int, out: int, chunked_prefill: int | None = None,
                            sliding_window: int | None = None,
                            page_size: int = N_Q) -> int:
    """SWA slots one cached prefix occupies while it stays matchable.

    Default (``chunked_prefill is None``) is the whole prefix: the stock SWA
    evictor deletes leaves, so the only state in which a prefix reliably
    matches is fully un-tombstoned (see swa_tokens_for_retention).

    With ``SGLANG_SWA_KEEP_PREFIX_TAIL`` the evictor trims each leaf to
    ``keep_target = sliding_window + 2*page`` (tombstoning the head) and never
    deletes short leaves, so what survives per prefix is one trailing window
    plus the owning request's live band -- independent of prompt length and of
    chunk size. Terms:

      retained_tail = w + 3*page      # trim tail lands in [w+2p, w+3p)
      live_band     = 2*w + page      # decode frontier lags <= window behind
                                      # the kept window (_evict_swa cadence)
                    + RECENT_TOKENS + N_Q   # mixed-KV HP-recent band (bf16
                                            # arms: overcount, safe)

    ``page_size`` is the radix-tree page (8 for the mixed-KV quant arms, 1 for
    bf16). ``chunked_prefill`` is the on/off switch only; it no longer enters
    the arithmetic -- the in-flight chunk is charged once per server in
    ``b_max`` (``chunk_tokens``), not per prefix.
    """
    if chunked_prefill is None:
        return ctx + out
    w = sliding_window or 0
    retained_tail = w + 3 * page_size
    live_band = 2 * w + page_size + RECENT_TOKENS + N_Q
    return retained_tail + live_band


def b_max(pool_tokens: int, ctx: int, out: int, max_reqs: int, graph_bs: int,
          swa_pool_tokens: int | None = None,
          swa_per_retained: int | None = None,
          chunk_tokens: int = 0) -> int:
    """Largest batch whose KV fits the pool, also bounded by scheduler and graphs.

    A batch above ``graph_bs`` would decode outside CUDA graphs -- a different
    code path, not a comparable measurement -- so it is a hard bound, not advice.

    The SWA tier bounds the batch at ``ctx + out`` per request, because that is
    what PREFIX RETENTION costs: keeping a cached prefix matchable means keeping
    its SWA un-tombstoned, and SWA is allocated 1:1 with full-tier tokens
    (common.py:697). Exceed it and the warm pass silently stops reusing --
    ``cached_tokens_median`` goes to 0, not to a smaller number -- because
    ``_match_prefix_helper`` truncates at ``best_value_len`` once a tombstone
    leaves fewer than ``sliding_window_size`` non-tombstone tokens behind it.

    Measured on gpt-oss at 90k (gate 4 of the SWA-sizing work): every arm obeys
    ``B*(ctx+out) <= swa_tokens``, bf16 and quant alike --

        bf16   tier 1,011,157  bs=10  need   910,240  -> cached 89,999
        bf16   tier   144,560  bs= 4  need   364,096  -> cached 0
        quant  tier 1,048,576  bs= 4  need   364,096  -> cached 89,728
        quant  tier 1,048,576  bs=71  need 6,462,704  -> cached 0

    Two corrections are folded in here. The bound used to be suppressed for
    quant arms (``swa_charges_per_token``) on the theory that the mixed pool
    spends SWA only on the HP prefix/recent parts. That WAS true when the
    published grid was measured and is not true now -- 55bc55d restored 1:1
    allocation -- which is why the quant arms show a caching cliff the bound
    would have predicted (task #32). And a later attempt to replace this with a
    per-request ``2*window + ring`` term was wrong FOR THE STOCK EVICTOR:
    ``swa_lock_ref`` governs whether ``evict`` may reclaim a node, not whether
    its slots are held, and the stock evictor deletes the leaf a match needs.
    With SGLANG_SWA_KEEP_PREFIX_TAIL that leaf survives, and a small
    ``swa_per_retained`` becomes the correct model again.

    ``chunk_tokens`` charges the in-flight prefill chunk against the SWA tier
    once per server: chunk SWA is held 1:1 until the chunk is inserted and
    tombstonable, and on a 131k tier an 8192-token chunk is 6.2% -- far past
    the 1% headroom, which covers page rounding and accounting slack only
    (measured on gpt-oss at 60k/bs=30: 99.3% nominal occupancy died 42 tokens
    short).
    """
    per_req = ctx + out
    bound = pool_tokens // per_req
    if swa_pool_tokens is not None:
        retained = swa_per_retained if swa_per_retained else per_req
        swa_budget = int(swa_pool_tokens * 0.99) - chunk_tokens
        bound = min(bound, max(0, swa_budget) // retained)
    return max(1, min(bound, max_reqs, graph_bs))


def fits_budget(geom: dict, arm: str, pool_tokens: int, max_reqs: int,
                mem_frac: float, card_bytes: int, tp: int = 1) -> tuple[bool, dict]:
    """Does the HP arena (allocated after profiling) still fit the reserve?

    Returns (ok, breakdown). The reserve is ``(1 - mem_frac) * free_before_weights``,
    which is approximately the whole card; a safety factor covers CUDA graphs,
    activations and fragmentation.
    """
    if arm == "bf16":
        return True, {"hp_gib": 0.0}
    hp = hp_arena_bytes(geom, max_reqs, hp_prefix_slots_for(max_reqs), tp)
    reserve = (1.0 - mem_frac) * card_bytes
    # 0.45, not 0.60: at mem_frac 0.90 an HP arena at 0.60 of the reserve
    # left too little for bs=64 CUDA graphs plus 90k prefill activations,
    # and every quant arm OOM'd partway through the 90k cells.
    ok = hp < 0.45 * reserve
    return ok, {
        "hp_gib": round(hp / GiB, 2),
        "reserve_gib": round(reserve / GiB, 2),
        "pool_gib": round(pool_tokens * cell_size(arm, geom, tp) / GiB, 2),
    }
