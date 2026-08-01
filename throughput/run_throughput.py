#!/usr/bin/env python3
"""Controlled decode-throughput benchmark: bs=1, bs=4, bs=max, no prefix sharing.

Measures aggregate decode throughput with prefill explicitly excluded, at three
operating points per (model, arm, input length):

  bs=1     equal-setting latency
  bs=4     scaling
  bs=max   best achievable per GPU -- the largest batch whose KV fits the pool

Two properties are enforced rather than hoped for:

  * No CROSS-REQUEST prefix sharing. Every prompt is a distinct random token
    stream, so no two requests share a prefix. This is the ambiguity that made
    OSCAR's Figure 4 unreproducible -- prefix structure alone moved one
    measurement between 1.08x and 5.66x.
  * Each request's OWN prefix is warmed before measuring. Cold, a bs=max cell
    never forms: prefilling 82 x 30k = 2.5M tokens takes minutes while 512
    decode tokens take ~15 s, so early requests finish generating before late
    ones start decoding and the batch never co-exists. The warm pass collapses
    every TTFT to a single decode step. It costs almost no extra wall clock --
    it does the prefill the cold run would have done anyway, just separated from
    the decode instead of interleaved with it.

Usage:
  python -u throughput/run_throughput.py \
      --config throughput/configs/qwen3_8b.json --gpus 0,1,2
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capacity  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BOOT_TIMEOUT_S = 2400
GPU_FREE_MIB = 2000
REQ_TIMEOUT_S = 5400

_print_lock = threading.Lock()


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO / q


def log(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def gpu_used_mib(gpu: int) -> int:
    r = sh(["nvidia-smi", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits", "-i", str(gpu)])
    try:
        return int(r.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return 10 ** 9


def card_bytes(gpu: int) -> int:
    r = sh(["nvidia-smi", "--query-gpu=memory.total",
            "--format=csv,noheader,nounits", "-i", str(gpu)])
    return int(r.stdout.strip().splitlines()[0]) * 1024 * 1024


def wait_gpu_free(gpu: int, timeout_s: int = 420) -> bool:
    """Poll to actual freedom. A fixed sleep after pkill has produced a
    contaminated pool size before (248633 vs 402927 on a clean card)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if gpu_used_mib(gpu) < GPU_FREE_MIB:
            return True
        time.sleep(5)
    return False


class Server:
    """One serve_method.sh instance, port-scoped so concurrent runs cannot
    tear down each other's servers."""

    def __init__(self, arm: dict, cfg: dict, gpus: list[int], port: int,
                 log_dir: Path, max_reqs: int, graph_bs: int, tag: str):
        self.arm, self.cfg, self.gpus, self.port = arm, cfg, gpus, port
        self.gpu = gpus[0]
        self.tp = len(gpus)
        self.max_reqs, self.graph_bs = max_reqs, graph_bs
        self.log = log_dir / f"serve_{arm['label']}_{tag}_gpu{gpus[0]}.log"
        self.mem_frac = cfg["serve"]["mem_frac"]
        self.proc: subprocess.Popen | None = None

    def _kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()
        self.proc = None
        # Belt and braces: a server that daemonised past the process group.
        sh(["pkill", "-9", "-f", f"[l]aunch_server.*--port {self.port}"])

    def start(self, mem_frac: float) -> None:
        self._kill()
        for g in self.gpus:
            wait_gpu_free(g)
        self.mem_frac = mem_frac
        self.log.parent.mkdir(parents=True, exist_ok=True)
        if self.log.exists():
            self.log.unlink()

        s = self.cfg["serve"]
        env = dict(os.environ)
        env.update({
            "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in self.gpus),
            "RADIX_CACHE": "1",
            "MAX_REQS": str(self.max_reqs),
            "PREFILL_BACKEND": s.get("prefill_backend", "fa3"),
            "TP": str(self.tp),
        })
        if self.arm["method"] != "bf16":
            env["SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS"] = str(
                capacity.hp_prefix_slots_for(self.max_reqs))
            # Without this the quant arms retain only the first prefill chunk
            # (~7936 tokens) while bf16 caches the whole prompt -- the warm pass
            # would then hand bf16 a silent advantage.
            env["SGLANG_MIXED_KV_PREFIX_REUSE_ACROSS_CHUNKS"] = "1"
        env.update({k: str(v) for k, v in self.arm.get("env", {}).items()})

        cmd = [
            "bash", str(REPO / "scripts/serve_method.sh"),
            self.cfg["model"], self.arm["method"],
            str(_abs(self.cfg["artifact_dir"])),
            "--port", str(self.port),
            "--context-length", str(s["ctx"]),
            "--mem-fraction-static", str(mem_frac),
            "--cuda-graph-max-bs", str(self.graph_bs),
            "--enable-cache-report",
            # Chunk size drives how much of a long prompt the quant arms can
            # retain: prefix reuse commits only up to the last COMPLETE prefill
            # chunk minus the recent window, so the cached fraction is
            # floor((L-1)/C)*C - RECENT. Smaller C -> higher coverage. At
            # C=8192 a 30k prompt caches 81%; at 4096, 94.7%.
            "--chunked-prefill-size", str(s.get("chunked_prefill_size", 4096)),
        ] + s.get("extra", "").split() + self.arm.get("extra", "").split()
        handle = self.log.open("w")
        self.proc = subprocess.Popen(cmd, env=env, stdout=handle, stderr=handle,
                                     cwd=REPO, start_new_session=True)

    def wait_ready(self) -> tuple[bool, str]:
        deadline = time.time() + BOOT_TIMEOUT_S
        while time.time() < deadline:
            t = self.log.read_text(errors="ignore")
            if "The server is fired up" in t:
                return True, "ok"
            for needle, why in (("CUDA out of memory", "oom"),
                                ("Not enough memory", "oom"),
                                ("Received sigquit", "crash"),
                                ("Traceback (most recent call last)", "crash")):
                if needle in t:
                    return False, why
            time.sleep(5)
        return False, "timeout"

    def pool_tokens(self) -> int | None:
        m = re.findall(r"max_total_num_tokens=(\d+)",
                       self.log.read_text(errors="ignore"))
        return int(m[-1]) if m else None

    def swa_pool_tokens(self) -> int | None:
        """SWA-tier size on hybrid models; None on dense ones.

        The scheduler's ``max_total_num_tokens`` reports the *full* tier only,
        so the SWA tier has to be read out of the sizing line. Which rule chose
        it -- pin, derived, or ratio -- is printed alongside the number there.
        """
        m = re.findall(r"swa_layer_tokens=(\d+)",
                       self.log.read_text(errors="ignore"))
        return int(m[-1]) if m else None

    def mark(self) -> int:
        return len(self.log.read_text(errors="ignore"))

    def since(self, mark: int) -> str:
        return self.log.read_text(errors="ignore")[mark:]

    def server_info(self) -> dict:
        try:
            r = requests.get(f"http://127.0.0.1:{self.port}/server_info", timeout=30)
            return r.json()
        except Exception:
            return {}

    def stop(self) -> None:
        self._kill()
        for g in self.gpus:
            wait_gpu_free(g)


# ---------------------------------------------------------------------------
# client


def make_batch(bs: int, ctx: int, vocab: int, seed: int) -> list[list[int]]:
    """Distinct random token streams -- zero prefix overlap by construction.

    Same scheme as sglang's own random-ids dataset: a distinct random offset per
    request, then a strided walk. Two requests could only collide if their
    offsets coincided modulo the vocabulary.
    """
    rng = random.Random(seed)
    hi = min(vocab, 30000)
    return [[(rng.randrange(10, hi) + i + j) % hi or 10 for j in range(ctx)]
            for i in range(bs)]


def one_request(port: int, ids: list[int], out_tokens: int) -> dict:
    """Stream one generation; record only first-token and completion times.

    Deliberately does no per-token parsing: at bs=max the batch emits thousands
    of SSE events per second and a heavyweight client loop would show up as
    decode latency. The /server_info cross-check below catches it if it does.
    """
    t0 = time.perf_counter()
    r = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json={"input_ids": ids,
              "sampling_params": {"max_new_tokens": out_tokens, "temperature": 0,
                                  "ignore_eos": True},
              "stream": True},
        stream=True, timeout=REQ_TIMEOUT_S)
    r.raise_for_status()
    first = None
    n = 0
    last = None
    for line in r.iter_lines():
        if line and line[:5] == b"data:":
            if first is None:
                first = time.perf_counter()
            n += 1
            if line[6:] != b"[DONE]":
                last = line[6:]
    done = time.perf_counter()

    # A short generation silently shrinks the decode window and inflates
    # throughput. With synthetic codebooks the model emits garbage, so EOS could
    # fire at any point and at a different rate per arm -- ignore_eos is set, but
    # trusting it unverified would be exactly the kind of thing that produces a
    # confident wrong number. Parse the final payload (one parse per request,
    # not per token) and report what actually happened.
    produced, finish = None, None
    if last:
        try:
            meta = json.loads(last).get("meta_info") or {}
            produced = meta.get("completion_tokens")
            finish = (meta.get("finish_reason") or {}).get("type")
        except (json.JSONDecodeError, AttributeError):
            pass
    return {"ttft": (first - t0) if first else float("nan"),
            "latency": done - t0, "events": n,
            "produced": produced, "finish": finish}


def run_batch(port: int, batch: list[list[int]], out_tokens: int) -> dict:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(batch)) as ex:
        res = list(ex.map(lambda ids: one_request(port, ids, out_tokens), batch))
    wall = time.perf_counter() - t0
    ttfts = [r["ttft"] for r in res]
    produced = [r["produced"] for r in res]
    # Per-request decode rate excludes that request's OWN prefill, so summing
    # over the batch is insensitive to requests entering decode at staggered
    # times. Comparing it against the window-based figure below is what tells
    # us whether stagger actually biased the measurement, instead of relying on
    # a spread threshold picked by hand.
    per_req = [(r["produced"] or 0) / max(r["latency"] - r["ttft"], 1e-9)
               for r in res]
    return {"wall": wall, "ttfts": ttfts,
            "first_ttft": min(ttfts), "last_ttft": max(ttfts),
            "sum_per_req_rate": sum(per_req),
            "events": sum(r["events"] for r in res),
            "produced_min": min((p for p in produced if p is not None), default=None),
            "produced_max": max((p for p in produced if p is not None), default=None),
            "finish_reasons": sorted({r["finish"] for r in res})}


def measure_cell(srv: Server, bs: int, ctx: int, out_tokens: int,
                 vocab: int, seed: int) -> dict:
    batch = make_batch(bs, ctx, vocab, seed)

    # Warm pass: populate each request's own prefix. Discarded.
    t_warm = time.perf_counter()
    run_batch(srv.port, batch, 1)
    warm_s = time.perf_counter() - t_warm

    mark = srv.mark()
    r = run_batch(srv.port, batch, out_tokens)
    tail = srv.since(mark)

    decode_window = r["wall"] - r["last_ttft"]
    m = {
        "batch_size": bs, "input_len": ctx, "output_len": out_tokens,
        "warm_s": round(warm_s, 2),
        "latency_s": round(r["wall"], 4),
        "first_ttft_s": round(r["first_ttft"], 4),
        "last_ttft_s": round(r["last_ttft"], 4),
        "decode_window_s": round(decode_window, 4),
        # Prefill is excluded by construction: after the LAST request's first
        # token, every request is decoding and no prefill remains.
        "decode_tok_s_total": round(bs * out_tokens / decode_window, 2),
        "decode_tok_s_per_req": round(out_tokens / decode_window, 3),
        # HEADLINE. Sum of per-request decode rates, each excluding that
        # request's own prefill. Unbiased under staggered entry, unlike the
        # window form above, which divides ALL bs*out tokens by only the
        # post-last_ttft window and so credits tokens emitted earlier to a
        # shorter window (measured: +5-7% at the largest batches, 0 when the
        # spread is 0). The two agree to <1% whenever entry is synchronous.
        "decode_tok_s": round(r["sum_per_req_rate"], 2),
        "decode_tok_s_stagger_free": round(r["sum_per_req_rate"], 2),
        "overall_tok_s": round(bs * (ctx + out_tokens) / r["wall"], 2),
    }

    cached = [int(x) for x in re.findall(r"#cached-token: (\d+)", tail)]
    cached_med = sorted(cached)[len(cached) // 2] if cached else 0
    running = [int(x) for x in re.findall(r"running-req: (\d+)", tail)]
    info = srv.server_info()
    try:
        gen_tp = float(info["internal_states"][0]["last_gen_throughput"])
    except Exception:
        gen_tp = None

    spread = (r["last_ttft"] - r["first_ttft"]) / max(decode_window, 1e-9)
    # Agreement between the window-based and stagger-free estimators is direct
    # evidence that staggered entry did not distort the number.
    est_gap = abs(m["decode_tok_s_stagger_free"] - m["decode_tok_s_total"]) / max(
        m["decode_tok_s_total"], 1e-9)
    m["gates"] = {
        # Every request must emit exactly output_len tokens. Anything shorter
        # means a request stopped at EOS, which shortens the decode window and
        # inflates tok/s -- by a different amount per arm.
        "full_length": (r["produced_min"] == out_tokens
                        and r["produced_max"] == out_tokens),
        "no_retract": ("Retract requests" not in tail
                       and "retract_decode: aborted" not in tail),
        # Measured-pass prefill must be a small fraction of the cold prefill.
        # This is the batch-invariant form: "#cached-token" is logged per
        # PREFILL BATCH, so at bs>4 it reports a sum across co-scheduled
        # requests and comparing it to a single ctx passes trivially.
        # Observed: bf16 ~0.03 (caches 100%), vq2 ~0.11 (caches ~95%, capped by
        # the final partial prefill chunk). A regression to no caching -> ~1.0.
        "warm_took": r["last_ttft"] <= 0.25 * max(warm_s, 1e-9),
        # The headline metric is stagger-robust, so a TTFT spread no longer
        # invalidates a cell; what still must hold is that the scheduler
        # actually ran all bs requests together. Spread and the inter-estimator
        # gap are recorded for audit rather than gated on.
        "batch_coexisted": (max(running) if running else 0) >= bs,
        "no_nan": "nan" not in tail.lower() and "inf detected" not in tail.lower(),
        # The server's own throughput counter updates on its decode-log
        # interval, so it is only meaningful once the run spans several
        # intervals; on a short cell it reports a stale value, not a mismatch.
        "server_agrees": (decode_window < 5.0 or
                          (gen_tp is not None
                           and abs(gen_tp - m["decode_tok_s_total"])
                           <= 0.10 * m["decode_tok_s_total"])),
    }
    m["produced_tokens"] = [r["produced_min"], r["produced_max"]]
    m["finish_reasons"] = r["finish_reasons"]
    m["cached_tokens_median"] = cached_med
    m["ttft_spread_frac"] = round(spread, 4)
    m["estimator_gap_frac"] = round(est_gap, 4)
    m["max_running_req"] = max(running) if running else None
    m["server_last_gen_throughput"] = gen_tp
    m["ok"] = all(m["gates"].values())
    return m


# ---------------------------------------------------------------------------
# driver


def run_arm(arm: dict, cfg: dict, geom: dict, gpus: list[int], port: int,
            out_dir: Path, log_dir: Path, hb: Path) -> None:
    label = arm["label"]
    lengths = cfg["input_lens"]
    out_tokens = cfg["output_len"]
    ceiling = cfg.get("bs_max_ceiling", 64)
    mem_frac = cfg["serve"]["mem_frac"]
    fallback = cfg["serve"].get("mem_frac_fallback", 0.85)
    card = card_bytes(gpus[0])
    tp = len(gpus)
    win = geom.get("sliding_window")
    # SGLANG_SWA_KEEP_PREFIX_TAIL changes what a cached prefix costs in SWA from
    # its whole length to ~(window + ring) tokens, so b_max's retention bound
    # must follow it or the arm is planned at a batch far below what it can hold.
    # Default mirrors serve_method.sh: ON for quant arms, stock for bf16 -- an
    # arm env of "0" opts back out. The two defaults must stay in sync or the
    # harness plans with the wrong retention model.
    keep_tail_default = "0" if arm["method"] == "bf16" else "1"
    keep_tail = str(
        arm.get("env", {}).get("SGLANG_SWA_KEEP_PREFIX_TAIL", keep_tail_default)
    ) in ("1", "true", "True")
    swa_per = (capacity.swa_per_retained_prefix(
        0, 0, cfg["serve"].get("chunked_prefill_size", 4096), win)
        if keep_tail else None)

    # --- calibration boot: read the pool with a small graph/request budget ---
    srv = Server(arm, cfg, gpus, port, log_dir, max_reqs=8, graph_bs=8, tag="cal")
    srv.start(mem_frac)
    ok, why = srv.wait_ready()
    if not ok and why == "oom" and mem_frac != fallback:
        log(f"{label}: calibration OOM at mem_frac={mem_frac}, retrying {fallback}")
        mem_frac = fallback
        srv.start(mem_frac)
        ok, why = srv.wait_ready()
    if not ok:
        log(f"{label}: CALIBRATION BOOT FAILED ({why}) -- skipping arm")
        srv.stop()
        return
    pool_cal = srv.pool_tokens()
    # Read the tier back rather than trusting the pin: an arm may override it,
    # and the engine page-aligns. capacity.b_max then charges each retained
    # prefix swa_per (small, keep-tail arms) or ctx+out (stock arms, whole
    # prefix -- the stock evictor's leaf deletion kills any partial retention).
    swa_cal = srv.swa_pool_tokens()
    srv.stop()

    chunk_tokens = cfg["serve"].get("chunked_prefill_size", 4096)
    predicted_cell = capacity.cell_size(
        "bf16" if arm["method"] == "bf16" else "int2", geom, tp)
    b_needed = max(capacity.b_max(pool_cal, c, out_tokens, 10 ** 6, ceiling,
                                  swa_cal, swa_per, chunk_tokens)
                   for c in lengths)
    fits, budget = capacity.fits_budget(geom, label, pool_cal, b_needed,
                                        mem_frac, card, tp)
    while not fits and b_needed > 4:
        b_needed //= 2
        fits, budget = capacity.fits_budget(geom, label, pool_cal, b_needed,
                                            mem_frac, card, tp)
    log(f"{label}: pool={pool_cal:,} swa={swa_cal if swa_cal is None else f'{swa_cal:,}'} "
        f"cell={predicted_cell}B graph_bs={b_needed} "
        f"hp={budget.get('hp_gib')}GiB mem_frac={mem_frac}")

    # --- measurement boot: graphs and request slots sized for the real batch ---
    srv = Server(arm, cfg, gpus, port, log_dir, max_reqs=b_needed,
                 graph_bs=b_needed, tag="run")
    srv.start(mem_frac)
    ok, why = srv.wait_ready()
    if not ok and why == "oom":
        log(f"{label}: measurement OOM at graph_bs={b_needed}; halving")
        b_needed = max(4, b_needed // 2)
        srv = Server(arm, cfg, gpus, port, log_dir, max_reqs=b_needed,
                     graph_bs=b_needed, tag="run")
        srv.start(mem_frac)
        ok, why = srv.wait_ready()
    if not ok:
        log(f"{label}: MEASUREMENT BOOT FAILED ({why}) -- skipping arm")
        srv.stop()
        return

    pool = srv.pool_tokens()
    swa_pool = srv.swa_pool_tokens()
    (out_dir / label).mkdir(parents=True, exist_ok=True)
    (out_dir / label / "provenance.json").write_text(json.dumps({
        "arm": label, "model": cfg["model"], "gpus": gpus, "tp": tp,
        "pool_tokens": pool, "pool_tokens_calibration": pool_cal,
        "swa_pool_tokens": swa_pool,
        "sliding_window": win, "swa_per_retained_prefix": swa_per,
        "predicted_cell_size_bytes": predicted_cell,
        "predicted_pool_gib": round(pool * predicted_cell / capacity.GiB, 2),
        "note": "predicted_* assume a uniform full-attention KV pool; hybrid-SWA models (gpt-oss) allocate the quant pool for full-attention layers only, so the prediction is an upper bound there",
        "mem_frac": mem_frac, "max_running_requests": b_needed,
        "cuda_graph_max_bs": b_needed, "method": arm["method"],
        "extra": arm.get("extra", ""),
        "env": arm.get("env", {}), "budget": budget,
        "artifact_dir": str(cfg["artifact_dir"]),
        "synthetic_artifacts": bool(cfg.get("synthetic_artifacts", False)),
    }, indent=2))

    # On hybrid-SWA models b_max is bounded by the SWA tier as well as the full
    # tier (see capacity.b_max) -- sizing off the full tier alone crashed the
    # quant arms and silently un-cached bf16's warm pass. quant_bmax_slack
    # predates that fix and stays only as an override knob; 1.0 is a no-op.
    slack = (float(cfg.get("quant_bmax_slack", 1.0))
             if arm["method"] != "bf16" else 1.0)
    # ``batch_sizes`` replaces the default {1, 4, b_max} triple with an explicit
    # ladder, for studies that need the throughput *curve* rather than three
    # points -- e.g. locating the batch at which decode saturates. Entries above
    # an arm's b_max are dropped rather than recorded infeasible, since the point
    # of a shared ladder is that arms reach different distances along it.
    ladder = cfg.get("batch_sizes")
    for ctx in lengths:
        bmax = capacity.b_max(pool, ctx, out_tokens, b_needed, b_needed,
                              swa_pool, swa_per, chunk_tokens)
        bmax = max(1, int(bmax * slack))
        # An explicit ladder means exactly those points: do NOT force b_max in.
        # b_max is a memory-derived ceiling, and reaching it can kill the server
        # (at 1:1 SWA allocation a bs=210/30k cell exhausts the SWA tier), which
        # takes every later cell for that arm down with it. A study that pins its
        # own batch list -- e.g. a kernel-config sweep -- must not have an
        # unreachable point appended to it.
        if ladder:
            points = {b for b in ladder if b <= bmax} or {bmax}
        else:
            points = {1, 4, bmax}
        for bs in sorted(points):
            if bs > bmax:
                cell = out_dir / label / f"in{ctx}_bs{bs}"
                cell.mkdir(parents=True, exist_ok=True)
                (cell / "metrics.json").write_text(json.dumps({
                    "batch_size": bs, "input_len": ctx, "ok": False,
                    "infeasible": True, "b_max": bmax, "pool_tokens": pool,
                    "reason": f"pool holds {pool:,} tokens; bs={bs} needs "
                              f"{bs * (ctx + out_tokens):,}",
                }, indent=2))
                log(f"{label} in{ctx} bs={bs}: INFEASIBLE (b_max={bmax})")
                continue
            tag = f"in{ctx}_bs{bs}"
            try:
                m = measure_cell(srv, bs, ctx, out_tokens, geom["vocab"],
                                 cfg.get("seed", 1))
            except Exception as e:
                log(f"{label} {tag}: FAILED {type(e).__name__}: {e}")
                hb.touch()
                continue
            m.update({"arm": label, "pool_tokens": pool, "b_max": bmax,
                      "mem_frac": mem_frac, "is_bs_max": bs == bmax})
            cell = out_dir / label / tag
            cell.mkdir(parents=True, exist_ok=True)
            (cell / "metrics.json").write_text(json.dumps(m, indent=2))
            bad = [k for k, v in m["gates"].items() if not v]
            log(f"{label} {tag}: {m['decode_tok_s_total']:>9,.0f} tok/s "
                f"({m['decode_tok_s_per_req']:.1f}/req)  "
                f"cached={m['cached_tokens_median']:,} "
                f"spread={m['ttft_spread_frac']:.3f}"
                + (f"  GATES FAILED: {bad}" if bad else "  ok"))
            hb.touch()

    srv.stop()


def aggregate(out_dir: Path, cfg: dict) -> dict:
    rows = {}
    for f in sorted(out_dir.glob("*/in*/metrics.json")):
        m = json.loads(f.read_text())
        # Re-derive against the current gate definition so summaries stay
        # consistent with cells measured before the metric switch.
        if "gates" in m:
            m["decode_tok_s"] = m.get("decode_tok_s",
                                      m.get("decode_tok_s_stagger_free"))
            g = dict(m["gates"])
            g["batch_coexisted"] = (m.get("max_running_req") or 0) >= m["batch_size"]
            m["gates"] = g
            m["ok"] = all(g.values())
        rows.setdefault(f.parent.parent.name, {})[f.parent.name] = m
    ref = cfg.get("reference", "bf16")
    summary = {"study": cfg["study"], "model": cfg["model"], "reference": ref,
               "rows": rows, "speedup_matched_bs": {}, "speedup_best": {}}

    # Matched batch size: same operating point on both arms (bs=1 and bs=4).
    for arm, cells in rows.items():
        if arm == ref:
            continue
        for cell, m in cells.items():
            r = rows.get(ref, {}).get(cell)
            if m.get("ok") and r and r.get("ok"):
                summary["speedup_matched_bs"].setdefault(arm, {})[cell] = round(
                    m["decode_tok_s"] / r["decode_tok_s"], 3)

    # bs=max: arms reach DIFFERENT batch sizes, which is the point -- a bigger
    # pool admits a bigger batch. Comparing per-cell would silently drop these,
    # so compare each arm's best achievable throughput at a given length.
    def best(arm: str, ctx: int) -> dict | None:
        cands = [m for c, m in rows.get(arm, {}).items()
                 if m.get("ok") and m.get("input_len") == ctx]
        return max(cands, key=lambda m: m["decode_tok_s"]) if cands else None

    for ctx in cfg["input_lens"]:
        rb = best(ref, ctx)
        if not rb:
            continue
        for arm in rows:
            ab = best(arm, ctx)
            if not ab:
                continue
            summary["speedup_best"].setdefault(arm, {})[f"in{ctx}"] = {
                "tok_s": ab["decode_tok_s"], "bs": ab["batch_size"],
                "ref_tok_s": rb["decode_tok_s"], "ref_bs": rb["batch_size"],
                "speedup": round(ab["decode_tok_s"] / rb["decode_tok_s"], 3),
            }
    (out_dir / "throughput_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--gpus", type=lambda s: [int(x) for x in s.split(",")],
                    default=[0])
    ap.add_argument("--base-port", type=int, default=32000)
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text())
    out_dir = REPO / "results/throughput" / cfg["study"]
    log_dir = REPO / "logs" / cfg["study"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    hb = REPO / "logs" / f"{cfg['study']}.heartbeat"

    if a.aggregate_only:
        print(json.dumps(aggregate(out_dir, cfg), indent=2)[:4000])
        return 0

    shutil.copy(a.config, out_dir / "config.json")
    geom = capacity.model_geometry(cfg["model"])
    log(f"{cfg['study']}: {cfg['model']} L={geom['layers']} H={geom['kv_heads']} "
        f"D={geom['head_dim']} vocab={geom['vocab']}")

    tp = cfg.get("tp", 1)
    groups = [a.gpus[i:i + tp] for i in range(0, len(a.gpus) - tp + 1, tp)]
    if not groups:
        raise SystemExit(f"need at least tp={tp} GPUs, got {a.gpus}")
    log(f"tp={tp}: {len(groups)} concurrent server slot(s) {groups}")

    jobs: queue.Queue = queue.Queue()
    for i, arm in enumerate(cfg["arms"]):
        jobs.put((i, arm))

    def worker(slot: int, group: list[int]) -> None:
        while True:
            try:
                i, arm = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                run_arm(arm, cfg, geom, group, a.base_port + slot, out_dir,
                        log_dir, hb)
            except Exception as e:
                log(f"{arm['label']}: ARM CRASHED {type(e).__name__}: {e}")
            finally:
                jobs.task_done()

    threads = [threading.Thread(target=worker, args=(i, g), daemon=True)
               for i, g in enumerate(groups)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        hb.touch()
        time.sleep(20)
    for t in threads:
        t.join()

    s = aggregate(out_dir, cfg)
    log(f"DONE {cfg['study']} -> {out_dir}/throughput_summary.json")
    print(json.dumps(s["speedup_best"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
