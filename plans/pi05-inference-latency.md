# pi05 inference latency on the A5000 — findings & optimizations

Companion to [`rtc-real-time-chunking.md`](./rtc-real-time-chunking.md). RTC *hides*
the ~600 ms latency behind the async queue; this work *reduces* it. Full write-up
and A/B table: **navid-a/shiraz issue #77**. Implementation:
[`almond_axol/lerobot/pi05_inference_opt.py`](../almond_axol/lerobot/pi05_inference_opt.py),
flags on [`inference-server`](../almond_axol/cli/inference_server.py), benchmark
[`almond_axol/diagnostics/bench_pi05.py`](../almond_axol/diagnostics/bench_pi05.py).

## Section 0 — read this first

- **The pi05 forward is ~494 ms on an A5000, not 600 ms** — the operator's 600 ms
  also includes server pre/post-proc + ZED read + gRPC. Benchmark the *forward*
  in isolation: `python -m almond_axol.diagnostics.bench_pi05 --sweep --breakdown`.
- **The model already runs bf16** (checkpoint `dtype=bfloat16`). The "easy fp32→bf16"
  win does not exist. **But the SigLIP vision tower is forced fp32** (a *train*-time
  optimizer-dtype workaround) — at inference that is free to flip to bf16 (~-70 ms,
  `max|Δ|≈1e-2`). This is the single biggest accuracy-safe win.
- **`denoise_step` deepcopies the whole 2B prefix KV-cache every step** (10×/infer).
  It is **load-bearing, not gratuitous**: the expert forward appends its 50 suffix
  tokens to the prefix cache (`DynamicLayer.update` → `cat`), so dropping the copy
  corrupts it (`RuntimeError: expand 1274→1324`). Replace with a **shallow container
  copy** (new layer objects, shared tensors) — bit-identical, no 23 MB/step clone.
- **The 10-step denoise loop (~300 ms) is ~30 ms/step of almost pure launch
  overhead** (a 300M expert over 50 tokens is <1 ms of compute). Fewer steps is the
  big lever; CUDA graphs are the lossless lever but `torch.compile` over the dynamic
  cache takes minutes to build and the first inference pays it (left opt-in).
- **SDPA is a no-op here** (~1.02×) on these sequence lengths *and* perturbs the
  output — intentionally **not** wired. Don't re-add it without new evidence.
- **TF32 did nothing** (vision isn't matmul-bound the way TF32 helps) — also dropped.

## Result

| Stack | ms | × | accuracy |
|---|---|---|---|
| baseline (10 steps) | 494 | 1.00 | — |
| `vision_bf16 + cheap_kv_cache` (10 steps) | ~406 | 1.22 | safe (`max|Δ|≤1e-2`) |
| **+ `num_inference_steps=6` (server default)** | **~300** (median 297, p90 317) | **1.65** | `max|Δ|≈5e-2` vs 10 |
| + `num_inference_steps=5` | ~284 (p90 299) | 1.74 | `max|Δ|≈8e-2` |

`vision_bf16` + `cheap_kv_cache` are accuracy-safe and on by default. The server
also defaults `num_inference_steps=6` (≈300 ms, the ≤300 ms target); set `10` for
the trained fidelity (~406 ms) or `5` for more speed (~284 ms). The step reduction
is the **only** change needing on-robot validation — compare 6- (or 5-) vs 10-step
predicted trajectories in `run-policy --shadow` before a powered run.
