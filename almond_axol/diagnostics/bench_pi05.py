"""
pi05 inference latency benchmark + optimization A/B harness.

Loads a real pi05 checkpoint exactly as ``axol inference-server`` does and times
the model forward (``predict_action_chunk``) on this machine's GPU, with a
component breakdown (vision encode / prefix LLM prefill / flow-matching denoise
loop). Each optimization is a toggle so the same script measures the baseline
and every candidate, and verifies that lossless ones don't change the output.
In --sweep mode the checkpoint is loaded once and a set of configs is walked, so
the ~40 s load is paid a single time.

This measures the dominant cost the operator sees as "~600 ms inference"; the
server's pre/post-processing is separate and small.

    cd third_party/axol
    .venv/bin/python -m almond_axol.diagnostics.bench_pi05 --sweep --breakdown
    .venv/bin/python -m almond_axol.diagnostics.bench_pi05 --vision-bf16 --sdpa --steps 5
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import time
import types

import torch

# Optimization implementations are the production ones, exercised here via toggles.
from ..lerobot.pi05_inference_opt import apply_vision_bf16, cheap_clone_cache

DEFAULT_CKPT = "/home/eevee/axol_eval_ckpts/milano_side/last/pretrained_model"


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def cuda_time_ms(fn, n: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out: list[float] = []
    for _ in range(n):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end))
    return out


def summary(samples: list[float]) -> str:
    s = sorted(samples)
    mean = statistics.mean(s)
    median = statistics.median(s)
    p90 = s[min(len(s) - 1, int(0.9 * len(s)))]
    std = statistics.pstdev(s) if len(s) > 1 else 0.0
    return f"mean {mean:7.1f} ms | median {median:7.1f} | p90 {p90:7.1f} | std {std:4.1f}"


# --------------------------------------------------------------------------- #
# Synthetic batch matching the real observation shapes
# --------------------------------------------------------------------------- #
def build_batch(policy, device: str, batch_size: int) -> dict:
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    cfg = policy.config
    batch: dict[str, torch.Tensor] = {}
    for key, feat in cfg.image_features.items():
        c, h, w = feat.shape
        batch[key] = torch.rand(batch_size, c, h, w, device=device)
    seq = cfg.tokenizer_max_length
    batch[OBS_LANGUAGE_TOKENS] = torch.randint(0, 257_152, (batch_size, seq), device=device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(
        batch_size, seq, dtype=torch.bool, device=device
    )
    return batch


# --------------------------------------------------------------------------- #
# Optimizations
# --------------------------------------------------------------------------- #
def _sdpa_attention_forward(module, query, key, value, attention_mask, scaling=None, **kwargs):
    """Drop-in for gemma ``eager_attention_forward`` backed by fused SDPA.

    Returns ``(attn_output[B, q, n_heads, head_dim], None)``; the incoming mask is
    the additive float mask the eager path already receives, so the math matches
    up to kernel-level reduction order.
    """
    import torch.nn.functional as F

    groups = query.shape[1] // key.shape[1]
    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    attn_mask = attention_mask
    if attn_mask is not None:
        attn_mask = attn_mask[:, :, :, : key.shape[-2]].to(query.dtype)
    out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, scale=scaling)
    return out.transpose(1, 2).contiguous(), None


class Patcher:
    """Apply/revert the in-process (reversible) optimizations for sweep mode."""

    def __init__(self, policy, modeling):
        self.policy = policy
        self.modeling = modeling
        self._orig_copy = modeling.copy
        self._orig_eager = modeling.modeling_gemma.eager_attention_forward
        self._orig_steps = policy.config.num_inference_steps
        self._orig_sample = policy.model.sample_actions

    def apply(self, *, sdpa=False, cheap_cache=False, steps=None, compile=False) -> None:
        self.modeling.copy = (
            types.SimpleNamespace(deepcopy=cheap_clone_cache) if cheap_cache else self._orig_copy
        )
        self.modeling.modeling_gemma.eager_attention_forward = (
            _sdpa_attention_forward if sdpa else self._orig_eager
        )
        self.policy.config.num_inference_steps = steps if steps is not None else self._orig_steps
        if compile:
            self.policy.model.sample_actions = torch.compile(
                self._orig_sample, mode="reduce-overhead"
            )
        else:
            self.policy.model.sample_actions = self._orig_sample


# --------------------------------------------------------------------------- #
# Component breakdown
# --------------------------------------------------------------------------- #
class Breakdown:
    def __init__(self, policy):
        self.policy = policy
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self._orig: dict = {}

    def _wrap(self, obj, attr, label):
        orig = getattr(obj, attr)
        self._orig[(obj, attr)] = orig

        def wrapped(*a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig(*a, **k)
            torch.cuda.synchronize()
            self.totals[label] = self.totals.get(label, 0.0) + (time.perf_counter() - t0) * 1e3
            self.counts[label] = self.counts.get(label, 0) + 1
            return out

        setattr(obj, attr, wrapped)

    def __enter__(self):
        m = self.policy.model
        self._wrap(m.paligemma_with_expert, "embed_image", "vision")
        self._wrap(m, "denoise_step", "denoise")
        return self

    def __exit__(self, *exc):
        for (obj, attr), orig in self._orig.items():
            setattr(obj, attr, orig)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_fixed(policy, batch, noise):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, img_masks = policy._preprocess_images(batch)
    return policy.model.sample_actions(
        images,
        img_masks,
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        noise=noise,
    )


def bench(policy, batch, args, label, noise=None, ref=None):
    samples = cuda_time_ms(
        lambda: policy.predict_action_chunk(batch), n=args.iters, warmup=args.warmup
    )
    diff = ""
    out = None
    if noise is not None:
        out = sample_fixed(policy, batch, noise)
        if ref is not None:
            d = (out - ref).abs().max().item()
            diff = f"  max|Δ|={d:.2e}"
    return statistics.mean(samples), samples, out, diff


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--sdpa", action="store_true")
    ap.add_argument("--cheap-cache", action="store_true")
    ap.add_argument("--vision-bf16", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument(
        "--prod",
        action="store_true",
        help="exercise the real server path: enable_pi05_inference_optimizations() + wrapped from_pretrained",
    )
    args = ap.parse_args()

    from lerobot.policies.pi05 import modeling_pi05

    if args.prod:
        # Mirror `axol inference-server` exactly: arm the optimizations, then load
        # via the (now-wrapped) from_pretrained so the patches apply on load.
        from ..lerobot.pi05_inference_opt import enable_pi05_inference_optimizations

        enable_pi05_inference_optimizations(
            vision_bf16=True,
            cheap_kv_cache=True,
            num_inference_steps=args.steps,
            compile=args.compile,
        )

    print(f"Loading policy from {args.ckpt} ...", flush=True)
    with contextlib.redirect_stdout(None):
        policy = modeling_pi05.PI05Policy.from_pretrained(args.ckpt)
        policy.to(args.device)
        policy.eval()
    cfg = policy.config
    batch = build_batch(policy, args.device, args.batch)
    patcher = Patcher(policy, modeling_pi05)

    gen = torch.Generator(device=args.device).manual_seed(0)
    noise = torch.randn(
        args.batch, cfg.chunk_size, cfg.max_action_dim, generator=gen, device=args.device
    )

    print("=" * 92)
    print(f"pi05 inference benchmark | {torch.cuda.get_device_name(0)}")
    print(
        f"dtype={cfg.dtype} | views={len(cfg.image_features)} | chunk={cfg.chunk_size} | "
        f"denoise_steps={cfg.num_inference_steps} | tok_len={cfg.tokenizer_max_length} | batch={args.batch}"
    )
    print("=" * 92)

    if args.breakdown:
        patcher.apply()
        with Breakdown(policy) as bd:
            for _ in range(5):
                policy.predict_action_chunk(batch)
        total = statistics.mean(
            cuda_time_ms(lambda: policy.predict_action_chunk(batch), n=10, warmup=3)
        )
        vis = bd.totals.get("vision", 0.0) / 5
        den = bd.totals.get("denoise", 0.0) / 5
        print(
            f"breakdown: vision(fp32) {vis:6.1f} ms | denoise {den:6.1f} ms | "
            f"prefill+misc {total - vis - den:6.1f} ms | total {total:6.1f} ms"
        )
        print("-" * 92)

    if args.sweep:
        # Reversible configs first; the baseline output is the equivalence reference.
        configs = [
            ("baseline", dict()),
            ("sdpa", dict(sdpa=True)),
            ("cheap-cache (lossless deepcopy fix)", dict(cheap_cache=True)),
            ("sdpa + cheap-cache", dict(sdpa=True, cheap_cache=True)),
            ("steps=8", dict(steps=8)),
            ("steps=6", dict(steps=6)),
            ("steps=5", dict(steps=5)),
            ("steps=4", dict(steps=4)),
            ("LOSSLESS combo (sdpa+cheap-cache)", dict(sdpa=True, cheap_cache=True)),
            ("LOSSLESS + steps=5", dict(sdpa=True, cheap_cache=True, steps=5)),
        ]
        base_mean = None
        ref = None
        for label, kw in configs:
            patcher.apply(**kw)
            try:
                mean, samples, out, diff = bench(policy, batch, args, label, noise=noise, ref=ref)
            except Exception as e:
                print(f"{label:<40} FAILED: {type(e).__name__}: {str(e)[:70]}", flush=True)
                continue
            if base_mean is None:
                base_mean, ref = mean, out
            sp = f" | {base_mean / mean:4.2f}x" if base_mean else ""
            note = "  (steps changed → Δ expected)" if kw.get("steps") else ""
            print(f"{label:<40} {summary(samples)}{sp}{diff}{note}", flush=True)
        patcher.apply()  # reset to eager baseline before the sticky vision-bf16 row

        # Sticky: vision-bf16 (param cast, can't cleanly revert in-process).
        apply_vision_bf16(policy)
        patcher.apply(sdpa=True, cheap_cache=True)
        mean, samples, out, diff = bench(policy, batch, args, "vis", noise=noise, ref=ref)
        sp = f" | {base_mean / mean:4.2f}x" if base_mean else ""
        print(f"{'vision-bf16 + sdpa + cheap-cache':<40} {summary(samples)}{sp}{diff}", flush=True)
        patcher.apply(sdpa=True, cheap_cache=True, steps=5)
        mean, samples, out, diff = bench(policy, batch, args, "vis5", noise=noise, ref=ref)
        sp = f" | {base_mean / mean:4.2f}x" if base_mean else ""
        print(
            f"{'vision-bf16 + sdpa + cheap-cache + steps=5':<40} {summary(samples)}{sp}{diff}"
            "  (steps changed → Δ expected)",
            flush=True,
        )
        print("=" * 92)
    else:
        if args.prod:
            # Patches already applied by the wrapped from_pretrained — just measure.
            label = (
                f"PROD (vision_bf16, cheap_kv_cache, steps={args.steps or cfg.num_inference_steps})"
            )
        else:
            if args.vision_bf16:
                apply_vision_bf16(policy)
            patcher.apply(
                sdpa=args.sdpa, cheap_cache=args.cheap_cache, steps=args.steps, compile=args.compile
            )
            label = "single-run"
        mean, samples, out, diff = bench(policy, batch, args, label, noise=noise, ref=None)
        print(f"{label:<52} {summary(samples)}")
        print("=" * 92)


if __name__ == "__main__":
    main()
