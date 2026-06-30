"""A5000 latency optimizations for pi05 policies served by ``axol inference-server``.

The pi05 forward (Gemma-2B VLM prefill + Gemma-300M flow-matching action expert)
measures ~494 ms on an RTX A5000 with the milano_side checkpoint, dominated by a
~300 ms / 10-step denoise loop that is almost pure kernel-launch overhead, a
~110 ms **fp32** SigLIP vision encode of 4 camera views, and a ~75 ms prefill.

Measured deltas (``almond_axol.diagnostics.bench_pi05``, RTX A5000, batch=1):

    baseline (10 steps)                              ~494 ms   1.00x
    + cheap_kv_cache   (lossless, bit-identical)     ~480 ms   1.03x   max|Δ|=0
    + vision_bf16      (max|Δ|≈1e-2 vs fp32 vision)  ~406 ms   1.22x
    + num_inference_steps 10→6 (max|Δ|≈5e-2)         ~300 ms   1.65x   (server default)
    + num_inference_steps 10→5 (max|Δ|≈8e-2)         ~284 ms   1.74x

``cheap_kv_cache`` and ``vision_bf16`` are accuracy-safe (the rest of the model
is already bf16) and default **on**. ``num_inference_steps`` trades flow-matching
fidelity for latency: this module's default (``None``) keeps the checkpoint's
value, but ``axol inference-server`` defaults it to 6 (≈300 ms) — validate any
reduction in shadow mode before a powered run. ``compile`` (CUDA graphs over the
denoise loop) is opt-in and adds first-inference warm-up latency. ``sdpa`` was
measured a no-op on these sequence lengths *and* perturbs the output, so it is
intentionally not wired.

These are runtime monkeypatches (LeRobot is pip-installed, not part of the fork),
mirroring ``inference_patch.disable_observation_similarity_filter``: each is
guarded so a LeRobot upgrade that moves the ground under us fails loudly here.
All patches are pi05-scoped, so a server asked to serve another policy type is
unaffected.
"""

from __future__ import annotations

import copy as _copy
import inspect
import logging
import types

_logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1. Lossless KV-cache clone — replaces the per-denoise-step deepcopy
# --------------------------------------------------------------------------- #
def cheap_clone_cache(cache):
    """Shallow *container* copy of a transformers ``DynamicCache``.

    pi05's ``denoise_step`` calls ``copy.deepcopy(past_key_values)`` every
    denoise step because the expert forward appends its 50 suffix tokens to the
    prefix cache (``DynamicLayer.update`` does ``self.keys = cat([self.keys,
    k])``); without a copy the canonical prefix grows 1274→1324→… and the next
    step mismatches. Deepcopy pays for that safety by cloning all 36 prefix KV
    tensors (~23 MB) on every step. We instead duplicate only the *container*
    objects — the appended ``cat`` lands on a fresh layer reference while the
    prefix tensors stay shared — which is bit-identical (verified max|Δ|=0) and
    drops the clone.
    """
    new = _copy.copy(cache)
    if hasattr(cache, "layers"):  # transformers >=5 DynamicCache
        new.layers = [_copy.copy(layer) for layer in cache.layers]
    elif hasattr(cache, "key_cache"):  # legacy tuple-list layout
        new.key_cache = list(cache.key_cache)
        new.value_cache = list(cache.value_cache)
    return new


class _CheapCopyProxy:
    """Stands in for the ``copy`` module *inside modeling_pi05 only*.

    ``deepcopy`` is redirected to :func:`cheap_clone_cache`; everything else
    delegates to the real ``copy`` module, so the rest of the module is
    unaffected and we never mutate the shared stdlib module.
    """

    def __init__(self, real):
        self.__dict__["_real"] = real

    def deepcopy(self, x):
        return cheap_clone_cache(x)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _patch_cheap_kv_cache() -> None:
    from lerobot.policies.pi05 import modeling_pi05

    src = inspect.getsource(modeling_pi05.PI05Pytorch.denoise_step)
    if "copy.deepcopy(past_key_values)" not in src:
        raise RuntimeError(
            "pi05 denoise_step no longer deepcopies past_key_values; the cheap "
            "KV-cache optimization in almond_axol.lerobot.pi05_inference_opt needs "
            "review against this LeRobot version (otherwise inference is silently "
            "un-optimized or, worse, the cache is corrupted)."
        )
    if isinstance(modeling_pi05.copy, _CheapCopyProxy):
        return
    modeling_pi05.copy = _CheapCopyProxy(modeling_pi05.copy)
    _logger.info("pi05: per-denoise-step KV-cache deepcopy → shallow clone (lossless).")


# --------------------------------------------------------------------------- #
# 2. bf16 vision tower
# --------------------------------------------------------------------------- #
def apply_vision_bf16(policy) -> None:
    """Run the SigLIP vision tower + projector in bf16 instead of fp32.

    pi05 keeps the vision path fp32 at *train* time only to dodge an optimizer
    dtype toggle; at inference there is no optimizer and the rest of the model is
    bf16 already, so bf16 vision halves the encode's memory traffic and uses the
    bf16 tensor cores (~70 ms saved on 4 views). ``embed_image`` hard-casts to
    fp32, so it is replaced with a bf16 variant.
    """
    import torch

    pwe = policy.model.paligemma_with_expert
    pwe.paligemma.model.vision_tower.to(torch.bfloat16)
    pwe.paligemma.model.multi_modal_projector.to(torch.bfloat16)
    scale = pwe.paligemma.config.text_config.hidden_size**0.5

    def embed_image_bf16(self, image):
        feats = self.paligemma.model.get_image_features(image.to(torch.bfloat16))
        return feats.pooler_output * scale

    pwe.embed_image = types.MethodType(embed_image_bf16, pwe)
    _logger.info("pi05: vision tower + projector cast to bf16.")


# --------------------------------------------------------------------------- #
# 3. Per-policy hooks applied at load (policy loads lazily on client connect)
# --------------------------------------------------------------------------- #
def _wrap_from_pretrained(
    *, vision_bf16: bool, num_inference_steps: int | None, compile: bool
) -> None:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    if getattr(PI05Policy.from_pretrained, "_axol_optimized", False):
        return
    orig = PI05Policy.from_pretrained.__func__

    def wrapped(cls, *args, **kwargs):
        policy = orig(cls, *args, **kwargs)
        if num_inference_steps is not None:
            policy.config.num_inference_steps = int(num_inference_steps)
            _logger.info("pi05: num_inference_steps → %d", num_inference_steps)
        if vision_bf16:
            apply_vision_bf16(policy)
        if compile:
            import torch

            policy.model.sample_actions = torch.compile(
                policy.model.sample_actions, mode="reduce-overhead"
            )
            _logger.info("pi05: torch.compile(sample_actions, reduce-overhead) enabled.")
        return policy

    wrapped._axol_optimized = True
    PI05Policy.from_pretrained = classmethod(wrapped)


# --------------------------------------------------------------------------- #
# Entry point — called by `axol inference-server` before `serve()`
# --------------------------------------------------------------------------- #
def enable_pi05_inference_optimizations(
    *,
    vision_bf16: bool = True,
    cheap_kv_cache: bool = True,
    num_inference_steps: int | None = None,
    compile: bool = False,
) -> None:
    """Install the pi05 latency optimizations on the policy-load path.

    Global/class-level patches (``cheap_kv_cache``) take effect immediately;
    per-policy patches (``vision_bf16``, ``num_inference_steps``, ``compile``)
    are applied when the server loads the policy on client connect.
    """
    if cheap_kv_cache:
        _patch_cheap_kv_cache()
    _wrap_from_pretrained(
        vision_bf16=vision_bf16,
        num_inference_steps=num_inference_steps,
        compile=compile,
    )
    _logger.info(
        "pi05 inference optimizations armed (vision_bf16=%s, cheap_kv_cache=%s, "
        "num_inference_steps=%s, compile=%s).",
        vision_bf16,
        cheap_kv_cache,
        num_inference_steps,
        compile,
    )
