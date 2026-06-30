"""LeRobot async-inference compatibility shims.

Isolated so both the auto-launched policy-server child process
(``run-policy``) and the standalone ``inference-server`` apply the exact
same patch through one guarded code path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


def disable_observation_similarity_filter() -> None:
    """Stop ``PolicyServer`` from dropping observations as "too similar".

    Upstream's ``observations_similar`` filter skips any observation whose
    joint-space L2 distance from the previous one is under a **hardcoded**
    1-rad tolerance (``lerobot.async_inference.helpers``). On Axol's 16-DOF
    arms at 60 Hz consecutive observations are almost always within that
    bound, so the filter drops nearly every observation and starves the
    action queue.

    LeRobot exposes no public knob for this — the tolerance is a function
    default that ``PolicyServer`` never threads through ``PolicyServerConfig``
    — so the only fix without an upstream change is to neutralize the module
    symbol before ``serve`` runs. This is a deliberate private-API
    dependency; it is guarded so a LeRobot upgrade that renames or removes
    the symbol fails loudly here instead of silently re-enabling the filter.

    (The clean long-term fix is to upstream a ``similarity_atol`` /
    ``skip_similar_observations`` field on ``PolicyServerConfig``.)
    """
    from lerobot.async_inference import policy_server as ps

    if not hasattr(ps, "observations_similar"):
        raise RuntimeError(
            "lerobot.async_inference.policy_server no longer defines "
            "'observations_similar'; the Axol observation-filter workaround "
            "needs review against the new LeRobot version (otherwise the "
            "policy server may silently drop observations and starve the "
            "action queue)."
        )

    ps.observations_similar = lambda *args, **kwargs: False
    _logger.debug("Disabled PolicyServer observation-similarity filter.")


@dataclass
class RTCServerSettings:
    """Server-side Real-Time Chunking knobs applied when the policy loads.

    Mirrors the tunable fields of LeRobot's ``RTCConfig``. ``prefix_attention_schedule``
    is the lower-cased CLI spelling (``linear`` / ``exp`` / ``ones`` / ``zeros``);
    it is mapped to the ``RTCAttentionSchedule`` enum when the config is built.
    """

    enabled: bool = True
    execution_horizon: int = 40
    prefix_attention_schedule: str = "linear"
    max_guidance_weight: float = 10.0


def enable_rtc_on_policy_server(settings: RTCServerSettings) -> None:
    """Turn on Real-Time Chunking (RTC) guidance in LeRobot's ``PolicyServer``.

    RTC treats new-chunk generation as an inpainting problem: the first actions
    of a fresh chunk are guided toward the still-unexecuted tail of the previous
    chunk, so consecutive chunks are consistent by construction and the arm never
    jumps or stalls at a chunk boundary. The flow-matching guidance itself already
    ships in the vendored pi0/pi05 model (``RTCProcessor.denoise_step``); this
    function just *enables* it and wires its two inputs through the async
    transport, which otherwise drops them:

    1. **Enable** — after ``SendPolicyInstructions`` loads the policy, install an
       ``RTCConfig`` on it and (re)build its RTC processor. No-op for policies
       whose config has no ``rtc_config`` field (e.g. ACT).
    2. **Prefix cache (server side)** — the server caches each chunk keyed by its
       origin timestep, so the client only sends an ``inference_delay`` integer
       (Phase 2), never the prefix tensor, keeping the gRPC message small. What is
       cached depends on the policy's action space:
         * **Absolute actions** — the *model-space* chunk (pre-postprocessor /
           normalized) is fed straight back as the prefix; consecutive chunks
           share the same absolute frame, so no re-anchoring is needed.
         * **Relative (delta) actions** (e.g. this milano pi05 policy: delta arm
           joints, absolute grippers) — model output is relative to the *previous*
           observation's state, so the model-space tail is in the wrong frame for
           the new chunk. We instead cache the *absolute* (postprocessed) chunk and,
           after the preprocessor caches the new observation's state, re-express the
           overlapping tail relative to that state and re-normalize via
           ``reanchor_relative_rtc_prefix`` — mirroring LeRobot's own
           ``RTCInferenceEngine``.
    3. **Thread kwargs** — the unexecuted prefix (the slice of the previous chunk
       overlapping the new chunk's timesteps) is forwarded as
       ``prev_chunk_left_over`` alongside ``inference_delay`` into
       ``predict_action_chunk``. The prefix is built inside ``_get_action_chunk``
       (after the preprocessor has cached the current state) so relative
       re-anchoring uses the correct anchor.

    Guidance is applied only when the incoming observation carries an
    ``inference_delay`` *and* the previous chunk overlaps the new one (origin moved
    forward by less than a chunk). An episode reset rewinds the client's timestep
    counter, so the backwards/again-too-far origin jump naturally disables guidance
    on the first chunk of a new episode — no separate reset signal is needed.

    Like :func:`disable_observation_similarity_filter`, this is a deliberate
    private-API dependency on ``PolicyServer``'s method names, guarded so a LeRobot
    bump that renames them fails loudly here instead of silently disabling RTC.
    """
    if not settings.enabled:
        _logger.debug("RTC disabled by settings; PolicyServer left unpatched.")
        return

    from lerobot.async_inference import policy_server as ps

    required = ("SendPolicyInstructions", "_get_action_chunk", "_predict_action_chunk")
    missing = [name for name in required if not hasattr(ps.PolicyServer, name)]
    if missing:
        raise RuntimeError(
            "lerobot PolicyServer is missing method(s) "
            f"{missing}; the Axol RTC enablement patch needs review against the "
            "new LeRobot version (otherwise Real-Time Chunking guidance would be "
            "silently disabled and the arm would stop-start at chunk boundaries)."
        )

    import torch

    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.rtc import reanchor_relative_rtc_prefix
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.processor import (
        NormalizerProcessorStep,
        RelativeActionsProcessorStep,
    )

    try:
        schedule = RTCAttentionSchedule(settings.prefix_attention_schedule.upper())
    except ValueError as exc:
        valid = [s.value for s in RTCAttentionSchedule]
        raise ValueError(
            f"Unknown RTC prefix_attention_schedule "
            f"{settings.prefix_attention_schedule!r}; valid (case-insensitive): "
            f"{[v.lower() for v in valid]}."
        ) from exc

    orig_send_instructions = ps.PolicyServer.SendPolicyInstructions
    orig_predict_action_chunk = ps.PolicyServer._predict_action_chunk

    def _reset_rtc_cache(self) -> None:
        # Previous chunk in model space (absolute policies) and absolute /
        # post-processor space (relative policies, for re-anchoring), plus the
        # execution index it started at. ``None`` means "no usable prefix yet".
        self._rtc_last_chunk_model = None
        self._rtc_last_chunk_abs = None
        self._rtc_last_origin = None
        # Per-call info captured from the incoming TimedObservation.
        self._rtc_pending = {}
        # RelativeActionsProcessorStep / NormalizerProcessorStep introspected from
        # the preprocessor; non-None only for relative-action policies.
        self._rtc_relative_step = None
        self._rtc_normalizer_step = None

    def patched_send_instructions(self, request, context):  # noqa: N802
        result = orig_send_instructions(self, request, context)
        _reset_rtc_cache(self)
        policy = self.policy
        cfg = getattr(policy, "config", None)
        if cfg is None or not hasattr(cfg, "rtc_config"):
            _logger.warning(
                "Loaded policy %r has no rtc_config field; RTC guidance is "
                "unavailable for this policy type and the run will fall back to "
                "plain chunk replacement.",
                getattr(self, "policy_type", "?"),
            )
            return result
        cfg.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=settings.execution_horizon,
            prefix_attention_schedule=schedule,
            max_guidance_weight=settings.max_guidance_weight,
        )
        # Rebuild the processor now that rtc_config exists (the policy was loaded
        # with rtc_config=None, so its processor is currently None).
        if hasattr(policy, "init_rtc_processor"):
            policy.init_rtc_processor()

        # Detect relative (delta) actions so the prefix can be re-anchored to the
        # current state before guidance (see module docstring). Mirrors
        # lerobot.rollout.inference.rtc.RTCInferenceEngine.
        steps = list(getattr(getattr(self, "preprocessor", None), "steps", []) or [])
        rel = next(
            (
                s
                for s in steps
                if isinstance(s, RelativeActionsProcessorStep)
                and getattr(s, "enabled", False)
            ),
            None,
        )
        if rel is not None:
            if rel.action_names is None:
                cfg_names = getattr(cfg, "action_feature_names", None)
                if cfg_names:
                    rel.action_names = list(cfg_names)
                else:
                    _logger.warning(
                        "Relative-action policy has no action_feature_names; the "
                        "relative mask falls back to all-dims-relative, which would "
                        "wrongly make the grippers relative. RTC prefix re-anchoring "
                        "may be inaccurate."
                    )
            self._rtc_relative_step = rel
            self._rtc_normalizer_step = next(
                (s for s in steps if isinstance(s, NormalizerProcessorStep)), None
            )
            self.logger.info(
                "RTC relative-action re-anchoring enabled (exclude=%s).",
                getattr(rel, "exclude_joints", None),
            )
        self.logger.info(
            "RTC guidance enabled: schedule=%s execution_horizon=%d "
            "max_guidance_weight=%.1f",
            schedule.value,
            settings.execution_horizon,
            settings.max_guidance_weight,
        )
        return result

    def _rtc_enabled(self) -> bool:
        policy = getattr(self, "policy", None)
        return bool(
            policy is not None and getattr(policy, "_rtc_enabled", lambda: False)()
        )

    def _build_prefix_kwargs(self) -> dict:
        """Build ``{inference_delay, prev_chunk_left_over}`` for the current call.

        Called from ``_get_action_chunk`` — i.e. after the preprocessor has cached
        the current observation's state — so relative re-anchoring uses the right
        anchor. Returns ``{}`` (no guidance) on the first chunk, an episode reset
        (origin rewound), or no chunk overlap.
        """
        if not _rtc_enabled(self):
            return {}
        pending = getattr(self, "_rtc_pending", None) or {}
        inference_delay = pending.get("inference_delay")
        new_origin = pending.get("new_origin")
        if inference_delay is None or new_origin is None:
            return {}
        prev_origin = getattr(self, "_rtc_last_origin", None)
        if prev_origin is None:
            return {}
        offset = new_origin - prev_origin
        # offset < 0  -> episode reset (timestep rewound); guidance would pin the
        #                new episode to a stale chunk, so skip it.
        # offset >= chunk -> the chunks don't overlap; nothing to guide toward.
        if offset < 0 or offset >= self.actions_per_chunk:
            return {}

        rel = getattr(self, "_rtc_relative_step", None)
        if rel is not None:
            # Relative policy: re-anchor the *absolute* leftover to the current
            # state, then re-normalize, so the prefix is in the new chunk's frame.
            prev_abs = getattr(self, "_rtc_last_chunk_abs", None)
            if prev_abs is None:
                return {}
            leftover = prev_abs[offset:]
            if leftover.shape[0] == 0:
                return {}
            state = rel.get_cached_state()
            if state is None:
                return {}
            try:
                prefix = reanchor_relative_rtc_prefix(
                    prev_actions_absolute=leftover,
                    current_state=state,
                    relative_step=rel,
                    normalizer_step=getattr(self, "_rtc_normalizer_step", None),
                    policy_device=self.device,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "RTC relative prefix re-anchoring failed (%r); skipping "
                    "guidance this step.",
                    exc,
                )
                return {}
        else:
            # Absolute policy: the model-space tail is already in the right frame.
            prev_model = getattr(self, "_rtc_last_chunk_model", None)
            if prev_model is None:
                return {}
            leftover = prev_model[offset:]
            if leftover.shape[0] == 0:
                return {}
            prefix = leftover

        return {
            "inference_delay": int(inference_delay),
            "prev_chunk_left_over": prefix,
        }

    def patched_get_action_chunk(self, observation):
        kwargs = _build_prefix_kwargs(self)
        chunk = self.policy.predict_action_chunk(observation, **kwargs)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)
        chunk = chunk[:, : self.actions_per_chunk, :]
        # Cache the model-space (pre-postprocessor) chunk for the next absolute-policy
        # prefix. Clone so the downstream per-step postprocessor can't mutate it
        # through a shared view.
        self._rtc_last_chunk_model = chunk[0].detach().clone()
        return chunk

    def patched_predict_action_chunk(self, observation_t):
        self._rtc_pending = {
            "inference_delay": getattr(observation_t, "inference_delay", None),
            "new_origin": observation_t.get_timestep(),
        }
        try:
            result = orig_predict_action_chunk(self, observation_t)
        finally:
            # Record the origin of the chunk we just produced so the *next* call
            # can align its prefix. Runs even on error so the cache origin stays
            # consistent with the chunk caches.
            self._rtc_last_origin = observation_t.get_timestep()
            self._rtc_pending = {}
        # Cache the absolute (postprocessed) chunk for relative re-anchoring on the
        # next call. ``result`` is the list[TimedAction] of unnormalized, absolute
        # robot-space actions returned to the client.
        try:
            if result and hasattr(result[0], "get_action"):
                self._rtc_last_chunk_abs = torch.stack(
                    [ta.get_action() for ta in result]
                ).detach()
        except Exception:  # noqa: BLE001
            self._rtc_last_chunk_abs = None
        return result

    ps.PolicyServer.SendPolicyInstructions = patched_send_instructions
    ps.PolicyServer._get_action_chunk = patched_get_action_chunk
    ps.PolicyServer._predict_action_chunk = patched_predict_action_chunk
    _logger.debug("Enabled PolicyServer Real-Time Chunking guidance.")
