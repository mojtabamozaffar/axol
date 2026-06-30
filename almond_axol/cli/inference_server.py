"""
axol inference-server

Serve policy inference for ``axol run-policy --server_host <this machine>``.

Runs LeRobot's async-inference ``PolicyServer`` in the foreground on a more
powerful machine (e.g. a desktop with a discrete GPU) on the same network as
the robot. The robot streams joint positions + camera frames to it over gRPC
and receives action chunks back; the policy itself (``--policy_path`` /
``--policy_type`` / ``--device``) is selected by the *client*, so one server
can serve different policies across sessions without restarting.

    axol inference-server                 # listen on 0.0.0.0:8765
    axol inference-server --port 9000

Then, on the robot:

    axol run-policy --server_host <server-ip> ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import LogLevel, RTCSchedule, parse

_logger = logging.getLogger(__name__)


@dataclass
class InferenceServerConfig:
    """Config for ``axol inference-server``.

    Real-Time Chunking (RTC) is on by default: the server guides the first
    actions of each new chunk toward the still-unexecuted tail of the previous
    one, so chunks are continuous by construction and the arm never stop-starts
    at a boundary. It activates only for pi0/pi05 policies and only once the
    client (``run-policy --aggregate_fn rtc``, the default) starts sending the
    measured ``inference_delay``; other policies / aggregators are unaffected.

    Args:
        host:      Interface to bind the gRPC server to. The default
                   (0.0.0.0) accepts connections from the whole network.
        port:      gRPC port (must match run-policy's ``--server_port``).
        fps:       Action chunk rate; must match run-policy's ``--fps``.
        rtc:       Enable RTC guidance (default True). Set False to serve plain
                   chunks (the client then falls back to queue replacement with
                   no inpainting guidance).
        rtc_execution_horizon: Number of leading chunk steps that receive prefix
                   guidance. Should sit at or just above the measured
                   ``inference_delay`` (≈ round_trip_s * fps) so the executed
                   portion of the chunk is guided for a smooth handoff, while
                   leaving a fresh tail that can still react to new observations.
                   Auto-capped to the available prefix overlap each step.
        rtc_prefix_attention_schedule: How prefix guidance decays across the
                   horizon (linear / exp / ones / zeros). ``linear`` is the
                   smoothest and the upstream default.
        rtc_max_guidance_weight: Upper clamp on the per-step guidance weight.
        log_level: Python logging level.

    pi05 latency knobs (see ``almond_axol.lerobot.pi05_inference_opt`` and
    ``python -m almond_axol.diagnostics.bench_pi05``). On an RTX A5000 the pi05
    forward is ~494 ms; the defaults below bring it to ~300 ms (median 297,
    1.65x). They apply only when the client serves a pi05 policy; other policy
    types are untouched.

        vision_bf16:         Run the SigLIP vision tower in bf16 (≈-70 ms,
                             max|Δ|≈1e-2 vs fp32; rest of the model is bf16).
        cheap_kv_cache:      Replace pi05's per-denoise-step KV-cache deepcopy
                             with a shallow clone (≈-13 ms, bit-identical).
        num_inference_steps: Flow-matching denoise steps (default 6; the
                             checkpoint trained at 10). Lower = faster but
                             coarser — 6 trades max|Δ|≈5e-2 vs 10 for ~90 ms; set
                             10 for trained fidelity, 5 (~284 ms) for more speed.
                             Validate a reduction in ``run-policy --shadow``
                             before a powered run.
        compile:             torch.compile (CUDA graphs) the denoise loop. Cuts
                             per-step launch overhead but adds first-inference
                             warm-up; opt-in.
    """

    host: str = "0.0.0.0"
    port: int = 8765
    fps: int = 60
    rtc: bool = True
    rtc_execution_horizon: int = 40
    rtc_prefix_attention_schedule: RTCSchedule = "linear"
    rtc_max_guidance_weight: float = 10.0
    log_level: LogLevel = "INFO"
    vision_bf16: bool = True
    cheap_kv_cache: bool = True
    num_inference_steps: int | None = 6
    compile: bool = False


def main(argv: list[str]) -> None:
    """Parse the CLI config and serve policy inference until Ctrl+C."""
    cfg = parse(InferenceServerConfig, argv)
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)

    from ..lerobot.inference_patch import (
        RTCServerSettings,
        disable_observation_similarity_filter,
        enable_rtc_on_policy_server,
    )

    disable_observation_similarity_filter()
    enable_rtc_on_policy_server(
        RTCServerSettings(
            enabled=cfg.rtc,
            execution_horizon=cfg.rtc_execution_horizon,
            prefix_attention_schedule=cfg.rtc_prefix_attention_schedule,
            max_guidance_weight=cfg.rtc_max_guidance_weight,
        )
    )

    from ..lerobot.pi05_inference_opt import enable_pi05_inference_optimizations

    enable_pi05_inference_optimizations(
        vision_bf16=cfg.vision_bf16,
        cheap_kv_cache=cfg.cheap_kv_cache,
        num_inference_steps=cfg.num_inference_steps,
        compile=cfg.compile,
    )

    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import serve

    from ..utils.ports import reclaim_port

    # The gRPC port is fixed (it must match run-policy's ``--server_port``), so
    # evict a leftover server from a crashed/previous run rather than failing to
    # bind. lerobot owns the socket once ``serve`` takes over.
    reclaim_port(cfg.port)

    _logger.info("Serving policy inference on %s:%d (Ctrl+C to stop).", cfg.host, cfg.port)
    try:
        serve(PolicyServerConfig(host=cfg.host, port=cfg.port, fps=cfg.fps))
    except KeyboardInterrupt:
        _logger.info("Inference server stopped.")
