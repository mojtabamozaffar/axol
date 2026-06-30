# Plan: Real-Time Chunking (RTC) for `axol run-policy`

## Problem

`axol run-policy` runs the policy via LeRobot's **async inference** split: an
`AxolRobotClient` on the zed box streams observations over gRPC to a
`PolicyServer` on the GPU desktop, which returns action chunks. The client
keeps the robot moving by:

- prefetching a new chunk when the action queue drains to
  `chunk_size_threshold` (0.9), and
- blending overlapping in-flight chunks with a **temporal ensemble** (ACT
  Algorithm 2) at the chunk boundary.

This is latency-*tolerant*, not latency-free. The runway is
`0.9 * actions_per_chunk / fps` — at `actions_per_chunk=50`, `fps=60` that's
**~750 ms** of buffered motion. On an **A5000 the pi05 forward pass measures
~600 ms**, which (after the ~60–70 ms ZED read + gRPC each way) leaves almost
no margin: the queue periodically empties and the arm pauses on its last
commanded pose, producing visible **stop–start motion**.

Temporal ensembling hides the chunk-boundary *discontinuity* but does nothing
to protect against the queue genuinely draining. **RTC** is the correct fix: it
treats new-chunk generation as an inpainting problem, guiding the first
`inference_delay` actions of the new chunk toward the still-unexecuted tail of
the previous chunk, so consecutive chunks are consistent by construction and
the robot never has to stall or jump.

## Key finding: the RTC algorithm already ships in vendored LeRobot

The hard math is **already implemented** and is baked into the pi0/pi05 model
(which runs on the GPU desktop):

- `lerobot/policies/rtc/modeling_rtc.py` — `RTCProcessor.denoise_step()` applies
  prefix guidance during flow-matching denoising. It computes a correction via
  autograd through the denoiser
  (`torch.autograd.grad(x1_t, x_t, ...)`), weighted by a prefix-attention
  schedule (`get_prefix_weights`, LINEAR/EXP/ONES/ZEROS) and clamped by
  `max_guidance_weight`.
- `lerobot/policies/rtc/configuration_rtc.py` — `RTCConfig`
  (`enabled`, `prefix_attention_schedule`, `max_guidance_weight`,
  `execution_horizon`).
- `lerobot/policies/rtc/action_queue.py` — `ActionQueue.merge()` /
  `_replace_actions_queue()` implement the client-side queue replacement that
  accounts for `inference_delay` (the RTC alternative to temporal ensembling).
- `lerobot/policies/pi05/configuration_pi05.py` — pi05 config already carries
  `rtc_config: RTCConfig | None`.
- `lerobot/policies/pi05/modeling_pi05.py` — `predict_action_chunk(batch,
  **kwargs)` accepts `ActionSelectKwargs = {inference_delay,
  prev_chunk_left_over}` and forwards them into `model.sample_actions`, which
  calls `denoise_step` with RTC guidance when `_rtc_enabled()`.

**We are not implementing RTC — we are enabling it and wiring its inputs
through the async transport, which currently drops them.**

## Why this spans both machines

RTC needs two things that live on opposite sides of the gRPC link, so neither
machine can do it alone:

| Need | Lives on | Why |
|------|----------|-----|
| Denoiser + autograd guidance | **GPU desktop** (server) | Requires the model weights and gradients; this is `RTCProcessor.denoise_step`. |
| Unexecuted action prefix + measured latency | **Zed box** (client) | Only the client knows which actions it already committed and the real round-trip delay. |
| Carrying prefix + delay between them | **gRPC transport** (shared LeRobot `async_inference`) | The bridge that joins the two. |

## Current gap in the async path

Even with `rtc_config.enabled=True`, RTC stays dormant on the async path:

- `lerobot/async_inference/policy_server.py` — `_predict_action_chunk` →
  `_get_action_chunk(observation)` calls `predict_action_chunk(observation)`
  **with no RTC kwargs**, so `inference_delay` / `prev_chunk_left_over` are
  always `None` and guidance never fires.
- The gRPC `TimedObservation` carries the observation only — there is no field
  for the prefix tensor or the delay.
- `AxolRobotClient` (`almond_axol/cli/run_policy.py`) aggregates with
  `temporal_ensemble`, which is incompatible with RTC: RTC returns a single,
  already-stitched chunk that should *replace* the queue rather than be blended.

## Implementation plan

Three seams, plus config and validation. Build the instrumentation first so we
can measure the real `inference_delay` before committing to the integration.

### Phase 0 — Instrumentation (prerequisite, low risk)

Goal: measure the real round-trip delay and confirm we are actually starving
the queue, so `inference_delay` is grounded in data, not a guess.

- **Client** (`almond_axol/cli/run_policy.py`): the control loop already records
  `action_queue_size` every tick (`control_loop_action`). Log per-episode
  **min / mean queue depth**; a min of 0 confirms starvation.
- **Server**: `PolicyServer` already logs inference time — surface it (or echo
  it back to the client) so we can compute
  `inference_delay ≈ round_trip_s * fps` empirically.
- Deliverable: a short readout per episode, e.g.
  `queue depth min=0 mean=12.3 | server infer=0.61s | est delay=37 steps`.

### Phase 1 — Server (GPU desktop): enable RTC and pass kwargs through

File(s): `almond_axol/cli/inference_server.py` and a small patch to / wrapper
around `lerobot/async_inference/policy_server.py` (mirror the existing
`almond_axol/lerobot/inference_patch.py` monkeypatch style — see
`disable_observation_similarity_filter`).

1. Build/load the policy with `rtc_config` enabled and tunable
   (`execution_horizon`, `prefix_attention_schedule`, `max_guidance_weight`).
   Expose these as `inference-server` CLI flags.
2. In `PolicyServer._predict_action_chunk` / `_get_action_chunk`, extract
   `inference_delay` and `prev_chunk_left_over` from the incoming observation
   and forward them:
   `predict_action_chunk(obs, inference_delay=..., prev_chunk_left_over=...)`.
3. `prev_chunk_left_over` must be in the **model action space** the denoiser
   operates in (pre-postprocessor / normalized). Decide whether the client
   sends raw actions and the server re-normalizes, or the server caches the
   last emitted chunk pre-postprocess and the client only sends an index/count.
   **Preferred: server-side cache** — the server already produced the previous
   chunk in model space (`action_tensor` before the per-step postprocessor
   loop), so caching it avoids a normalization round-trip and a large tensor on
   the wire. The client then only needs to send how many of those actions it
   actually executed (the delay), and the server derives the leftover tail
   itself. (For relative-action policies, see
   `lerobot/policies/rtc/relative.py:reanchor_relative_rtc_prefix`.)

### Phase 2 — Transport (shared LeRobot async_inference)

Carry the minimal RTC state from client to server.

- Add `inference_delay: int` to `TimedObservation` (and the gRPC
  serialization). This is the only field strictly required if we adopt the
  server-side prefix cache from Phase 1.3.
- If we instead send the prefix explicitly, add
  `prev_chunk_left_over` (a `(T_prev, action_dim)` tensor) as well; prefer the
  cache to keep the message small.
- Keep this change isolated and feature-gated so non-RTC runs are byte-for-byte
  unchanged.

### Phase 3 — Client (zed box): produce inputs, swap the aggregator

File: `almond_axol/cli/run_policy.py` (`AxolRobotClient`).

1. Compute `inference_delay` = timesteps consumed during the round trip
   (`measured_round_trip_s * fps`), using the Phase 0 measurement. Smooth it
   (EMA) and/or use `lerobot/policies/rtc/latency_tracker.py` rather than a raw
   per-call value, so a single slow inference doesn't whipsaw the guidance.
2. Track the previously emitted chunk so the leftover tail is well-defined
   (or, with the server-side cache, just send the delay — see Phase 1.3).
3. Add an `aggregate_fn` value `"rtc"`. When selected:
   - **Bypass `_temporal_ensemble_aggregate`.** RTC returns one coherent chunk;
     do not blend overlapping chunks.
   - Adopt RTC queue semantics from
     `lerobot/policies/rtc/action_queue.py` (`merge` →
     `_replace_actions_queue`): on a new chunk, replace the queue from the
     execution point forward, accounting for `inference_delay`.
   - Preserve the existing gripper carve-out concern: confirm the inpainting
     guidance doesn't smear the bang-bang gripper channels
     (`_GRIPPER_INDICES = (7, 15)`); if it does, exclude those dims from the
     prefix weights.
4. Keep `temporal_ensemble` as the default; `--aggregate_fn rtc` opts in.

### Phase 4 — Validation

No hardware in CI, so validate in layers:

1. **Unit / offline**: feed a synthetic previous chunk + delay into
   `predict_action_chunk` on the server and assert the first `inference_delay`
   actions track the prefix (boundary continuity) while the tail stays free.
2. **Shadow mode** (`--shadow`): run RTC end-to-end with the arms floating in
   gravity comp; compare predicted-action continuity at chunk boundaries in
   Rerun (RTC vs. temporal_ensemble) without actuating.
3. **Powered**: with Phase 0 instrumentation, confirm the queue no longer hits
   0 and the stop–start motion is gone at the A5000's ~600 ms latency. Sweep
   `execution_horizon` and `prefix_attention_schedule`.

## Tuning notes / risks

- With ~600 ms latency at 60 Hz, `inference_delay ≈ 36` against a 50-step
  chunk — we are guiding most of the chunk. RTC quality degrades as
  `inference_delay / chunk_size → 1`. Mitigations, in parallel with RTC:
  - **Reduce inference time**: fewer flow-matching denoising steps, bf16,
    `torch.compile`/CUDA graphs on the server.
  - **Increase `actions_per_chunk`**: more runway and a smaller
    `inference_delay / chunk_size` ratio (at the cost of acting on staler
    observations).
- RTC's autograd-through-denoiser guidance adds per-denoise-step cost on the
  server; measure its impact on the ~600 ms budget (it may partially offset the
  benefit).
- Feature-gate everything behind `--aggregate_fn rtc` + server RTC flags so the
  current temporal-ensemble path remains the tested default.

## File reference index

- `almond_axol/cli/run_policy.py` — `AxolRobotClient` (control/observation
  loops, `_temporal_ensemble_aggregate`); client-side changes (Phases 0, 3).
- `almond_axol/cli/inference_server.py` — server entry point; RTC enable +
  flags (Phase 1).
- `almond_axol/lerobot/inference_patch.py` — existing monkeypatch pattern to
  mirror for server-side kwarg pass-through.
- `lerobot/async_inference/policy_server.py` — `_predict_action_chunk` /
  `_get_action_chunk` (Phase 1) and `TimedObservation` transport (Phase 2).
- `lerobot/policies/rtc/` — `modeling_rtc.py` (guidance, server),
  `action_queue.py` (queue merge, client), `latency_tracker.py` (delay
  estimation, client), `relative.py` (relative-action prefix re-anchoring).
- `lerobot/policies/pi05/{configuration_pi05,modeling_pi05}.py` — `rtc_config`
  and `predict_action_chunk(**ActionSelectKwargs)`.
