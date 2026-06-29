# Handoff — `axol run-policy --shadow` (no-actuate shadow mode)

This document describes a small feature added to **axol** (`run-policy`) to de-risk the
**first deployment** of a trained policy on the physical robot, and how to replicate +
use it on the **Zed box**. It ships with `shadow_mode.patch` in this folder.

---

## 1. What it is

A new flag, **`axol run-policy --shadow`**, that runs the *entire* real policy pipeline —
cameras → observations → (offloaded) inference → action chunks → aggregation → Rerun —
**but never moves the robot.** The policy's predicted actions are streamed to a Rerun
viewer instead of being sent to the motors.

It is the answer to: *"What would this policy actually do on this exact scene, right now,
before I let it touch the arms?"*

## 2. Why it exists / when to use it

A first powered run of a never-deployed policy is the highest-risk moment. Shadow mode is
the **go/no-go gate** in the safe-deployment sequence (see §6): you confirm — on live, real
observations — that the 4 camera streams are correctly keyed, the policy loads and infers,
and its intended motions look sane, **with zero possibility of motion**, before any
powered run.

Use it:
- The first time a given checkpoint is run on the robot.
- After any change to cameras, wiring, the serving stack, or the checkpoint.
- To sanity-check language conditioning (does the predicted motion change with `--task`?).

## 3. What it does to behavior (the safety contract)

When `--shadow` is set:
- **No actuation, ever.** Every call that could move the arms is gated:
  - the policy action in the control loop (`send_action` is skipped), and
  - the between-episode **return-to-rest** trajectory (all three call sites: initial,
    re-record `r`, and save `s`).
  So episode transitions don't move the arms either.
- **Predicted actions are visualized.** The policy's intended joint targets are logged to
  Rerun under the **`action.predicted.*`** namespace, alongside the current joints under
  **`observation.state`** — so you can compare "what the policy wants" vs. "where the arm
  is" on one timeline.
- **Viz-only, no dataset.** `--repo_id` is ignored (a hand-guided observation paired with
  an *unexecuted* prediction is not a meaningful rollout).
- Everything else is identical to a normal run: offloaded inference, the `s`/`r`/`q`
  episode controls, and the `--episode_time_s` safety cap all behave as usual.

Topology is unchanged: with `--server_host`, inference still runs on the GPU desktop's
`axol inference-server`; `--shadow` only changes what the **robot side** does with the
returned actions.

## 4. Files changed

The feature is two functional files (in `shadow_mode.patch`) plus one optional docs file:

| File | Change |
|---|---|
| `almond_axol/cli/run_policy.py` | `RunPolicyConfig.shadow` flag; thread it through the client builder + `AxolRobotClient`; gate `send_action` in `control_loop_action`; gate the 3 `return_to_rest` calls; force viz-only (ignore `--repo_id`); run the capture thread in shadow even without a dataset. |
| `almond_axol/lerobot/rollout.py` | `RolloutCaptureThread` accepts `dataset=None` (skip recording) + a `shadow` flag; logs the predicted action under `action.predicted.*`. |
| `docs/cli/run-policy.mdx` | *(optional, cosmetic)* a `--shadow` row in the flag table. |

**Built against axol commit `aeadfc54`** ("Add --joints filter to can.receive diagnostic
(#93)", 2026-06-27).

## 5. Replicate on the Zed box

```bash
# 0. Confirm the Zed box axol version matches (clean apply needs aeadfc54).
cd <axol-repo-on-zedbox> && git rev-parse HEAD

# locate the repo if unsure:
#   python -c "import almond_axol,os;print(os.path.dirname(os.path.dirname(almond_axol.__file__)))"

# 1. Copy the patch over (from the GPU desktop; cable IP of the Zed box is 192.168.50.2):
#   scp experiments/pi05_eval/shadow_mode.patch user@192.168.50.2:/tmp/

# 2. Apply it:
git apply --check /tmp/shadow_mode.patch     # dry-run; no output = clean
git apply        /tmp/shadow_mode.patch

# 3. Verify:
python -m py_compile almond_axol/cli/run_policy.py almond_axol/lerobot/rollout.py
axol run-policy --help | grep -i shadow      # the flag should appear
```

**If the Zed box is at a different axol version** (or it's not a git checkout): apply the
changes by hand from the diff in `shadow_mode.patch`. It's 11 small, independent edits,
each anchored on a distinctive line — the load-bearing ones are the `shadow` config field,
the `if self._shadow:` gate at the `send_action` line, the three `return_to_rest` guards,
and the `RolloutCaptureThread` dataset/shadow handling; the rest is plumbing the argument
through.

## 6. How to use it (worked example: `milano_side`)

**Prerequisites**
- GPU-desktop `axol inference-server` is up on `:8765` (it loads the policy; needs the
  PaliGemma HF token resolved on that machine).
- The cable is up: GPU desktop `192.168.50.1` ↔ Zed box `192.168.50.2`.
- A Rerun viewer is running **on the GPU desktop** (where you watch), listening for the
  robot:
  ```bash
  rerun --bind 0.0.0.0 --port 9876
  ```

**Run shadow mode (on the Zed box)**
```bash
axol run-policy --shadow \
  --policy_path /home/eevee/axol_eval_ckpts/milano_side/last/pretrained_model \
  --policy_type pi05 \
  --task "milano_side_placement" \
  --server_host 192.168.50.1 --server_port 8765 --fps 60 \
  --robot_config.cameras "{left_arm: {serial: 50613840, stereo: true, eyes: both, width: 1920, height: 1200}, right_arm: {serial: 53186417, stereo: true, eyes: both, width: 1920, height: 1200}}" \
  --robot_config.axol_config.left.wrist_3.mass 0.9 \
  --robot_config.axol_config.right.wrist_3.mass 0.9 \
  --rerun_ip 192.168.50.1 --episode_time_s 20
```
(`--policy_path` is the path **on the GPU desktop** — the server loads it from its own disk.)

**What to expect / how to read it**
- The arms **do not move**. The policy keeps inferring on the live scene.
- Hand-guide the compliant arms through the task. In Rerun, watch `action.predicted.*`
  (the policy's intended joint targets) move toward the goal and compare against
  `observation.state` (current joints). Confirm the predictions are sensible, not
  constant / NaN / wildly out of range. Note: predictions lag the scene by the inference
  latency (~tens of ms) — expected.
- `s` / `r` / `q` and `--episode_time_s` work as usual; nothing is recorded.

## 7. Acceptance test (do this once, before trusting it)

With the motors **powered**: run the command above, let it infer for a while, and press
`r` to cycle an episode. **Confirm the arms never move** — neither during inference nor on
the episode transition. That proves the gate. Only then proceed to a powered run.

## 8. Safe-deployment sequence (shadow mode is step 4)

1. Health, no motion: `axol motor-health`, `axol gravity-comp`, ZED serial listing.
2. Contract, no robot: the launcher dry-run preflight (`run_axol_eval.py run` without
   `--execute`).
3. Known-good replay (powered, no policy): `axol replay-dataset` of a trusted teleop
   episode, `--rerun_ip`, `stiffness 0` first.
4. **Shadow mode** (this feature) — the go/no-go gate.
5. First live run: drop `--shadow`, add guardrail dials, `--rerun_ip`, hand on `q`:
   `--robot_config.axol_config.max_step_rad 0.1`,
   `--robot_config.axol_config.left_stiffness 0.2 --robot_config.axol_config.right_stiffness 0.2`,
   short `--episode_time_s`.
6. Escalate one dial per run, reviewing Rerun each time.

## 9. Limitations / caveats

- **Not an e-stop replacement.** There is no software e-stop in `run-policy`; the aborts are
  the `q` key, a short `--episode_time_s`, and the physical e-stop. Shadow mode removes
  motion risk only because it never actuates.
- **Server dependency.** Shadow still needs the GPU-desktop `inference-server` up (and its
  HF/PaliGemma access) — it exercises the *real* serving path on purpose.
- **Version match.** The patch applies cleanly only at axol `aeadfc54`; otherwise apply by
  hand (§5).
- **Comparison is qualitative.** Shadow shows *predicted vs. current*, not a quantitative
  human-vs-policy score on a moving robot (that would be the deferred VR-teleop variant).

## 10. Pointers

- Patch: `experiments/pi05_eval/shadow_mode.patch`
- Operator runbook (camera/task/wiring details, escalation): `experiments/pi05_eval/README.md` (§3.6)
- axol CLI reference: `third_party/axol/docs/cli/run-policy.mdx` (the `--shadow` row)
