"""
Shared rollout machinery for policy CLIs.

Pulled out of ``axol run-policy`` so other policy-running CLIs can reuse
the same episode plumbing without duplicating it:

- :class:`IKResetController` — collision-aware return-to-rest backed by
  an out-of-process JAX IK worker.
- :class:`ActionPublisher` — single-slot thread-safe handoff of the most
  recently executed action.
- :class:`RolloutCaptureThread` — fixed-rate thread that pairs a
  timestamp-aligned observation with the latest published action and
  appends it to a ``LeRobotDataset``.
- :class:`ShadowGravityCompThread` — fixed-rate thread that holds the arms
  in gravity compensation so they float and stay hand-guidable during a
  shadow-mode run (which never actuates via ``send_action``).
- :func:`stdin_watcher` — ``s`` / ``r`` / ``q`` keystroke watcher with
  no-block ``select`` polling.

All four are LeRobot-flavoured: the capture thread depends on
``lerobot.datasets.lerobot_dataset.LeRobotDataset``, ``build_dataset_frame``,
and ``log_rerun_data``; the reset controller talks to the JAX IK worker via
``almond_axol.teleop``. The module lives under ``almond_axol/lerobot``
alongside the other LeRobot adapters.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

from ..constants import ARM_JOINTS

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.types import RobotAction

    from .rerun_robot import AxolRerunRobot
    from .robot.robot_axol import AxolRobot

_logger = logging.getLogger(__name__)


class IKResetController:
    """Collision-aware return-to-rest, backed by an IK worker subprocess.

    Mirrors the reset path used by ``AxolVRTeleop`` (collect-data) but
    without the VR server. ``start()`` spawns ``run_ik_worker`` (JAX +
    JITed solver, ~10-20 s); ``wait_ready()`` blocks on the handshake;
    ``return_to_rest()`` plans a joint-space trajectory and streams its
    waypoints to the impedance controller. Spawn before ``client.start()``
    so the IK JIT overlaps with the policy load.
    """

    def __init__(
        self,
        *,
        rest_pose_left: list[float] | None = None,
        rest_pose_right: list[float] | None = None,
    ) -> None:
        import numpy as np

        from ..kinematics.config import KinematicsConfig
        from ..teleop.config import VRTeleopConfig

        # Start from the teleop defaults, then override the rest pose if the
        # caller supplied one (e.g. ``run-policy --rest_pose_left/right``), so
        # powered resets land in the same pose the demos were collected from.
        # The IK worker reads these off ``vr_cfg`` and pre-settles them to a
        # manipulability-balanced fixed point, exactly as in teleop/collect-data.
        self._vr_cfg = VRTeleopConfig()
        for name, pose in (
            ("rest_pose_left", rest_pose_left),
            ("rest_pose_right", rest_pose_right),
        ):
            if pose is None:
                continue
            if len(pose) != len(ARM_JOINTS):
                raise ValueError(
                    f"{name} must have {len(ARM_JOINTS)} elements "
                    f"(one per arm joint), got {len(pose)}"
                )
            setattr(self._vr_cfg, name, np.asarray(pose, dtype=np.float32))
        self._kin_cfg = KinematicsConfig()
        self._proc: Any | None = None
        self._conn: Any | None = None
        self._q_init: Any | None = None
        self._left_indices: list[int] | None = None
        self._right_indices: list[int] | None = None
        self._ready = False

    def start(self) -> None:
        """Spawn the IK worker subprocess. Non-blocking; pair with ``wait_ready``."""
        import multiprocessing as mp

        from ..teleop.worker import run_ik_worker

        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=run_ik_worker,
            args=(child_conn, self._vr_cfg, self._kin_cfg, None, None),
            name="axol-ik-worker",
            daemon=True,
        )
        proc.start()
        child_conn.close()
        self._proc = proc
        self._conn = parent_conn

    def wait_ready(self, timeout: float = 60.0) -> None:
        """Block until the IK worker has finished JIT compilation."""
        if self._ready:
            return
        if self._conn is None:
            raise RuntimeError("IK reset controller not started")
        if not self._conn.poll(timeout):
            raise TimeoutError(
                f"IK worker did not become ready within {timeout:.1f}s "
                "(JAX JIT compilation may have stalled)."
            )
        msg = self._conn.recv()
        if not (isinstance(msg, tuple) and msg[0] == "ready"):
            raise RuntimeError(f"Unexpected IK worker handshake: {msg!r}")
        import numpy as np

        _, q_init, left_indices, right_indices, _startup_traj = msg
        self._q_init = np.asarray(q_init, dtype=np.float32)
        self._left_indices = [int(i) for i in left_indices]
        self._right_indices = [int(i) for i in right_indices]
        self._ready = True

    def return_to_rest(self, robot: "AxolRobot") -> None:
        """Plan and play a collision-aware trajectory to the rest pose."""
        self._play_to_target(robot, target="rest")

    def return_to_zero(self, robot: "AxolRobot") -> None:
        """Plan and play a collision-aware trajectory to the all-zeros arm pose.

        Mirrors collect-data's end-of-session return-to-zero: the arm joints
        are eased down to 0 (grippers left where they are) via the same
        collision-aware IK trajectory as ``return_to_rest``, so the arms
        descend smoothly instead of dropping the instant the motors release.
        """
        self._play_to_target(robot, target="zero")

    def _play_to_target(self, robot: "AxolRobot", *, target: str) -> None:
        """Plan and stream a collision-aware reset trajectory to ``target``.

        ``target`` is ``"rest"`` (the worker's configured rest pose) or
        ``"zero"`` (the current pose with both arms' joints zeroed; grippers
        keep their current value).
        """
        import numpy as np

        from ..constants import Joint
        from ..teleop.filter import ResetInterpolator

        self.wait_ready()
        assert self._conn is not None
        assert self._q_init is not None
        assert self._left_indices is not None
        assert self._right_indices is not None

        pos_l, pos_r = robot.positions
        pos_l = np.asarray(pos_l, dtype=np.float32)
        pos_r = np.asarray(pos_r, dtype=np.float32)

        q_current = self._q_init.copy()
        for i, gi in enumerate(self._left_indices):
            q_current[gi] = float(pos_l[i])
        for i, gi in enumerate(self._right_indices):
            q_current[gi] = float(pos_r[i])

        if target == "zero":
            # Explicit zero-arm target: the worker plans a collision-aware path
            # to it (the 3-tuple "reset" form). Grippers stay where they are.
            q_target = q_current.copy()
            q_target[self._left_indices] = 0.0
            q_target[self._right_indices] = 0.0
            self._conn.send(("reset", q_current, q_target))
        else:
            # 2-tuple form: the worker returns to its configured rest pose.
            self._conn.send(("reset", q_current))
        result = self._conn.recv()
        if not (isinstance(result, tuple) and result[0] == "reset_traj"):
            raise RuntimeError(f"Unexpected IK worker response: {result!r}")
        _, _q_rest, traj = result
        if not traj:
            _logger.warning("IK worker returned an empty reset trajectory; skipping.")
            return

        interp = ResetInterpolator()
        interp.set_trajectory(traj, float(pos_l[7]), float(pos_r[7]))

        joints = list(Joint)
        play_hz = float(self._vr_cfg.frequency)
        period = 1.0 / play_hz
        while interp.is_active():
            t0 = time.perf_counter()
            new_q, l_grip, r_grip, _done = interp.step()
            if new_q is None:
                break
            arm_left = np.asarray(new_q)[self._left_indices]
            arm_right = np.asarray(new_q)[self._right_indices]
            action: dict[str, float] = {}
            for j in joints:
                if j in ARM_JOINTS:
                    ai = ARM_JOINTS.index(j)
                    action[f"left_{j.value}.pos"] = float(arm_left[ai])
                    action[f"right_{j.value}.pos"] = float(arm_right[ai])
                else:
                    action[f"left_{j.value}.pos"] = float(l_grip)
                    action[f"right_{j.value}.pos"] = float(r_grip)
            robot.send_action(action)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def stop(self) -> None:
        """Signal shutdown, close the pipe, and reap the subprocess."""
        if self._conn is not None:
            try:
                self._conn.send(None)
            except Exception:  # noqa: BLE001
                pass
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None
        if self._proc is not None:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.kill()
            self._proc = None


class ActionPublisher:
    """Thread-safe single-slot publisher for the most recently executed action.

    Updated by the control loop after every ``robot.send_action`` call,
    read by :class:`RolloutCaptureThread` to pair each dataset frame with
    the action that drove the robot at that tick.

    Also carries the most recent *predicted action chunk* (the policy's raw
    next-N prediction), versioned so the capture thread can redraw the 3D
    future-trajectory path only when a fresh chunk arrives instead of every
    tick. The chunk is independent of the per-tick ``latest`` action.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: "RobotAction | None" = None
        self._first_event = threading.Event()
        self._chunk: "list[RobotAction] | None" = None
        self._chunk_version = 0

    def publish(self, action: "RobotAction") -> None:
        snap = dict(action)
        with self._lock:
            self._latest = snap
        self._first_event.set()

    def latest(self) -> "RobotAction | None":
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def publish_chunk(self, chunk: "list[RobotAction]") -> None:
        """Store the latest predicted action chunk (for the 3D path viz)."""
        snap = [dict(a) for a in chunk]
        with self._lock:
            self._chunk = snap
            self._chunk_version += 1

    def latest_chunk(self) -> "tuple[int, list[RobotAction] | None]":
        """Return ``(version, chunk_copy)``; version changes on each new chunk."""
        with self._lock:
            if self._chunk is None:
                return self._chunk_version, None
            return self._chunk_version, [dict(a) for a in self._chunk]

    def wait_for_first(self, timeout: float) -> bool:
        return self._first_event.wait(timeout=timeout)

    def reset(self) -> None:
        with self._lock:
            self._latest = None
            self._chunk = None
        self._first_event.clear()


class RolloutCaptureThread(threading.Thread):
    """Tick at ``fps`` Hz and append one ``(obs, action)`` row per tick.

    Each tick samples a global-timestamp-aligned observation via
    ``AxolRobot.get_observation`` and pairs it with the latest action
    published by the control loop.

    ``dataset`` may be ``None`` (shadow mode): the row is then only streamed
    to Rerun, never recorded. When ``shadow`` is set the published action is
    the policy's *predicted* (unexecuted) action, logged under a distinct
    ``predicted.*`` namespace to set it apart from an executed-action run.

    ``robot_viz`` (optional) adds a 3D task-space view to the Rerun stream: the
    live robot posed by forward kinematics from the current joint state, plus
    the predicted/commanded end-effector pose per arm. It only fires when
    ``rerun_ip`` is set.
    """

    def __init__(
        self,
        *,
        publisher: ActionPublisher,
        robot: "AxolRobot",
        dataset: "LeRobotDataset | None",
        robot_obs_proc: Callable[[Any], Any],
        fps: int,
        task: str,
        rerun_ip: str | None,
        shadow: bool = False,
        robot_viz: "AxolRerunRobot | None" = None,
    ) -> None:
        super().__init__(name="axol-rollout-capture", daemon=True)
        self.publisher = publisher
        self.robot = robot
        self.dataset = dataset
        self.robot_obs_proc = robot_obs_proc
        self.fps = fps
        self.task = task
        self.rerun_ip = rerun_ip
        self.shadow = shadow
        self.robot_viz = robot_viz
        self._last_chunk_version = -1
        self.stop_event = threading.Event()

    def run(self) -> None:
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame
        from lerobot.utils.visualization_utils import log_rerun_data

        if not self.publisher.wait_for_first(timeout=10.0):
            _logger.warning(
                "Rollout capture thread saw no action snapshot within 10s; exiting."
            )
            return
        if self.stop_event.is_set():
            return

        frame_interval = 1.0 / self.fps
        recording_start = time.perf_counter()
        tick = 0

        while not self.stop_event.is_set():
            target_perf_ts = recording_start + tick * frame_interval

            wait_s = target_perf_ts - time.perf_counter()
            if wait_s > 0 and self.stop_event.wait(timeout=wait_s):
                return

            try:
                obs = self.robot.get_observation()
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "Capture tick %d: get_observation failed (%s).", tick, exc
                )
                tick += 1
                continue

            action = self.publisher.latest()
            if action is None:
                tick += 1
                continue

            obs_processed = self.robot_obs_proc(obs)
            if self.dataset is not None:
                obs_frame = build_dataset_frame(
                    self.dataset.features, obs_processed, prefix=OBS_STR
                )
                act_frame = build_dataset_frame(
                    self.dataset.features, action, prefix=ACTION
                )
                if self.stop_event.is_set():
                    return
                self.dataset.add_frame({**obs_frame, **act_frame, "task": self.task})

            if self.rerun_ip:
                if self.shadow:
                    # The policy's unexecuted wish — log under action.predicted.*
                    # so it reads distinctly from a normal executed-action run.
                    predicted = {f"predicted.{k}": v for k, v in action.items()}
                    log_rerun_data(observation=obs_processed, action=predicted)
                else:
                    log_rerun_data(observation=obs_processed, action=action)

                # 3D task-space view: live robot (FK from current joints), the
                # predicted/commanded EE pose per arm, and the predicted future
                # trajectory path per arm (from the latest action chunk). Only
                # pass a chunk when it changed, so the ~50-pose path FK runs once
                # per inference, not every tick (the path persists between chunks).
                # Best-effort — a viz hiccup must never disturb the capture loop.
                if self.robot_viz is not None:
                    try:
                        left_pos, right_pos = self.robot.positions
                        version, chunk = self.publisher.latest_chunk()
                        new_chunk = None
                        if chunk is not None and version != self._last_chunk_version:
                            new_chunk = chunk
                            self._last_chunk_version = version
                        self.robot_viz.log_frame(
                            left_pos=left_pos,
                            right_pos=right_pos,
                            action=action,
                            action_chunk=new_chunk,
                        )
                    except Exception as exc:  # noqa: BLE001
                        _logger.debug("3D robot viz tick %d failed: %s", tick, exc)

            tick += 1


class ShadowGravityCompThread(threading.Thread):
    """Hold both arms in gravity compensation for the duration of a shadow run.

    Shadow mode never actuates via ``send_action``, so without this loop the
    arms would receive no torque command at all: enabled but commanded to
    zero torque, they hang limp and sag under gravity. This thread ticks
    :meth:`AxolRobot.gravity_compensate` at ``rate_hz`` so the arms instead
    *float* — supporting their own weight while staying freely hand-guidable —
    giving the operator soft, controllable motion to walk the scene through
    the task while the policy infers on it. All seven arm joints are freed
    each cycle (``free_joints=None``); the gripper is held softly.

    It runs for the whole session — across episodes *and* the between-episode
    reset prompts — so the arms stay compliant the entire time the operator is
    hand-guiding or resetting the scene. ``gravity_compensate`` reads the
    cached joint positions, so robot telemetry must be active (it is whenever
    ``telemetry_hz > 0``, the default); a failed tick (e.g. a transient CAN
    error) is logged and the loop continues rather than killing the run.

    The loop can be paused via :meth:`pause` / :meth:`resume`: while paused it
    stops sending impedance commands so a powered return-to-rest (which drives
    the arms with stiff position targets) isn't fought by the float loop. Pause
    around the reset, then resume to float again from the new pose.
    """

    def __init__(
        self,
        *,
        robot: "AxolRobot",
        kd: float = 0.25,
        rate_hz: float = 250.0,
    ) -> None:
        super().__init__(name="axol-shadow-gravity-comp", daemon=True)
        self.robot = robot
        self.kd = kd
        self.rate_hz = rate_hz
        self.stop_event = threading.Event()
        # Active by default; cleared by ``pause()`` to suspend commanding.
        self._active = threading.Event()
        self._active.set()

    def pause(self) -> None:
        """Stop sending gravity-comp commands (e.g. during a return-to-rest)."""
        self._active.clear()

    def resume(self) -> None:
        """Resume gravity-comp commanding after a :meth:`pause`."""
        self._active.set()

    def run(self) -> None:
        dt = 1.0 / self.rate_hz
        while not self.stop_event.is_set():
            if not self._active.is_set():
                # Paused: idle without commanding so a concurrent return-to-rest
                # owns the arms. Poll the stop event so teardown stays prompt.
                if self.stop_event.wait(timeout=0.05):
                    return
                continue
            loop_start = time.perf_counter()
            try:
                self.robot.gravity_compensate(kd=self.kd)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("Shadow gravity-comp tick failed (%s).", exc)
            wait_s = dt - (time.perf_counter() - loop_start)
            if wait_s > 0 and self.stop_event.wait(timeout=wait_s):
                return


def stdin_watcher(
    stop_event: threading.Event,
    result: dict[str, str | None],
) -> None:
    """Watch stdin for ``s`` / ``r`` / ``q`` on its own line.

    Uses ``select.select`` so it never blocks past the stop event. Sets
    ``result["choice"]`` to the first valid keystroke received.
    """
    import select
    import sys

    while not stop_event.is_set():
        ready, _, _ = select.select([sys.stdin], [], [], 0.25)
        if not ready:
            continue
        line = sys.stdin.readline()
        if not line:
            return
        ch = line.strip().lower()
        if ch in ("s", "r", "q"):
            result["choice"] = ch
            return
