"""3D Axol robot + predicted end-effector pose in Rerun, for ``run-policy``.

``run-policy``'s Rerun stream otherwise shows only camera feeds and one scalar
plot per state/action dimension. This adds a **task-space 3D view**:

- the live robot, posed by forward kinematics from the current joint state,
- the policy's predicted (shadow) / commanded end-effector pose for each arm,
  drawn as a labelled point + orientation triad, with a line back to where the
  arm actually is, and
- the predicted **future trajectory path** of each hand — FK of the policy's
  full action chunk (the next ~50 predictions), drawn as a fading polyline.

FK uses :mod:`yourdfpy` on the bundled URDF (``almond_axol/kinematics/urdf``) —
the same lightweight loader the Cartesian-observation FK uses, but without the
jax/pyroki IK graph: per frame we only need link transforms, which is a few
numpy matmuls and stays off the GPU the camera relay / policy server need.

Meshes are logged once (static, in each link's local frame); every tick logs one
``Transform3D`` per link plus the EE markers. Everything is expressed in the
robot base frame, which (because the per-link transforms are absolute, not
chained) coincides with the Rerun world origin.

Entity scheme (drives the blueprint + per-entity toggles)::

    robot/<link>          Mesh3D (static) + Transform3D     # the live robot
    ee/<side>/current     Points3D                          # current EE (dim)
    ee/<side>/predicted   Transform3D (axes) + Points3D     # predicted EE (bright)
    ee/<side>/reach       LineStrips3D                      # current -> predicted
    ee/<side>/path        LineStrips3D + Points3D           # predicted next-N path

The predicted EE / path are derived from the *action(s)* the control loop
publishes: for a joint-space policy (the common case) we run FK on the predicted
joint angles; for a Cartesian policy (``observe_cartesian``) the action already
*is* an EE pose, so we read it straight out.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import rerun as rr

from ..constants import (
    ARM_JOINTS,
    URDF_PATH,
    Joint,
    urdf_arm_joint_names,
    urdf_body_name,
)

_logger = logging.getLogger(__name__)

_ROBOT_ROOT = "robot"

# Per-arm marker colours (RGB). Left = blue, right = orange — matching the
# end-effector traces drawn elsewhere in the Axol viz. "predicted" is the bright
# variant, "current" the dim one, so the eye reads the gap between them.
_COLORS: dict[str, dict[str, tuple[int, int, int]]] = {
    "left": {"current": (95, 130, 200), "predicted": (40, 160, 255)},
    "right": {"current": (200, 130, 70), "predicted": (255, 140, 25)},
}

# Cartesian action keys (observe_cartesian policies). Axis order matches
# AxolForwardKinematics.ee_poses / AxolRobot's Cartesian feature layout:
# position x/y/z then axis-angle rotation vector rx/ry/rz.
_EE_AXES = ("x", "y", "z", "rx", "ry", "rz")
_LEFT_EE_KEYS = [f"left_ee.{a}" for a in _EE_AXES]
_RIGHT_EE_KEYS = [f"right_ee.{a}" for a in _EE_AXES]

# Visual sizing, in metres / metres-of-axis. Tuned large so the predicted motion
# reads clearly against (and through) the translucent robot mesh — bump these if
# you want it even more prominent.
_CURRENT_RADIUS = 0.005
_PREDICTED_RADIUS = 0.01
_AXIS_LENGTH = 0.08
_PATH_LINE_RADIUS = 0.005
_PATH_POINT_RADIUS = 0.008

# Robot mesh appearance. The link meshes are logged with a translucent light-grey
# albedo so the bright predicted EE markers / future path show through the arms
# instead of being occluded by them. Alpha is 0-255; lower = more see-through.
_MESH_RGBA = (185, 195, 210, 90)


def _resolve_mesh(filename: str, urdf_dir: Path) -> Optional[Path]:
    """Resolve a URDF mesh ref (``package://assembly/meshes/X.stl``) to a real path."""
    name = filename
    if "://" in name:
        name = name.split("://", 1)[1]
        # drop the package root segment (e.g. 'assembly/'), keep 'meshes/X.stl'
        parts = name.split("/", 1)
        name = parts[1] if len(parts) == 2 else parts[0]
    p = urdf_dir / name
    return p if p.exists() else None


def _axisangle_to_mat3(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues: axis-angle rotation vector (radians) -> 3x3 rotation matrix.

    Used only for Cartesian-mode predicted poses, which arrive as ``[rx, ry, rz]``.
    A small numpy implementation so the viz needs no jax/scipy at runtime.
    """
    rotvec = np.asarray(rotvec, dtype=np.float64).ravel()
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    k = rotvec / theta
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return R.astype(np.float32)


class AxolRerunRobot:
    """Logs the 3D Axol robot and the policy's predicted EE pose to Rerun.

    Construct once (after ``init_rerun``): the constructor loads the URDF and
    logs every link mesh as a static entity. Then call :meth:`log_frame` each
    capture tick with the current joint positions and the latest action.

    Args:
        urdf_path: Override the bundled URDF (defaults to ``constants.URDF_PATH``).
        draw_meshes: Log link meshes (set False to draw only the EE markers,
            e.g. on a box where the STLs are missing).
        shadow: Whether the run is in shadow mode — only affects marker labels
            ("pred" vs "cmd"); the geometry is identical.
    """

    def __init__(
        self,
        *,
        urdf_path: str | Path | None = None,
        draw_meshes: bool = True,
        shadow: bool = True,
    ) -> None:
        import yourdfpy

        self._urdf_path = Path(urdf_path) if urdf_path else URDF_PATH
        if not self._urdf_path.exists():
            raise FileNotFoundError(f"Axol URDF not found at {self._urdf_path}")
        self._urdf = yourdfpy.URDF.load(
            str(self._urdf_path),
            load_collision_meshes=False,
            build_collision_scene_graph=False,
        )
        self._base = self._urdf.base_link
        self._joint_names = {
            "left": urdf_arm_joint_names(is_left=True),
            "right": urdf_arm_joint_names(is_left=False),
        }
        self._ee_link = {
            "left": urdf_body_name(Joint.GRIPPER, is_left=True),
            "right": urdf_body_name(Joint.GRIPPER, is_left=False),
        }
        self._label = "pred" if shadow else "cmd"
        if draw_meshes:
            self._log_meshes()

    # ------------------------------------------------------------------
    # Static setup
    # ------------------------------------------------------------------

    def _log_meshes(self) -> None:
        """Log each link's visual mesh once (static, baked into link-local frame)."""
        import trimesh

        urdf_dir = self._urdf_path.parent
        logged = 0
        for link_name, link in self._urdf.link_map.items():
            for visual in getattr(link, "visuals", []) or []:
                geom = getattr(getattr(visual, "geometry", None), "mesh", None)
                if geom is None or not getattr(geom, "filename", None):
                    continue
                mpath = _resolve_mesh(geom.filename, urdf_dir)
                if mpath is None:
                    continue
                try:
                    tm = trimesh.load(str(mpath), force="mesh")
                    verts = np.asarray(tm.vertices, dtype=np.float32)
                    # Bake the visual origin so the per-link Transform3D alone
                    # places the mesh correctly.
                    origin = np.asarray(
                        getattr(visual, "origin", np.eye(4)), dtype=np.float32
                    )
                    if origin.shape == (4, 4):
                        verts = (origin[:3, :3] @ verts.T).T + origin[:3, 3]
                    rr.log(
                        f"{_ROBOT_ROOT}/{link_name}",
                        rr.Mesh3D(
                            vertex_positions=verts,
                            triangle_indices=np.asarray(tm.faces, dtype=np.uint32),
                            vertex_normals=np.asarray(
                                tm.vertex_normals, dtype=np.float32
                            ),
                            # Translucent grey so the predicted EE markers / path
                            # read through the arms instead of being occluded.
                            albedo_factor=_MESH_RGBA,
                        ),
                        static=True,
                    )
                    logged += 1
                except Exception as exc:  # noqa: BLE001 - skip a bad mesh, keep the rest
                    _logger.debug("Skipping mesh for link %s (%s).", link_name, exc)
        _logger.info("3D robot viz: logged %d link meshes (static).", logged)

    # ------------------------------------------------------------------
    # Per-frame
    # ------------------------------------------------------------------

    def _arm_cfg(self, left7: Sequence[float], right7: Sequence[float]) -> dict:
        cfg = {n: float(left7[i]) for i, n in enumerate(self._joint_names["left"])}
        cfg.update(
            {n: float(right7[i]) for i, n in enumerate(self._joint_names["right"])}
        )
        return cfg

    def _pose_robot(
        self, left7: Sequence[float], right7: Sequence[float]
    ) -> dict[str, Optional[np.ndarray]]:
        """FK from the current arm joints; log every link's Transform3D.

        Returns the EE world position for each arm (read off the same FK pass).
        """
        self._urdf.update_cfg(self._arm_cfg(left7, right7))
        ee_pos: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        for link_name in self._urdf.link_map:
            try:
                T = np.asarray(
                    self._urdf.get_transform(link_name, self._base), dtype=np.float32
                )
            except Exception:  # noqa: BLE001 - a link without a path to base
                continue
            rr.log(
                f"{_ROBOT_ROOT}/{link_name}",
                rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]),
            )
            if link_name == self._ee_link["left"]:
                ee_pos["left"] = T[:3, 3]
            elif link_name == self._ee_link["right"]:
                ee_pos["right"] = T[:3, 3]
        return ee_pos

    def _predicted_ee(
        self, action: dict
    ) -> Optional[dict[str, tuple[np.ndarray, np.ndarray]]]:
        """Predicted EE ``(position, 3x3 rotation)`` per arm from the action dict.

        Cartesian-mode actions carry the EE pose directly; joint-mode actions are
        run through the same FK as the robot pose. Returns ``None`` if the action
        lacks the expected keys (so the caller just skips the predicted markers).
        Must be called *before* :meth:`_pose_robot`, since the joint-mode branch
        re-poses the URDF (the robot pose then restores it to the current state).
        """
        if _LEFT_EE_KEYS[0] in action:  # Cartesian policy
            out = {}
            for side, keys in (("left", _LEFT_EE_KEYS), ("right", _RIGHT_EE_KEYS)):
                try:
                    vec = np.array([float(action[k]) for k in keys], dtype=np.float32)
                except (KeyError, TypeError):
                    return None
                out[side] = (vec[:3], _axisangle_to_mat3(vec[3:6]))
            return out

        # Joint-space policy: FK on the predicted joint angles.
        try:
            pred = {
                side: [float(action[f"{side}_{j.value}.pos"]) for j in ARM_JOINTS]
                for side in ("left", "right")
            }
        except (KeyError, TypeError):
            return None
        self._urdf.update_cfg(self._arm_cfg(pred["left"], pred["right"]))
        out = {}
        for side in ("left", "right"):
            T = np.asarray(
                self._urdf.get_transform(self._ee_link[side], self._base),
                dtype=np.float32,
            )
            out[side] = (T[:3, 3], T[:3, :3])
        return out

    def _ee_path(self, actions: Sequence[dict]) -> dict[str, Optional[np.ndarray]]:
        """EE positions ``(N, 3)`` per arm for a chunk of future actions.

        Runs the same per-action FK as :meth:`_predicted_ee` over the chunk, so
        the result is the predicted 3D path each hand would follow. Like
        ``_predicted_ee`` it leaves the URDF re-posed, so call it *before*
        :meth:`_pose_robot`.
        """
        pts: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        for a in actions:
            ee = self._predicted_ee(a)
            if ee is None:
                continue
            pts["left"].append(ee["left"][0])
            pts["right"].append(ee["right"][0])
        return {
            side: (np.asarray(v, dtype=np.float32) if v else None)
            for side, v in pts.items()
        }

    def _log_ee_path(self, side: str, pts: Optional[np.ndarray]) -> None:
        """Draw one arm's predicted path: a line + per-waypoint points.

        Points fade from the arm's bright colour (now) to a darker tint (far
        horizon) so the direction of travel reads at a glance.
        """
        if pts is None or len(pts) < 2:
            return
        base = np.asarray(_COLORS[side]["predicted"], dtype=np.float32)
        t = np.linspace(0.0, 1.0, len(pts))[:, None]
        cols = (base[None, :] * (1.0 - 0.7 * t)).astype(np.uint8)  # bright -> dark
        rr.log(
            f"ee/{side}/path",
            rr.LineStrips3D(
                [pts], colors=[base.astype(np.uint8)], radii=_PATH_LINE_RADIUS
            ),
            rr.Points3D(pts, colors=cols, radii=_PATH_POINT_RADIUS),
        )

    def _log_ee_markers(
        self,
        side: str,
        current_pos: Optional[np.ndarray],
        predicted: Optional[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        col = _COLORS[side]
        if current_pos is not None:
            rr.log(
                f"ee/{side}/current",
                rr.Points3D(
                    [current_pos], radii=_CURRENT_RADIUS, colors=[col["current"]]
                ),
            )
        if predicted is not None:
            pos, rot = predicted
            # Transform3D places the entity frame at the predicted pose; the
            # Points3D at the local origin therefore lands exactly there, and
            # axis_length draws the orientation triad.
            rr.log(
                f"ee/{side}/predicted",
                rr.Transform3D(translation=pos, mat3x3=rot, axis_length=_AXIS_LENGTH),
                rr.Points3D(
                    [[0.0, 0.0, 0.0]],
                    radii=_PREDICTED_RADIUS,
                    colors=[col["predicted"]],
                    labels=[f"{side[0].upper()} {self._label}"],
                ),
            )
            if current_pos is not None:
                rr.log(
                    f"ee/{side}/reach",
                    rr.LineStrips3D([[current_pos, pos]], colors=[col["predicted"]]),
                )

    def log_frame(
        self,
        *,
        left_pos: Sequence[float],
        right_pos: Sequence[float],
        action: dict | None = None,
        action_chunk: Sequence[dict] | None = None,
    ) -> None:
        """Log one frame: pose the robot + draw both arms' predicted EE & path.

        Args:
            left_pos: Current left-arm joint positions (Joint order, >=7); a
                trailing gripper entry is ignored.
            right_pos: Current right-arm joint positions, same convention.
            action: Single published action for the immediate predicted EE
                point + orientation triad (joint targets, or EE pose in
                Cartesian mode). Defaults to ``action_chunk[0]`` if omitted.
            action_chunk: The policy's predicted future actions (the next ~50);
                each is FK'd to give the 3D path the hand would follow. ``None``
                leaves the previously-logged path in place (it persists in the
                viewer until the next chunk).
        """
        left7 = np.asarray(left_pos, dtype=np.float32).ravel()[:7]
        right7 = np.asarray(right_pos, dtype=np.float32).ravel()[:7]

        # FK that re-poses the URDF (the chunk path, then the single predicted
        # point) must run before _pose_robot, which restores the current state.
        paths = self._ee_path(action_chunk) if action_chunk else None
        if action is None and action_chunk:
            action = action_chunk[0]
        predicted = self._predicted_ee(action) if action else None
        current = self._pose_robot(left7, right7)

        for side in ("left", "right"):
            self._log_ee_markers(
                side, current[side], predicted[side] if predicted else None
            )
            if paths is not None:
                self._log_ee_path(side, paths[side])


def axol_run_policy_blueprint(camera_keys: Sequence[str]):
    """Blueprint for run-policy: 3D (robot + EE) | camera grid | scalar signals.

    Cameras are logged by LeRobot's ``log_rerun_data`` at single-segment paths
    like ``observation.left_arm_left`` (the dots are literal, not path
    separators), so each camera view targets that exact origin. The signals
    panel is a single time-series over all scalars (state + predicted action).
    """
    import rerun.blueprint as rrb

    view3d = rrb.Spatial3DView(
        origin="/", contents=["robot/**", "ee/**"], name="3D · Axol + predicted EE"
    )
    signals = rrb.TimeSeriesView(origin="/", name="signals")
    cam_views = [
        rrb.Spatial2DView(origin=f"observation.{k}", name=k) for k in camera_keys
    ]
    if not cam_views:
        # No cameras (e.g. a 3D-only offline replay): drop the camera column.
        return rrb.Blueprint(
            rrb.Horizontal(view3d, signals, column_shares=[3, 2]),
            collapse_panels=True,
        )
    return rrb.Blueprint(
        rrb.Horizontal(
            view3d,
            rrb.Grid(*cam_views),
            signals,
            column_shares=[3, 3, 2],
        ),
        collapse_panels=True,
    )
