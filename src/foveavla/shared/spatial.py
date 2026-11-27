"""Robot-frame 3D for the SO101 arm and its external camera: where the hand is, and where it is relative to the gaze.

Two pieces, plain numpy:
* forward kinematics of the SO101 follower (the chain of TheRobotStudio's `so101_new_calib.urdf`): arm joints in
  degrees -> the tool point in the robot base frame, in metres;
* a pinhole camera calibrated against that frame (`Calibration`, fitted by examples/so101/calibrate_camera.py): base
  point <-> pixel of the external RGB view.

`spatial_state` packs what the policy gets: the hand position (from the joints alone: the arm knows where its hand is
without seeing it), and the gaze offset - the shortest move that puts the hand on the line of sight through the centre
of the foveal box. The offset shrinks to the calibration error (~2 cm) when the hand reaches what the fovea looks at.
It uses no depth on purpose: placing the gaze point in 3D with the depth of the box was tried and was off by more
than 15 cm in a tenth of the cases (the arm in front of the target, translucent drawers); where along the line of
sight the target lies is left to the depth streams the model already has.
"""

import dataclasses
import json
import pathlib

import numpy as np

SPATIAL_STATE_DIM = 7  # hand xyz, gaze offset xyz, has_gaze


def _transform(xyz, rpy) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    matrix[:3, 3] = xyz
    return matrix


# Joint origins of so101_new_calib.urdf, base -> shoulder_pan -> shoulder_lift -> elbow_flex -> wrist_flex ->
# wrist_roll; every joint turns about its local z axis.
_CHAIN = (
    _transform([0.0388353, 0.0, 0.0624], [3.14159, 0.0, -3.14159]),
    _transform([-0.0303992, -0.0182778, -0.0542], [-1.5708, -1.5708, 0.0]),
    _transform([-0.11257, -0.028, 0.0], [0.0, 0.0, 1.5708]),
    _transform([-0.1349, 0.0052, 0.0], [0.0, 0.0, -1.5708]),
    _transform([0.0, -0.0611, 0.0181], [1.5708, 0.0486795, 3.14159]),
)
# gripper_frame_link in the gripper link: the point between the jaw tips.
URDF_TOOL = (-0.0079, -0.000218121, -0.0981274)


def forward_kinematics(joints_rad: np.ndarray, tool: np.ndarray) -> np.ndarray:
    """[..., 5] URDF joint angles -> [..., 3] position of `tool` (a point in the gripper link) in the base frame."""
    joints_rad = np.asarray(joints_rad, np.float64)
    pose = np.broadcast_to(np.eye(4), (*joints_rad.shape[:-1], 4, 4))
    for j, origin in enumerate(_CHAIN):
        c, s = np.cos(joints_rad[..., j]), np.sin(joints_rad[..., j])
        turn = np.zeros((*joints_rad.shape[:-1], 4, 4))
        turn[..., 0, 0], turn[..., 0, 1], turn[..., 1, 0], turn[..., 1, 1] = c, -s, s, c
        turn[..., 2, 2] = turn[..., 3, 3] = 1.0
        pose = pose @ origin @ turn
    return pose[..., :3, :3] @ np.asarray(tool, np.float64) + pose[..., :3, 3]


@dataclasses.dataclass(frozen=True)
class Calibration:
    """Robot state -> URDF joints, and the camera in the base frame (x_cam = rotation @ x_base + translation)."""

    joint_sign: tuple[float, ...]
    joint_offset_deg: tuple[float, ...]
    tool: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], ...]
    translation: tuple[float, float, float]
    focal: float
    center: tuple[float, float]
    image_size: tuple[int, int]  # width, height of the view the pixels refer to

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Calibration":
        data = json.loads(pathlib.Path(path).read_text())
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: _freeze(v) for k, v in data.items() if k in fields})

    def hand(self, joints_deg: np.ndarray) -> np.ndarray:
        """[..., 5] arm joints as the robot reports them (degrees) -> [..., 3] tool point in the base frame (m)."""
        joints = np.asarray(joints_deg, np.float64)[..., :5] * np.asarray(self.joint_sign) + np.asarray(
            self.joint_offset_deg
        )
        return forward_kinematics(np.deg2rad(joints), np.asarray(self.tool))

    def project(self, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """[..., 3] base points -> ([..., 2] pixels, [...] depth along the optical axis)."""
        cam = np.asarray(point, np.float64) @ np.asarray(self.rotation).T + np.asarray(self.translation)
        return self.focal * cam[..., :2] / cam[..., 2:3] + np.asarray(self.center), cam[..., 2]

    def backproject(self, pixel: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """[..., 2] pixels and [...] metric depth along the optical axis -> [..., 3] base points."""
        pixel, depth = np.asarray(pixel, np.float64), np.asarray(depth, np.float64)
        cam = np.concatenate([(pixel - np.asarray(self.center)) / self.focal, np.ones_like(depth)[..., None]], -1)
        return (cam * depth[..., None] - np.asarray(self.translation)) @ np.asarray(self.rotation)

    def on_height(self, pixel: np.ndarray, height: float) -> np.ndarray:
        """The base point at z = `height` that the camera sees at `pixel`."""
        rotation = np.asarray(self.rotation)
        origin = -rotation.T @ np.asarray(self.translation)
        ray = rotation.T @ np.array([*(np.asarray(pixel, np.float64) - np.asarray(self.center)) / self.focal, 1.0])
        return origin + ray * (height - origin[2]) / ray[2]


def _freeze(value):
    return tuple(_freeze(v) for v in value) if isinstance(value, list) else value


def gaze_offset(calibration: Calibration, hand: np.ndarray, pixel: np.ndarray) -> np.ndarray:
    """From `hand` (base frame) to the nearest point of the camera ray through `pixel` of the calibrated view."""
    rotation = np.asarray(calibration.rotation)
    origin = -rotation.T @ np.asarray(calibration.translation)
    ray = rotation.T @ np.array(
        [*(np.asarray(pixel, np.float64) - np.asarray(calibration.center)) / calibration.focal, 1.0]
    )
    ray /= np.linalg.norm(ray)
    return origin + ray * ((hand - origin) @ ray) - hand


def spatial_state(calibration: Calibration, joints_deg: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    """[hand xyz, gaze offset xyz, has_gaze] in metres, float32.

    `box` is the foveal box, xyxy normalized to the calibrated view; without one (None, NaN, empty) there is no gaze:
    the offset is zero and the flag 0.
    """
    hand = calibration.hand(joints_deg)
    box = None if box is None else np.asarray(box, np.float64)
    has_gaze = (
        box is not None and box.shape == (4,) and bool(np.all(np.isfinite(box))) and box[2] > box[0] and box[3] > box[1]
    )
    offset = np.zeros(3)
    if has_gaze:
        width, height = calibration.image_size
        offset = gaze_offset(calibration, hand, [(box[0] + box[2]) / 2 * width, (box[1] + box[3]) / 2 * height])
    return np.concatenate([hand, offset, [float(has_gaze)]]).astype(np.float32)
