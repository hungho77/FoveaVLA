"""SO101 inputs and outputs: five arm joints and the gripper; inputs are DroidInputs (state = joints + gripper),
optionally with the motion image of the top view and with the spatial state (hand position and gaze offset in the
robot frame)."""

import dataclasses
import pathlib
import time

import numpy as np
from foveavla_client import foveation
from foveavla_client import image_tools

from foveavla import transforms
from foveavla.policies import droid_policy
from foveavla.shared import motion
from foveavla.shared import spatial

MOTION_IMAGE_KEY = "motion_0_rgb"
# Arm and top camera of hungho77/so101-left-arm-task3, fitted by examples/so101/calibrate_camera.py.
LEFTARM_CALIBRATION = pathlib.Path(__file__).with_name("so101_leftarm_calibration.json")


class SO101MotionInputs(transforms.DataTransformFn):
    """DroidInputs plus the motion image (foveavla.shared.motion) under MOTION_IMAGE_KEY.

    The motion image comes from the first of:
      observation/motion_image             - ready made (training batches, offline eval);
      observation/exterior_image_1_left_prev - the full view of an earlier frame, sent by the client;
      the full view of the previous call   - kept here, so a robot client that replans every few frames needs no change.
    The kept view is dropped when the sample has a true "reset" entry or the previous call is older than `max_gap_s`;
    the first call after that sees a still scene, as the first frames of a training episode do.
    `observation/motion_image_mask` = False hides the stream from the model (as the training dropout does).
    """

    def __init__(self, max_gap_s: float = 3.0):
        self._inputs = droid_policy.DroidInputs()
        self._max_gap_s = max_gap_s
        self._previous: np.ndarray | None = None
        self._previous_time = 0.0

    def __call__(self, data: dict) -> dict:
        inputs = self._inputs(data)
        full = inputs["image"]["base_0_rgb"]
        if "observation/motion_image" in data:
            image = droid_policy._parse_image(data["observation/motion_image"])  # noqa: SLF001
        else:
            full = image_tools.resize_with_pad(full, 224, 224)
            now = time.monotonic()
            if data.get("reset") or now - self._previous_time > self._max_gap_s:
                self._previous = None
            previous = data.get("observation/exterior_image_1_left_prev", self._previous)
            if previous is None:
                image = motion.still_image(full.shape)
            else:
                previous = image_tools.resize_with_pad(droid_policy._parse_image(previous), 224, 224)  # noqa: SLF001
                image = motion.motion_image(full, previous)
            self._previous, self._previous_time = full, now
        inputs["image"][MOTION_IMAGE_KEY] = image
        inputs["image_mask"][MOTION_IMAGE_KEY] = np.bool_(data.get("observation/motion_image_mask", True))
        return inputs


class BoxMemory:
    """The last valid box the policy returned: the box a robot client crops its next foveal image with."""

    def __init__(self, max_gap_s: float = 3.0):
        self._max_gap_s = max_gap_s
        self._box: np.ndarray | None = None
        self._time = 0.0

    def store(self, box: np.ndarray, valid) -> None:
        if bool(np.all(valid)):
            self._box, self._time = np.asarray(box, np.float32).reshape(4), time.monotonic()

    def recall(self, *, reset: bool = False) -> np.ndarray | None:
        if reset or time.monotonic() - self._time > self._max_gap_s:
            self._box = None
        return self._box


class SO101SpatialInputs(transforms.DataTransformFn):
    """`inner` inputs with the spatial state (foveavla.shared.spatial) appended to the state: where the hand is in
    the robot frame (from the joints alone), how far it is from the line of sight through the centre of the foveal
    box (the gaze offset), and whether there is such a box.

    The gaze is the centre of `observation/foveal_region`, the part of the full view that the foveal image shows
    (xyxy, normalized; NaN or None when it shows the whole view). A robot client that renders its crop from the last
    valid box the policy returned (image_tools.extract_foveal_crop_direct) does not have to send it: with `memory`,
    shared with SO101Outputs, the region of that crop is used instead; the box is forgotten on a true "reset" entry
    or after a pause. `observation/gaze_mask` = False hides the gaze offset from the model (as the training dropout
    does).
    """

    def __init__(
        self,
        inner: transforms.DataTransformFn,
        calibration: spatial.Calibration | str | pathlib.Path = LEFTARM_CALIBRATION,
        memory: BoxMemory | None = None,
        context_scale: float = 1.5,
    ):
        self._context_scale = context_scale
        self._inner = inner
        self._calibration = (
            calibration if isinstance(calibration, spatial.Calibration) else spatial.Calibration.load(calibration)
        )
        self._memory = memory

    def __call__(self, data: dict) -> dict:
        inputs = self._inner(data)
        if "observation/foveal_region" in data:
            region = data["observation/foveal_region"]
        else:
            region = self._crop_region(
                None if self._memory is None else self._memory.recall(reset=bool(data.get("reset")))
            )
        if not data.get("observation/gaze_mask", True):
            region = None
        state = np.asarray(inputs["state"], np.float32)
        inputs["state"] = np.concatenate([state, spatial.spatial_state(self._calibration, state[:5], region)])
        return inputs

    def _crop_region(self, box: np.ndarray | None) -> np.ndarray | None:
        """The part of the full view that a client's crop around `box` shows."""
        if box is None:
            return None
        width, height = self._calibration.image_size
        box = np.clip(box, 0.0, 1.0)
        try:
            region = image_tools.resolve_foveal_crop_direct_bbox(
                (height, width), box, context_scale=self._context_scale
            )
        except ValueError:
            return None
        return np.asarray(region, np.float32) / np.array([width, height, width, height], np.float32)


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    # Set (and shared with SO101SpatialInputs) to remember the returned box for the next query's gaze point.
    memory: BoxMemory | None = None

    def __call__(self, data: dict) -> dict:
        data = foveation.canonicalize_fields(data)
        # The model pads actions to 32 dimensions; SO101 uses five joints and a gripper.
        outputs = {"actions": np.asarray(data["actions"][..., :6])}
        for key in ("future_interaction_box", "future_interaction_box_valid"):
            if key in data:
                outputs[key] = np.asarray(data[key])
        if self.memory is not None and "future_interaction_box" in outputs:
            self.memory.store(outputs["future_interaction_box"], outputs.get("future_interaction_box_valid", True))
        return outputs
