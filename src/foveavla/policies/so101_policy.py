"""SO101 inputs and outputs: five arm joints and the gripper; inputs are DroidInputs (state = joints + gripper),
optionally with the motion image of the top view."""

import dataclasses
import time

import numpy as np
from foveavla_client import foveation
from foveavla_client import image_tools

from foveavla import transforms
from foveavla.policies import droid_policy
from foveavla.shared import motion

MOTION_IMAGE_KEY = "motion_0_rgb"


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


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        data = foveation.canonicalize_fields(data)
        # The model pads actions to 32 dimensions; SO101 uses five joints and a gripper.
        outputs = {"actions": np.asarray(data["actions"][..., :6])}
        for key in ("future_interaction_box", "future_interaction_box_valid"):
            if key in data:
                outputs[key] = np.asarray(data[key])
        return outputs
