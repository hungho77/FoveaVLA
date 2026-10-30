"""SO101 outputs: five arm joints and the gripper. Inputs reuse DroidInputs (state = joints + gripper)."""

import dataclasses

import numpy as np
from foveavla_client import foveation

from foveavla import transforms


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
