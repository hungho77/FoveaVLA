"""Server-side foveation for SO101 policies: one top RGB-D camera, rendered like so101_port/build_store.py."""

from __future__ import annotations

import time

import numpy as np
from foveavla_client import base_policy
from typing_extensions import override

from droid_foveation import repack_direct
from foveavla.shared.fast_foundation_depth import _sample_aligned_depth

OUTPUT_SIZE = 224
# Context scale of the exact crop used in training (CROP_ARGS in so101_port/build_store.py).
CONTEXT_SCALE = 1.5


class SO101FoveationPolicy(base_policy.BasePolicy):
    """Render the full view, foveal crop and aligned metric depth expected by ``DroidInputs``.

    The client sends the raw top frame and keeps the foveation state, as on the DROID robot:
      observation/top_rgb      uint8 [H, W, 3], the full-resolution top camera
      observation/top_depth_m  float [H, W], metric depth in meters on top_rgb's pixel grid, as LeRobot recorded it
                               for training (raw sensor image, no rs.align); 0 or non-finite is invalid
      observation/state        float [6], five arm joints plus the gripper, in dataset units
      observation/foveation_box  normalized [x1, y1, x2, y2] from the previous response's future_interaction_box
                                 (when future_interaction_box_valid), or None for the full frame on the first query
      prompt                   task instruction
    """

    def __init__(self, policy: base_policy.BasePolicy, metadata: dict) -> None:
        self._policy = policy
        self._metadata = metadata

    def _prepare(self, observation: dict) -> tuple[dict, float]:
        started = time.monotonic()
        observation = dict(observation)
        image = np.asarray(observation.pop("observation/top_rgb"), dtype=np.uint8)
        depth = np.asarray(observation.pop("observation/top_depth_m"), dtype=np.float32)
        state = np.asarray(observation.pop("observation/state"), dtype=np.float32)
        box = observation.pop("observation/foveation_box", None)
        if image.ndim != 3 or image.shape[2] != 3 or depth.shape != image.shape[:2]:
            raise ValueError(f"Expected top_rgb [H, W, 3] and top_depth_m [H, W], got {image.shape} and {depth.shape}")
        if state.shape != (6,):
            raise ValueError(f"Expected a 6-dim SO101 state, got {state.shape}")
        valid = np.isfinite(depth) & (depth > 0)
        depth = np.where(valid, depth, 0.0).astype(np.float32)

        full = repack_direct.resize_with_pad(image, OUTPUT_SIZE)
        full_depth, full_valid = _sample_aligned_depth(depth, valid, OUTPUT_SIZE, crop_bbox=None)
        if box is None:
            crop, crop_depth, crop_valid = full, full_depth, full_valid
        else:
            crop_bbox = repack_direct.exact_crop_bbox(image.shape, np.asarray(box, dtype=np.float32), CONTEXT_SCALE)
            crop = repack_direct.render_square_crop(image, crop_bbox, OUTPUT_SIZE)
            crop_depth, crop_valid = _sample_aligned_depth(depth, valid, OUTPUT_SIZE, crop_bbox=crop_bbox)
        observation.update(
            {
                "observation/exterior_image_1_left": full,
                "observation/foveal_image": crop,
                "observation/joint_position": state[:5],
                "observation/gripper_position": state[5:6],
                # The training store keeps float16 depth, then decodes to float32.
                "observation/metric_depth_full": full_depth.astype(np.float16).astype(np.float32),
                "observation/metric_depth_crop": crop_depth.astype(np.float16).astype(np.float32),
                "observation/pointmap_quality_full": full_valid.astype(np.float32),
                "observation/pointmap_quality_crop": crop_valid.astype(np.float32),
            }
        )
        return observation, (time.monotonic() - started) * 1000

    @staticmethod
    def _add_timing(result: dict, foveation_ms: float) -> dict:
        result.setdefault("policy_timing", {})["foveation_ms"] = foveation_ms
        return result

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        observation, foveation_ms = self._prepare(obs)
        return self._add_timing(self._policy.infer(observation), foveation_ms)

    def infer_actions_with_pending_bbox(self, obs: dict, *, noise=None):
        observation, foveation_ms = self._prepare(obs)
        result, pending = self._policy.infer_actions_with_pending_bbox(observation, noise=noise)
        return self._add_timing(result, foveation_ms), pending

    def infer_future_bbox(self, pending):
        return self._policy.infer_future_bbox(pending)

    def reset(self) -> None:
        self._policy.reset()

    @property
    def metadata(self) -> dict:
        return self._metadata
