"""Server-side frame preparation for the π0.5 SO101 drawer baselines (openpi `pi05_so101_drawer_top[_wrist]`)."""

from __future__ import annotations

import time

import numpy as np
from foveavla_client import base_policy
from typing_extensions import override

from droid_foveation import repack_direct

OUTPUT_SIZE = 224
# FoveaVLA-only fields an SO101 client may also send; the π0.5 baselines ignore them.
_FOVEAVLA_ONLY_KEYS = ("observation/top_depth_m", "observation/foveation_box")


class SO101Pi05Policy(base_policy.BasePolicy):
    """Letterbox raw SO101 frames to 224x224 as the training store did, then call an openpi SO101 policy.

    Same request keys as SO101FoveationPolicy, so one client can query either server:
      observation/top_rgb    uint8 [H, W, 3], the full-resolution top camera
      observation/wrist_rgb  uint8 [H, W, 3], the full-resolution wrist camera (required when use_wrist)
      observation/state      float [6], five arm joints plus the gripper, in dataset units
      prompt                 task instruction
    """

    def __init__(self, policy: base_policy.BasePolicy, *, use_wrist: bool) -> None:
        self._policy = policy
        self._use_wrist = use_wrist

    def _prepare(self, observation: dict) -> tuple[dict, float]:
        started = time.monotonic()
        observation = dict(observation)
        for key in _FOVEAVLA_ONLY_KEYS:
            observation.pop(key, None)
        top = repack_direct.resize_with_pad(np.asarray(observation.pop("observation/top_rgb"), np.uint8), OUTPUT_SIZE)
        wrist = observation.pop("observation/wrist_rgb", None)
        if wrist is None:
            if self._use_wrist:
                raise ValueError("This baseline uses the wrist camera: send observation/wrist_rgb")
            wrist = np.zeros_like(top)  # SO101Inputs zeroes and masks the slot anyway when use_wrist is False
        else:
            wrist = repack_direct.resize_with_pad(np.asarray(wrist, np.uint8), OUTPUT_SIZE)
        observation.update({"observation/image": top, "observation/wrist_image": wrist})
        return observation, (time.monotonic() - started) * 1000

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        observation, prepare_ms = self._prepare(obs)
        result = self._policy.infer(observation)
        result.setdefault("policy_timing", {})["letterbox_ms"] = prepare_ms
        return result

    def reset(self) -> None:
        self._policy.reset()

    @property
    def metadata(self) -> dict:
        return self._policy.metadata
