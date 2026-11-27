"""Batched SO101 samples from the store written by so101_port/build_store.py, in the DROID foveal batch layout.

Each batch matches what DroidFovealBatchDataset yields after the RLDS join, so the DROID repack and input transforms
apply unchanged: one top camera (letterboxed full view + foveal crop), metric depth for both, 5 arm joints plus the
gripper, absolute action chunks, and the future interaction box with its mask. Multi-task stores carry a task_index
per frame and the task strings in store.json "prompts"; single-task stores use store.json "prompt" for every frame.

With `motion_lag=(lo, hi)` each sample also carries the motion image of its full view (foveavla.shared.motion) against
the frame `lag` steps earlier in the same episode, `lag` drawn uniformly from lo..hi per sample (the first frame of the
episode when the sample is closer to the start than that).

With `gaze_dropout` set (a probability) each sample carries `observation/foveal_region`, the part of the full view its
foveal crop shows (xyxy, normalized), for the spatial state (foveavla.shared.spatial). The region is NaN - no gaze -
where the crop is the full view, as at the first query of an episode on the robot, and for a further `gaze_dropout`
share of the samples, so the policy also works without it.
"""

from collections.abc import Iterator, Sequence
import json
import pathlib

import numpy as np
import pandas as pd

from foveavla.shared import motion

_CROP_MODE_FULL = 1  # droid_foveation.repack_direct.CROP_MODE_FULL: the "crop" is the whole view


class SO101FovealBatchDataset:
    def __init__(
        self,
        store_dir: str,
        *,
        batch_size: int,
        action_horizon: int,
        episodes: Sequence[int] | None = None,
        shuffle: bool = True,
        repeat: bool = True,
        seed: int = 0,
        with_wrist: bool = False,
        motion_lag: tuple[int, int] | None = None,
        gaze_dropout: float | None = None,
    ):
        root = pathlib.Path(store_dir)
        self._arrays = {k: np.load(root / f"{k}.npy", mmap_mode="r") for k in ("full", "crop", "depth", "qual")}
        if with_wrist:
            self._arrays["wrist"] = np.load(root / "wrist.npy", mmap_mode="r")
        meta = pd.read_parquet(root / "meta.parquet")
        store = json.loads((root / "store.json").read_text())
        if "task_index" in meta and "prompts" in store:
            prompts = np.array([p.encode() for p in store["prompts"]], dtype=object)
            self._prompt = prompts[meta["task_index"].to_numpy()]
        else:
            self._prompt = np.full(len(meta), store["prompt"].encode(), dtype=object)
        self._state = np.stack(meta["state"].to_numpy()).astype(np.float32)
        self._action = np.stack(meta["action"].to_numpy()).astype(np.float32)
        self._future_box = np.stack(meta["future_box"].to_numpy()).astype(np.float32)
        self._future_mask = meta["future_mask"].to_numpy().astype(bool)
        episode = meta["episode_index"].to_numpy()
        # Last frame of each sample's episode, for chunks that run past the end (repeat the final action, as DROID).
        last = pd.Series(np.arange(len(meta))).groupby(episode).transform("max").to_numpy()
        self._first = pd.Series(np.arange(len(meta))).groupby(episode).transform("min").to_numpy()
        self._motion_lag = motion_lag
        self._gaze_dropout = gaze_dropout
        if gaze_dropout is not None:
            region = np.stack(meta["crop_bbox"].to_numpy()).astype(np.float32)  # source pixels
            full = meta["crop_mode"].to_numpy() == _CROP_MODE_FULL
            # Stores built before "image_size" was recorded: the full-view crops span the whole image.
            width, height = store.get("image_size") or region[full, 2:].max(0)
            self._region = region / np.array([width, height, width, height], np.float32)
            self._region[full] = np.nan
        self._chunk = np.minimum(np.arange(len(meta))[:, None] + np.arange(action_horizon)[None], last[:, None])
        keep = np.ones(len(meta), bool) if episodes is None else np.isin(episode, np.asarray(episodes))
        self._indices = np.flatnonzero(keep)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._repeat = repeat
        self._rng = np.random.default_rng(seed)
        self._with_wrist = with_wrist

    def __len__(self) -> int:
        return len(self._indices) // self._batch_size

    def __iter__(self) -> Iterator[dict]:
        while True:
            order = self._rng.permutation(self._indices) if self._shuffle else self._indices
            for start in range(0, len(order) - self._batch_size + 1, self._batch_size):
                yield self._batch(np.sort(order[start : start + self._batch_size]))
            if not self._repeat:
                return

    def _batch(self, idx: np.ndarray) -> dict:
        a = {k: v[idx] for k, v in self._arrays.items()}
        batch = {
            "observation/exterior_image_1_left": a["full"],
            "observation/foveal_image": a["crop"],
            "observation/joint_position": self._state[idx, :5],
            "observation/gripper_position": self._state[idx, 5:6],
            "observation/metric_depth_full": a["depth"][:, 0].astype(np.float32),
            "observation/metric_depth_crop": a["depth"][:, 1].astype(np.float32),
            "observation/pointmap_quality_full": a["qual"][:, 0].astype(np.float32),
            "observation/pointmap_quality_crop": a["qual"][:, 1].astype(np.float32),
            "actions": self._action[self._chunk[idx]],
            "prompt": self._prompt[idx],
            "future_interaction_box": self._future_box[idx],
            "future_interaction_box_mask": self._future_mask[idx],
        }
        if self._with_wrist:
            batch["observation/wrist_image_left"] = a["wrist"]
        if self._motion_lag is not None:
            lag = self._rng.integers(self._motion_lag[0], self._motion_lag[1] + 1, len(idx))
            earlier = np.maximum(idx - lag, self._first[idx])
            batch["observation/motion_image"] = motion.motion_image(a["full"], self._arrays["full"][earlier])
        if self._gaze_dropout is not None:
            region = self._region[idx].copy()
            region[self._rng.random(len(idx)) < self._gaze_dropout] = np.nan
            batch["observation/foveal_region"] = region
        return batch
