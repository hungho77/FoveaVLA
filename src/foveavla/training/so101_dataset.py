"""Batched SO101 samples from the store written by so101_port/build_store.py, in the DROID foveal batch layout.

Each batch matches what DroidFovealBatchDataset yields after the RLDS join, so the DROID repack and input transforms
apply unchanged: one top camera (letterboxed full view + foveal crop), metric depth for both, 5 arm joints plus the
gripper, absolute action chunks, and the future interaction box with its mask.
"""

from collections.abc import Iterator, Sequence
import json
import pathlib

import numpy as np
import pandas as pd


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
    ):
        root = pathlib.Path(store_dir)
        self._arrays = {k: np.load(root / f"{k}.npy", mmap_mode="r") for k in ("full", "crop", "depth", "qual")}
        if with_wrist:
            self._arrays["wrist"] = np.load(root / "wrist.npy", mmap_mode="r")
        meta = pd.read_parquet(root / "meta.parquet")
        self._prompt = json.loads((root / "store.json").read_text())["prompt"].encode()
        self._state = np.stack(meta["state"].to_numpy()).astype(np.float32)
        self._action = np.stack(meta["action"].to_numpy()).astype(np.float32)
        self._future_box = np.stack(meta["future_box"].to_numpy()).astype(np.float32)
        self._future_mask = meta["future_mask"].to_numpy().astype(bool)
        episode = meta["episode_index"].to_numpy()
        # Last frame of each sample's episode, for chunks that run past the end (repeat the final action, as DROID).
        last = pd.Series(np.arange(len(meta))).groupby(episode).transform("max").to_numpy()
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
            "prompt": np.array([self._prompt] * len(idx), dtype=object),
            "future_interaction_box": self._future_box[idx],
            "future_interaction_box_mask": self._future_mask[idx],
        }
        if self._with_wrist:
            batch["observation/wrist_image_left"] = a["wrist"]
        return batch
