"""Compare the JAX and PyTorch FoveaVLA-SO101 serving policies on raw frames of a held-out episode.

Both policies get the same observations (through SO101FoveationPolicy, as served) and the same action noise, and the
foveation box is chained from the JAX prediction so both see identical crops. Prints the action and box differences
and each policy's error against the recorded actions.

    JAX_PLATFORMS=cpu python scripts/compare_so101_policies.py --device cuda
"""

import argparse
import glob
import json
import time

import numpy as np
import pandas as pd

from foveavla.policies import policy_config
from foveavla.policies.so101_foveation_policy import SO101FoveationPolicy
from foveavla.training import config

import check_so101_server as check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax-checkpoint", default="checkpoints/so101_drawer_red_block/foveavla_single")
    parser.add_argument("--pytorch-checkpoint", default="checkpoints/so101_drawer_red_block/foveavla_single_pytorch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--root", default="data/so101_drawer")
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--queries", type=int, default=3)
    args = parser.parse_args()

    cfg = config.get_config("foveavla_so101_single")
    policies = {
        "jax": SO101FoveationPolicy(policy_config.create_trained_policy(cfg, args.jax_checkpoint), {}),
        "torch": SO101FoveationPolicy(
            policy_config.create_trained_policy(cfg, args.pytorch_checkpoint, pytorch_device=args.device), {}
        ),
    }

    info = json.load(open(f"{args.root}/meta/info.json"))["features"]["observation.images.top_depth"]["info"]
    prompt = pd.read_parquet(f"{args.root}/meta/tasks.parquet").index[0]
    episodes = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{args.root}/meta/episodes/chunk-*/*.parquet"))])
    row = episodes[episodes.episode_index == args.episode].iloc[0]
    n = int(row["length"])
    videos = {}
    for cam, fmt in (("top", "rgb24"), ("top_depth", "gray12le")):
        key = f"videos/observation.images.{cam}"
        path = f"{args.root}/{key}/chunk-{int(row[key + '/chunk_index']):03d}/file-{int(row[key + '/file_index']):03d}.mp4"
        videos[cam] = check.decode(path, float(row[key + "/from_timestamp"]), args.queries * check.REPLAN, fmt)
    data = pd.concat(
        [pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
         for f in sorted(glob.glob(f"{args.root}/data/chunk-*/*.parquet"))]
    )
    data = data[data.episode_index == args.episode].sort_values("frame_index")
    state = np.stack(data["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(data["action"].to_numpy()).astype(np.float32)

    rng = np.random.default_rng(0)
    box = None
    for t in range(0, args.queries * check.REPLAN, check.REPLAN):
        raw = {
            "observation/top_rgb": videos["top"][t],
            "observation/top_depth_m": check.dequantize(videos["top_depth"][t], info),
            "observation/state": state[t],
            "observation/foveation_box": box,
            "prompt": prompt,
        }
        noise = rng.standard_normal((cfg.model.action_horizon, cfg.model.action_dim)).astype(np.float32)
        target = action[np.minimum(np.arange(t, t + cfg.model.action_horizon), n - 1)]
        out = {}
        for name, policy in policies.items():
            observation, _ = policy._prepare(raw)  # noqa: SLF001
            started = time.monotonic()
            out[name] = policy._policy.infer(observation, noise=noise)  # noqa: SLF001
            out[name]["seconds"] = time.monotonic() - started
        j, p = out["jax"], out["torch"]
        print(
            f"t={t:3d} |actions jax-torch| mean={np.abs(j['actions'] - p['actions']).mean():.3f} "
            f"max={np.abs(j['actions'] - p['actions']).max():.3f}  "
            f"mae jax={np.abs(j['actions'] - target).mean():.2f} torch={np.abs(p['actions'] - target).mean():.2f}  "
            f"box jax={np.round(j['future_interaction_box'], 3).tolist()} "
            f"torch={np.round(p['future_interaction_box'], 3).tolist()}  "
            f"time jax={j['seconds']:.1f}s torch={p['seconds']:.2f}s"
        )
        if bool(j["future_interaction_box_valid"]):
            box = np.asarray(j["future_interaction_box"], np.float32)


if __name__ == "__main__":
    main()
