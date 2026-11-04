"""Query a running serve_so101.py with raw frames from a held-out LeRobot episode and compare to the published eval.

Sends the raw top RGB and metric depth (plus the wrist RGB with --wrist, for π0.5 baseline A) as an SO101 client
would, chains the predicted foveation box between queries ("self" mode of so101_port/eval_foveavla.py), and prints
action-chunk MAE per query next to the reference row from eval/<model>/queries.csv. Flow sampling is stochastic, so
expect the same ballpark, not identical numbers.
"""

import argparse
import glob
import json
import math
import time

import av
import numpy as np
import pandas as pd
from foveavla_client import websocket_client_policy

REPLAN = 8
DEPTH_QMAX = 4095


def decode(path, t0, n, fmt):
    out = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        container.seek(int(max(t0 - 1.0, 0) / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            i = int(round((frame.time - t0) * 30))
            if i < 0:
                continue
            if i >= n:
                break
            out.append(frame.to_ndarray(format=fmt))
    return out


def dequantize(code, info):
    lo, hi, shift = info["video.depth_min"], info["video.depth_max"], info["video.shift"]
    norm = code.astype(np.float32) / DEPTH_QMAX
    if info["video.use_log"]:
        lmin, lmax = math.log(lo + shift), math.log(hi + shift)
        depth = np.exp(norm * (lmax - lmin) + lmin) - shift
    else:
        depth = norm * (hi - lo) + lo
    return np.where(code > 0, depth, 0.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/so101_drawer")
    parser.add_argument("--reference", default="checkpoints/so101_drawer_red_block/eval/C/queries.csv")
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--wrist", action="store_true", help="also send observation/wrist_rgb (model A)")
    args = parser.parse_args()

    info = json.load(open(f"{args.root}/meta/info.json"))["features"]["observation.images.top_depth"]["info"]
    prompt = pd.read_parquet(f"{args.root}/meta/tasks.parquet").index[0]
    episodes = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{args.root}/meta/episodes/chunk-*/*.parquet"))])
    row = episodes[episodes.episode_index == args.episode].iloc[0]
    n = int(row["length"])
    frames = args.queries * REPLAN
    videos = {}
    for cam, fmt in (("top", "rgb24"), ("top_depth", "gray12le")) + ((("wrist", "rgb24"),) if args.wrist else ()):
        key = f"videos/observation.images.{cam}"
        path = f"{args.root}/{key}/chunk-{int(row[key + '/chunk_index']):03d}/file-{int(row[key + '/file_index']):03d}.mp4"
        videos[cam] = decode(path, float(row[key + "/from_timestamp"]), frames, fmt)
    data = pd.concat(
        [pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
         for f in sorted(glob.glob(f"{args.root}/data/chunk-*/*.parquet"))]
    )
    data = data[data.episode_index == args.episode].sort_values("frame_index")
    state = np.stack(data["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(data["action"].to_numpy()).astype(np.float32)
    reference = pd.read_csv(args.reference)
    # FoveaVLA rows chain their own boxes ("self"); the π0.5 baselines have one mode ("na").
    reference = reference[reference["mode"].isin(["self", "na"]) & (reference.episode == args.episode)].set_index("t")

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("server model:", client.get_server_metadata().get("model_name"))
    box = None
    for t in range(0, frames, REPLAN):
        started = time.monotonic()
        out = client.infer(
            {
                "observation/top_rgb": videos["top"][t],
                "observation/top_depth_m": dequantize(videos["top_depth"][t], info),
                "observation/state": state[t],
                "observation/foveation_box": box,
                "prompt": prompt,
                **({"observation/wrist_rgb": videos["wrist"][t]} if args.wrist else {}),
            }
        )
        elapsed = time.monotonic() - started
        chunk = np.asarray(out["actions"])
        err = np.abs(chunk - action[np.minimum(np.arange(t, t + len(chunk)), n - 1)])
        if bool(out.get("future_interaction_box_valid", False)):
            box = np.asarray(out["future_interaction_box"], np.float32)
        ref = reference.loc[t] if t in reference.index else None
        print(
            f"t={t:3d} actions={chunk.shape} mae={err.mean():.2f} mae_exec={err[:REPLAN].mean():.2f}"
            + (f" (ref {ref['mae']:.2f} / {ref['mae_exec']:.2f})" if ref is not None else "")
            + f" box={np.round(box, 3).tolist() if box is not None else None} {elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
