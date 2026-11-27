"""Open-loop eval of a FoveaVLA-SO101 checkpoint on held-out episodes.

Queries every REPLAN frames. The crop at each query comes from one of two sources:
  self - the box the model predicted at the previous query (the first query sees the full frame), as on the robot;
  gt   - an exact crop around the labelled interaction box at t (full frame where the label is invalid).
Reports action-chunk MAE against the recorded absolute actions, and IoU of the predicted future box against the
label at t+REPLAN where that label is valid.

--episodes picks other episodes than the single-task held-out set, each prompted with its own task. With --steps the
summary also holds the open-loop MSE/MAE of openpi's scripts/eval_so101_open_loop.py (the gr00t protocol: chunks tiled
over the first --steps frames, one error per episode, mean over episodes), for comparison with the pi0.5 runs.

A model with a motion stream gets the motion image of the full view against the frame --motion-lag steps earlier (a
still image at the start of the episode); --motion off hides the stream from the model instead.
A model with the spatial state gets the region of the full view its crop shows (none at the first self query), from
which the policy computes the gaze offset; --gaze off hides the gaze from the model instead.

Run from the repo root with PYTHONPATH=src:packages/foveavla-client/src:examples/so101 (see README.md).
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

import build_store as bs
from build_store import rd, _sample_aligned_depth
from foveavla.policies import policy_config
from foveavla.shared import motion as _motion
from foveavla.training import config as _config

REPLAN = 8


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def obs_for(img, depth, dvalid, state, prompt, box, earlier=None, motion_on=True, gaze_on=True):
    if box is None:
        mode, cb = rd.CROP_MODE_FULL, (0, 0, img.shape[1], img.shape[0])
    else:
        mode, cb = rd.CROP_MODE_EXACT, rd.exact_crop_bbox(img.shape, box, bs.CROP_ARGS.context_scale)
    full = rd.resize_with_pad(img, bs.OUT)
    crop = full if mode == rd.CROP_MODE_FULL else rd.render_square_crop(img, cb, bs.OUT)
    fd, fv = _sample_aligned_depth(depth, dvalid, bs.OUT, crop_bbox=None)
    cd, cv = (fd, fv) if mode == rd.CROP_MODE_FULL else _sample_aligned_depth(depth, dvalid, bs.OUT, crop_bbox=cb)
    region = np.asarray(cb, np.float32) / np.tile(np.asarray(img.shape[1::-1], np.float32), 2)
    extra = {"observation/foveal_region": np.full(4, np.nan, np.float32) if mode == rd.CROP_MODE_FULL else region,
             "observation/gaze_mask": gaze_on}
    if earlier is not None:
        extra |= {"observation/motion_image": _motion.motion_image(full, rd.resize_with_pad(earlier, bs.OUT)),
                 "observation/motion_image_mask": motion_on}
    return extra | {
        "observation/exterior_image_1_left": full,
        "observation/foveal_image": crop,
        "observation/joint_position": state[:5],
        "observation/gripper_position": state[5:6],
        "observation/metric_depth_full": fd,
        "observation/metric_depth_crop": cd,
        "observation/pointmap_quality_full": fv.astype(np.float32),
        "observation/pointmap_quality_crop": cv.astype(np.float32),
        "prompt": prompt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="foveavla_so101_single")
    ap.add_argument("--modes", default="self,gt")
    ap.add_argument("--episodes", type=int, nargs="+", default=list(_config.SO101_HELDOUT_EPISODES))
    ap.add_argument("--steps", type=int, default=None, help="also report the tiled open-loop MSE/MAE over this many frames")
    ap.add_argument("--norm-std", type=float, default=None, help="absolute action std, for the normalized MAE")
    ap.add_argument("--motion-lag", type=int, default=REPLAN, help="frame lag of the motion image (motion models)")
    ap.add_argument("--motion", choices=("on", "off"), default="on", help="off: hide the motion stream from the model")
    ap.add_argument("--gaze", choices=("on", "off"), default="on", help="off: hide the gaze (spatial models)")
    a = ap.parse_args()
    cfg = _config.get_config(a.config)
    policy = policy_config.create_trained_policy(cfg, a.checkpoint)
    has_motion = getattr(cfg.model, "motion_image_key", None) is not None
    info = json.load(open(f"{a.root}/meta/info.json"))["features"]["observation.images.top_depth"]["info"]
    prompts = pd.read_parquet(f"{a.root}/meta/tasks.parquet").sort_values("task_index").index.tolist()
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.root}/meta/episodes/chunk-*/*.parquet"))])
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "task_index", "observation.state", "action"])
                    for f in sorted(glob.glob(f"{a.root}/data/chunk-*/*.parquet"))])
    rows, tiled = [], []
    for e in a.episodes:
        r = eps[eps.episode_index == e].iloc[0]
        n = int(r["length"])
        vids = {}
        for cam, fmt in (("top", "rgb24"), ("top_depth", "gray12le")):
            k = f"videos/observation.images.{cam}"
            p = f"{a.root}/{k}/chunk-{int(r[k + '/chunk_index']):03d}/file-{int(r[k + '/file_index']):03d}.mp4"
            vids[cam], _ = bs.decode(p, float(r[k + "/from_timestamp"]), n, fmt)
        d = df[df.episode_index == e].sort_values("frame_index")
        state = np.stack(d["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(d["action"].to_numpy()).astype(np.float32)
        prompt = prompts[int(d["task_index"].iloc[0])]
        B = pd.read_parquet(f"{a.boxes}/episode_{e:03d}.parquet")
        gt_box = B[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        gt_valid = B["valid"].to_numpy()
        for mode in a.modes.split(","):
            prev, errs = None, []
            for t in range(0, n, REPLAN):
                depth, dvalid = bs.dequantize(vids["top_depth"][t], info)
                box = prev if mode == "self" else (gt_box[t] if gt_valid[t] else None)
                earlier = vids["top"][max(t - a.motion_lag, 0)] if has_motion else None
                out = policy.infer(obs_for(vids["top"][t], depth, dvalid, state[t], prompt, box, earlier,
                                           a.motion == "on", a.gaze == "on"))
                chunk = np.asarray(out["actions"])
                target = action[np.minimum(np.arange(t, t + len(chunk)), n - 1)]
                err = np.abs(chunk - target)
                if a.steps and t % len(chunk) == 0 and t < min(a.steps, n - len(chunk)):
                    errs.append(chunk - target)
                pb, pv = np.asarray(out["future_interaction_box"], np.float32), bool(out["future_interaction_box_valid"])
                if pv:
                    prev = pb
                f = t + REPLAN
                rows.append(dict(mode=mode, episode=e, t=t, mae=float(err.mean()), mae_exec=float(err[:REPLAN].mean()),
                                 **{f"mae_j{j}": float(err[:, j].mean()) for j in range(6)}, box_valid=pv,
                                 **dict(zip(("px1", "py1", "px2", "py2"), map(float, pb))),
                                 iou=float(iou(pb, gt_box[f])) if (pv and f < n and gt_valid[f]) else np.nan))
            if errs:
                d_ = np.concatenate(errs)
                tiled.append(dict(mode=mode, episode=e, mse=float((d_ ** 2).mean()), mae=float(np.abs(d_).mean()),
                                  steps=len(d_)))
        print(f"episode {e} done", flush=True)
    R = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    R.to_csv(f"{a.out}/queries.csv", index=False)
    summary = {m: {k: float(g[k].mean()) for k in ["mae", "mae_exec"] + [f"mae_j{j}" for j in range(6)] + ["box_valid", "iou"]}
               | {"queries": int(len(g))} for m, g in R.groupby("mode")}
    result = {"checkpoint": a.checkpoint, "episodes": list(a.episodes), "replan": REPLAN, "summary": summary}
    if has_motion:
        result["motion"] = {"lag": a.motion_lag, "stream": a.motion}
    if getattr(cfg.data, "spatial_calibration", None) is not None:
        result["gaze"] = a.gaze
    if tiled:
        T = pd.DataFrame(tiled)
        T.to_csv(f"{a.out}/open_loop.csv", index=False)
        result["open_loop"] = {"steps": a.steps, "per_episode": tiled, "average": {
            m: {"mse": float(g.mse.mean()), "mae": float(g.mae.mean())}
            | ({"mae_normalized": float(g.mae.mean() / a.norm_std)} if a.norm_std else {}) for m, g in T.groupby("mode")}}
    json.dump(result, open(f"{a.out}/summary.json", "w"), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
