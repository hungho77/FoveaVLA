"""Open-loop eval of a FoveaVLA-SO101 checkpoint on held-out episodes.

Queries every REPLAN frames. The crop at each query comes from one of two sources:
  self - the box the model predicted at the previous query (the first query sees the full frame), as on the robot;
  gt   - an exact crop around the labelled interaction box at t (full frame where the label is invalid).
Reports action-chunk MAE against the recorded absolute actions, and IoU of the predicted future box against the
label at t+REPLAN where that label is valid.

Run from the repo root with PYTHONPATH=src:packages/foveavla-client/src:examples/so101 (see README.md).
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

import build_store as bs
from build_store import rd, _sample_aligned_depth
from foveavla.policies import policy_config
from foveavla.training import config as _config

REPLAN = 8


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def obs_for(img, depth, dvalid, state, prompt, box):
    if box is None:
        mode, cb = rd.CROP_MODE_FULL, (0, 0, img.shape[1], img.shape[0])
    else:
        mode, cb = rd.CROP_MODE_EXACT, rd.exact_crop_bbox(img.shape, box, bs.CROP_ARGS.context_scale)
    full = rd.resize_with_pad(img, bs.OUT)
    crop = full if mode == rd.CROP_MODE_FULL else rd.render_square_crop(img, cb, bs.OUT)
    fd, fv = _sample_aligned_depth(depth, dvalid, bs.OUT, crop_bbox=None)
    cd, cv = (fd, fv) if mode == rd.CROP_MODE_FULL else _sample_aligned_depth(depth, dvalid, bs.OUT, crop_bbox=cb)
    return {
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
    a = ap.parse_args()
    cfg = _config.get_config(a.config)
    policy = policy_config.create_trained_policy(cfg, a.checkpoint)
    info = json.load(open(f"{a.root}/meta/info.json"))["features"]["observation.images.top_depth"]["info"]
    prompt = pd.read_parquet(f"{a.root}/meta/tasks.parquet").index[0]
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.root}/meta/episodes/chunk-*/*.parquet"))])
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
                    for f in sorted(glob.glob(f"{a.root}/data/chunk-*/*.parquet"))])
    rows = []
    for e in _config.SO101_HELDOUT_EPISODES:
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
        B = pd.read_parquet(f"{a.boxes}/episode_{e:03d}.parquet")
        gt_box = B[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        gt_valid = B["valid"].to_numpy()
        for mode in a.modes.split(","):
            prev = None
            for t in range(0, n, REPLAN):
                depth, dvalid = bs.dequantize(vids["top_depth"][t], info)
                box = prev if mode == "self" else (gt_box[t] if gt_valid[t] else None)
                out = policy.infer(obs_for(vids["top"][t], depth, dvalid, state[t], prompt, box))
                chunk = np.asarray(out["actions"])
                target = action[np.minimum(np.arange(t, t + len(chunk)), n - 1)]
                err = np.abs(chunk - target)
                pb, pv = np.asarray(out["future_interaction_box"], np.float32), bool(out["future_interaction_box_valid"])
                if pv:
                    prev = pb
                f = t + REPLAN
                rows.append(dict(mode=mode, episode=e, t=t, mae=float(err.mean()), mae_exec=float(err[:REPLAN].mean()),
                                 **{f"mae_j{j}": float(err[:, j].mean()) for j in range(6)}, box_valid=pv,
                                 iou=float(iou(pb, gt_box[f])) if (pv and f < n and gt_valid[f]) else np.nan))
        print(f"episode {e} done", flush=True)
    R = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    R.to_csv(f"{a.out}/queries.csv", index=False)
    summary = {m: {k: float(g[k].mean()) for k in ["mae", "mae_exec"] + [f"mae_j{j}" for j in range(6)] + ["box_valid", "iou"]}
               | {"queries": int(len(g))} for m, g in R.groupby("mode")}
    json.dump({"checkpoint": a.checkpoint, "heldout": list(_config.SO101_HELDOUT_EPISODES), "replan": REPLAN,
               "summary": summary}, open(f"{a.out}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
