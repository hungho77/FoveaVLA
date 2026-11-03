"""Build the FoveaVLA training store for the SO101 drawer/red-block dataset.

For every frame t of every episode it writes, at 224x224:
  full   - top RGB letterboxed (base_0_rgb)
  crop   - foveal crop of the top RGB around the interaction box at t (left_wrist_0_rgb slot), with the paper's crop
           sampling: 10% full image, 45% jittered, 45% exact, context scale 1.5, seeded per sample
  wrist  - wrist RGB letterboxed (only for the pi0.5 top+wrist baseline)
  depth  - metric depth (m) for the full view and the crop, nearest-sampled on the same grids, f16 [2,224,224]
  qual   - depth validity for both views, u8 [2,224,224]
plus meta.parquet: episode/frame index, state, crop mode/bbox, future box (t+OFFSET) and its mask.
Crop rendering and depth sampling reuse the official droid_foveation / foveavla functions unchanged.

Run from the repo root with PYTHONPATH=src:packages/foveavla-client/src:examples/so101 (see README.md).
"""
import argparse, glob, json, math, os, sys, types
from concurrent.futures import ProcessPoolExecutor
import av, numpy as np, pandas as pd

sys.modules.setdefault("lmdb", types.ModuleType("lmdb"))  # repack_direct imports lmdb at module level; unused here
from droid_foveation import repack_direct as rd  # noqa: E402
from foveavla.shared.fast_foundation_depth import _sample_aligned_depth  # noqa: E402

OUT = 224
OFFSET = 8  # future box offset = replan interval (code default)
CROP_ARGS = argparse.Namespace(seed=0, context_scale=1.5, full_image_probability=0.10, jitter_probability=0.45,
                               jitter_center_fraction=0.25, jitter_scale_min=1.0, jitter_scale_max=4.0 / 3.0)
# LeRobot depth dequantization (lerobot/datasets/depth_utils.py), parameters from meta/info.json
DEPTH_QMAX = 4095


def dequantize(code, info):
    lo, hi, shift = info["video.depth_min"], info["video.depth_max"], info["video.shift"]
    norm = code.astype(np.float32) / DEPTH_QMAX
    if info["video.use_log"]:
        lmin, lmax = math.log(lo + shift), math.log(hi + shift)
        depth = np.exp(norm * (lmax - lmin) + lmin) - shift
    else:
        depth = norm * (hi - lo) + lo
    return depth.astype(np.float32), code > 0


def decode(path, t0, n, fmt):
    out = [None] * n
    with av.open(path) as c:
        st = c.streams.video[0]
        c.seek(int(max(t0 - 1.0, 0) / st.time_base), stream=st, backward=True)
        for fr in c.decode(st):
            i = int(round((fr.time - t0) * 30))
            if i < 0:
                continue
            if i >= n:
                break
            out[i] = fr.to_ndarray(format=fmt)
    missing = [i for i, x in enumerate(out) if x is None]
    for i in missing:  # repeat the previous frame for the (rare) dropped frame
        out[i] = out[i - 1] if i else next(x for x in out if x is not None)
    return out, len(missing)


def process(job):
    root, row, offset0, boxes_path, out_dir, info = job
    e, n = int(row["episode_index"]), int(row["length"])
    vids = {}
    for cam, fmt in (("top", "rgb24"), ("wrist", "rgb24"), ("top_depth", "gray12le")):
        k = f"videos/observation.images.{cam}"
        p = f"{root}/{k}/chunk-{int(row[k + '/chunk_index']):03d}/file-{int(row[k + '/file_index']):03d}.mp4"
        vids[cam], miss = decode(p, float(row[k + "/from_timestamp"]), n, fmt)
        if miss > 2:
            raise RuntimeError(f"episode {e} {cam}: {miss} missing frames")
    B = pd.read_parquet(boxes_path) if os.path.exists(boxes_path) else None
    box = B[["x1", "y1", "x2", "y2"]].to_numpy(np.float32) if B is not None else np.full((n, 4), np.nan, np.float32)
    valid = B["valid"].to_numpy() if B is not None else np.zeros(n, bool)
    mm = {k: np.load(f"{out_dir}/{k}.npy", mmap_mode="r+") for k in ("full", "crop", "wrist", "depth", "qual")}
    meta = []
    for t in range(n):
        g = offset0 + t
        img = vids["top"][t]
        key = f"so101_drawer--{e}--{t}"
        if valid[t]:
            mode, cb = rd.choose_crop(img.shape, box[t], key, CROP_ARGS)
        else:
            mode, cb = rd.CROP_MODE_FULL, (0, 0, img.shape[1], img.shape[0])
        mm["full"][g] = rd.resize_with_pad(img, OUT)
        mm["crop"][g] = rd.resize_with_pad(img, OUT) if mode == rd.CROP_MODE_FULL else rd.render_square_crop(img, cb, OUT)
        mm["wrist"][g] = rd.resize_with_pad(vids["wrist"][t], OUT)
        depth, dvalid = dequantize(vids["top_depth"][t], info)
        fd, fv = _sample_aligned_depth(depth, dvalid, OUT, crop_bbox=None)
        cd, cv = (fd, fv) if mode == rd.CROP_MODE_FULL else _sample_aligned_depth(depth, dvalid, OUT, crop_bbox=cb)
        mm["depth"][g] = np.stack([fd, cd]).astype(np.float16)
        mm["qual"][g] = np.stack([fv, cv]).astype(np.uint8)
        f = t + OFFSET
        fmask = bool(f < n and valid[f])
        meta.append(dict(index=g, episode_index=e, frame_index=t, crop_mode=int(mode), crop_bbox=list(map(int, cb)),
                         box_valid=bool(valid[t]), future_box=(box[f] if fmask else np.zeros(4, np.float32)).tolist(),
                         future_mask=fmask))
    for m in mm.values():
        m.flush()
    return pd.DataFrame(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    info = json.load(open(f"{a.root}/meta/info.json"))["features"]["observation.images.top_depth"]["info"]
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.root}/meta/episodes/chunk-*/*.parquet"))])
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    N = int(eps["length"].sum())
    shapes = {"full": ((N, OUT, OUT, 3), np.uint8), "crop": ((N, OUT, OUT, 3), np.uint8),
              "wrist": ((N, OUT, OUT, 3), np.uint8), "depth": ((N, 2, OUT, OUT), np.float16),
              "qual": ((N, 2, OUT, OUT), np.uint8)}
    for k, (shape, dt) in shapes.items():
        np.lib.format.open_memmap(f"{a.out}/{k}.npy", mode="w+", dtype=dt, shape=shape).flush()
    offsets = np.concatenate([[0], eps["length"].cumsum().to_numpy()[:-1]])
    jobs = [(a.root, r, int(o), f"{a.boxes}/episode_{int(r['episode_index']):03d}.parquet", a.out, info)
            for (_, r), o in zip(eps.iterrows(), offsets)]
    metas = []
    with ProcessPoolExecutor(a.workers) as ex:
        for m in ex.map(process, jobs):
            metas.append(m)
            print(f"episode {int(m.episode_index.iloc[0])}: {len(m)} frames, box valid {m.box_valid.mean():.2f}, "
                  f"future mask {m.future_mask.mean():.2f}", flush=True)
    meta = pd.concat(metas).sort_values("index").reset_index(drop=True)
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "index", "observation.state", "action"])
                    for f in sorted(glob.glob(f"{a.root}/data/chunk-*/*.parquet"))]).sort_values("index")
    assert (df["index"].to_numpy() == meta["index"].to_numpy()).all(), "store index != dataset index"
    meta["state"] = df["observation.state"].to_numpy()
    meta["action"] = df["action"].to_numpy()
    meta.to_parquet(f"{a.out}/meta.parquet")
    task = pd.read_parquet(f"{a.root}/meta/tasks.parquet").index[0]
    json.dump({"frames": N, "episodes": len(eps), "offset": OFFSET, "prompt": task, "crop_args": vars(CROP_ARGS)},
              open(f"{a.out}/store.json", "w"), indent=1)
    print("done", N)


if __name__ == "__main__":
    main()
