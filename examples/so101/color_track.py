"""Colour track of the red cube and the blue cube in the top view, for build_focus_boxes.py.

Both cubes have a colour nothing else in the scene has, so a colour threshold locates them more precisely than a
detector, also inside the translucent drawer and in the gripper. Every --stride-th frame: centroid of the largest dense
red / blue cluster (NaN when fewer than MIN_PX pixels).

  python color_track.py --root <dataset> --out <color dir> [--workers 8]
Output: <out>/episode_XXX.parquet with columns f, rx, ry, rn, bx, by, bn (pixels of the 640x360 view, cluster size).
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from detect_objects import decode

MIN_PX, HALF = 25, 25
KEY = "videos/observation.images.top"


def blob(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_PX:
        return np.nan, np.nan, len(xs)
    keep = (np.abs(xs - np.median(xs)) < HALF) & (np.abs(ys - np.median(ys)) < HALF)
    if keep.sum() < MIN_PX:
        return np.nan, np.nan, int(keep.sum())
    return float(xs[keep].mean()), float(ys[keep].mean()), int(keep.sum())


def track(job):
    root, out, row, stride = job
    e, n = int(row["episode_index"]), int(row["length"])
    path = f"{root}/{KEY}/chunk-{int(row[KEY + '/chunk_index']):03d}/file-{int(row[KEY + '/file_index']):03d}.mp4"
    frames = decode(path, float(row[KEY + "/from_timestamp"]), n)
    rows = []
    for i in range(0, n, stride):
        r, g, b = (frames[i][..., j].astype(int) for j in range(3))
        red = (r > 110) & (r > 2.0 * g) & (r > 1.6 * b)
        blue = (b > 90) & (b > 1.5 * g) & (b > 2.5 * r)  # the arm is cyan: its green is as high as its blue
        rows.append((i, *blob(red), *blob(blue)))
    pd.DataFrame(rows, columns="f rx ry rn bx by bn".split()).to_parquet(f"{out}/episode_{e:03d}.parquet")
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.root}/meta/episodes/chunk-*/*.parquet"))])
    jobs = [(a.root, a.out, r, a.stride) for _, r in eps.iterrows()
            if not os.path.exists(f"{a.out}/episode_{int(r['episode_index']):03d}.parquet")]
    with ProcessPoolExecutor(a.workers) as ex:
        for e in ex.map(track, jobs):
            print(e, flush=True)


if __name__ == "__main__":
    main()
