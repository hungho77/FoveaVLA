"""Calibrate the SO101 arm against its external RGB-D camera, for the spatial state (foveavla.shared.spatial).

No calibration target is needed; the recorded episodes are enough:
* the table plane, fitted to the background of the metric depth, gives the camera's roll, pitch and height over the
  robot's base plane (the arm stands on the table);
* the carried cube gives image positions of the tool point: while the gripper holds the red cube, the cube's colour
  centroid (color_track.py) is where the tool point is in the image;
* the grasp and the release give 3D positions of it: at the grasp the tool point is at the cube, at the release it is
  over the place target, and both are seen free of the arm earlier in the episode: the labelled box, a few frames
  after the fovea moved there, placed in 3D with the depth under it (gaze_from_window) is a 3D anchor. A tenth of
  the anchors are far off (depth through the translucent drawers), hence the robust loss; they still pin down the
  direction along the camera rays, which pixels leave open (without anchors the fit was 15 px off in the image and
  up to 8 cm off in space).
The fit moves the base on the table (yaw, x, y, a small height offset), the joint zero offsets, the tool point and the
focal length until the tool point of the forward kinematics lands on those pixels and anchors.

  python calibrate_camera.py --store <store dir> --color <color_track dir> --boxes <boxes dir with summary.csv> \
      --out src/foveavla/policies/so101_leftarm_calibration.json [--heldout 66 68 ...]
Run from the repo root with PYTHONPATH=src.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from foveavla.shared import spatial

# Anchor depth: the surface under a box is this percentile of the base-frame heights seen in its central part.
SUPPORT_PERCENTILE = 25.0
SUPPORT_WINDOW = 0.6
MIN_DEPTH_PIXELS = 12
MIN_SIZE = 0.3  # as build_focus_boxes.py: smaller red clusters are the arm's wires, not the cube
SIZE = 224  # side of the store's letterboxed views


def carried_pairs(store, color, boxes):
    """(episode, arm joints, cube pixel) for the frames where the gripper carries the red cube."""
    meta = pd.read_parquet(f"{store}/meta.parquet", columns=["index", "episode_index", "state"])
    state = np.stack(meta["state"].to_numpy())
    first = meta.groupby("episode_index")["index"].min()
    rows = []
    for _, r in pd.read_csv(f"{boxes}/summary.csv").iterrows():
        if r.task == "pot" or not np.isfinite(r.grasp):  # the orange target of the pot task passes for red
            continue
        col = pd.read_parquet(f"{color}/episode_{int(r.episode):03d}.parquet")
        big = col.rn >= MIN_SIZE * np.nanpercentile(col.rn, 90)
        c = col[(col.f >= r.grasp + 3) & (col.f < r.release - 3) & big].dropna(subset=["rx", "ry"])
        rows += [(int(r.episode), *state[int(first[int(r.episode)]) + int(f), :5], x, y) for f, x, y in zip(c.f, c.rx, c.ry)]
    out = np.asarray(rows, np.float64)
    return out[:, 0].astype(int), out[:, 1:6], out[:, 6:8]


def support_window(image_size, box, depth, quality):
    """The box centre (source pixels) and the valid depth samples in the central part of the box (their source
    pixels and metric depths), or None. `depth` / `quality` are the store's letterboxed full view."""
    box = np.asarray(box, np.float64)
    if not np.all(np.isfinite(box)):
        return None
    width, height = image_size
    scale = SIZE / max(width, height)
    pad = np.array([(SIZE - width * scale) / 2, (SIZE - height * scale) / 2])
    center = np.array([(box[0] + box[2]) / 2 * width, (box[1] + box[3]) / 2 * height])
    half = np.array([(box[2] - box[0]) * width, (box[3] - box[1]) * height]) * SUPPORT_WINDOW / 2
    x1, y1 = np.maximum(np.floor((center - half) * scale + pad).astype(int), 0)
    x2, y2 = np.minimum(np.ceil((center + half) * scale + pad).astype(int), SIZE)
    window = np.asarray(depth[y1:y2, x1:x2], np.float64)
    valid = (np.asarray(quality[y1:y2, x1:x2]) > 0) & np.isfinite(window) & (window > 0)
    if valid.sum() < MIN_DEPTH_PIXELS:
        return None
    rows, columns = np.nonzero(valid)
    return center, (np.stack([columns + x1, rows + y1], -1) + 0.5 - pad) / scale, window[valid]


def gaze_from_window(calibration, center, pixels, depths):
    """The point on the ray through `center` at the height of the surface seen in the window."""
    heights = calibration.backproject(pixels, depths)[:, 2]
    return calibration.on_height(center, float(np.percentile(heights, SUPPORT_PERCENTILE)))


def anchors(store, boxes, image_size, frames=3, margin=15, saccade=30):
    """Grasp and release events with a view of their target before the arm gets there.

    Returns a list of dicts: episode, kind (grasp, release_<task>), arm joints at the event, and for up to `frames`
    earlier frames the support window of the labelled box.
    """
    meta = pd.read_parquet(f"{store}/meta.parquet", columns=["index", "episode_index", "state"])
    state = np.stack(meta["state"].to_numpy())
    first = meta.groupby("episode_index")["index"].min()
    depth = np.load(f"{store}/depth.npy", mmap_mode="r")
    qual = np.load(f"{store}/qual.npy", mmap_mode="r")
    out = []
    for _, r in pd.read_csv(f"{boxes}/summary.csv").iterrows():
        if r.task == "pot" or not np.isfinite(r.grasp):
            continue
        e, g0 = int(r.episode), int(first[int(r.episode)])
        labels = pd.read_parquet(f"{boxes}/episode_{e:03d}.parquet")
        phase, box = labels.phase.to_numpy(), labels[["x1", "y1", "x2", "y2"]].to_numpy(np.float64)
        for kind, t, name in (("grasp", int(r.grasp), "pick"), (f"release_{r.task}", int(r.release), "place")):
            start = np.flatnonzero(phase == name)
            if not len(start) or start[0] + saccade >= t - margin:
                continue
            ts = np.unique(np.linspace(start[0] + saccade, t - margin, frames).astype(int))
            windows = [support_window(image_size, box[x], np.asarray(depth[g0 + x, 0]).astype(np.float32),
                                              np.asarray(qual[g0 + x, 0])) for x in ts]
            windows = [w for w in windows if w is not None]
            if not windows:
                continue
            joints = state[g0 + t : g0 + t + 5, :5].mean(0) if kind == "grasp" else state[g0 + max(t - 5, 0) : g0 + t, :5].mean(0)
            out.append({"episode": e, "kind": kind, "joints": joints, "windows": windows})
    return out


def background_depth(store, width, height, frames=300):
    """Per-pixel far depth over frames spread across the store (the arm passes in front of the scene): source pixel
    coordinates [N, 2] and metric depth [N]."""
    depth = np.load(f"{store}/depth.npy", mmap_mode="r")
    qual = np.load(f"{store}/qual.npy", mmap_mode="r")
    scale = SIZE / max(width, height)
    pad = int(round((SIZE - height * scale) / 2))
    rows = slice(pad, SIZE - pad)
    idx = np.linspace(0, len(depth) - 1, frames).astype(int)
    d = np.asarray(depth[idx, 0, rows]).astype(np.float32)
    v = np.asarray(qual[idx, 0, rows]) > 0
    d[~v] = np.nan
    ok = v.mean(0) > 0.5
    far = np.full(ok.shape, np.nan, np.float32)
    far[ok] = np.nanpercentile(d[:, ok], 70, axis=0)
    y, x = np.nonzero(ok)
    return np.stack([(x + 0.5) / scale, (y + 0.5) / scale], -1), far[ok].astype(np.float64)


def table_plane(pixels, depth, focal, center, tolerance=0.006, trials=300):
    """Largest plane of the background, in the camera frame: unit normal pointing away from the camera, distance."""
    points = np.c_[(pixels - center) / focal * depth[:, None], depth]
    rng = np.random.default_rng(0)
    best = None
    for _ in range(trials):
        p = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(normal) < 1e-9:
            continue
        inliers = np.abs((points - p[0]) @ (normal / np.linalg.norm(normal))) < tolerance
        if best is None or inliers.sum() > best.sum():
            best = inliers
    inside = points[best]
    normal = np.linalg.svd(inside - inside.mean(0))[2][-1]
    normal = normal if normal[2] > 0 else -normal
    return normal, float(normal @ inside.mean(0)), float(best.mean())


def camera_pose(normal, distance, yaw, x, y, lift):
    """Camera pose (x_cam = rotation @ x_base + translation) of a base standing on the plane: its z axis is the plane
    normal towards the camera, its origin `lift` under the plane at plane coordinates (x, y), turned by `yaw`."""
    up = -normal
    e1 = np.cross(up, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    ex = np.cos(yaw) * e1 + np.sin(yaw) * e2
    rotation = np.stack([ex, np.cross(up, ex), up], 1)
    return rotation, normal * distance + x * e1 + y * e2 - lift * up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--color", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout", type=int, nargs="*", default=[], help="episodes left out of the fit, for the report")
    ap.add_argument("--image-size", type=int, nargs=2, default=[640, 360], metavar=("W", "H"))
    ap.add_argument("--focal", type=float, nargs=2, default=[300.0, 480.0], metavar=("LO", "HI"), help="search range")
    a = ap.parse_args()
    width, height = a.image_size
    center = np.array([width / 2, height / 2])
    episode, joints, pixel = carried_pairs(a.store, a.color, a.boxes)
    fit = ~np.isin(episode, a.heldout)
    events = anchors(a.store, a.boxes, (width, height))
    kinds = sorted({ev["kind"] for ev in events})
    print(f"{len(joints)} carried-cube pixels, {len(events)} grasp / release anchors {kinds}", flush=True)
    bg_pixels, bg_depth = background_depth(a.store, width, height)
    tool0 = np.asarray(spatial.URDF_TOOL)

    def calibration(p, focal, plane):
        rotation, translation = camera_pose(plane[0], plane[1], *p[8:12])
        return spatial.Calibration(
            joint_sign=(1.0,) * 5, joint_offset_deg=tuple(p[:5]), tool=tuple(p[5:8]),
            rotation=tuple(map(tuple, rotation)), translation=tuple(translation), focal=focal,
            center=tuple(center), image_size=(width, height),
        )  # fmt: skip

    def error(p, focal, plane, rows):
        c = calibration(p, focal, plane)
        return c.project(c.hand(joints[rows]))[0] - pixel[rows]

    def anchor_error(p, focal, plane, heldout):
        """Per event: tool point at the event minus (gaze point of its target + the tool's height over it), metres."""
        c = calibration(p, focal, plane)
        rows = []
        for ev in events:
            if (ev["episode"] in a.heldout) != heldout:
                continue
            gaze = [gaze_from_window(c, *w) for w in ev["windows"]]
            lift = np.array([0.0, 0.0, p[12 + kinds.index(ev["kind"])]])
            rows.append(c.hand(ev["joints"]) - np.median(gaze, 0) - lift)
        return np.asarray(rows).reshape(-1, 3)

    def residual(p, focal, plane, with_anchors=True):
        # 1 cm in space weighs as 8 px in the image, and each anchor as ANCHOR_WEIGHT pixel pairs. Weak priors keep
        # the weakly observed directions physical: joint zeros near the URDF's, the tool point near the gripper
        # frame, the base on the table.
        space = anchor_error(p, focal, plane, False).ravel() * 800 * 3 if with_anchors else []
        return np.r_[error(p, focal, plane, fit).ravel(), space, p[:5] * 0.8, (p[5:8] - tool0) * 400, p[11] * 400]

    lo = np.r_[[-45.0] * 5, tool0 - 0.03, -2 * np.pi, -1.5, -1.5, -0.06, [0.0] * len(kinds)]
    hi = np.r_[[45.0] * 5, tool0 + 0.03, 2 * np.pi, 1.5, 1.5, 0.06, [0.08] * len(kinds)]
    options = {"bounds": (lo, hi), "loss": "soft_l1", "f_scale": 8.0}
    best = None
    for focal in np.arange(a.focal[0], a.focal[1] + 1, 20.0):
        plane = table_plane(bg_pixels, bg_depth, focal, center)
        start = None  # pixels only, from several yaws: cheap, and close enough to start the full fit from
        for yaw in np.arange(0, 2 * np.pi, np.pi / 3):
            p0 = np.r_[np.zeros(5), tool0, yaw, 0.0, 0.0, 0.0, [0.02] * len(kinds)]
            sol = least_squares(residual, p0, args=(focal, plane, False), max_nfev=100, **options)
            if start is None or sol.cost < start.cost:
                start = sol
        sol = least_squares(residual, start.x, args=(focal, plane), max_nfev=150, **options)
        if best is None or sol.cost < best[0].cost:
            best = (sol, focal, plane)
        print(f"focal {focal:.0f}: cost {sol.cost:.0f}, best so far {best[0].cost:.0f} (focal {best[1]:.0f})", flush=True)
    sol, focal, plane = best
    c = calibration(sol.x, focal, plane)
    report = {"pairs": int(fit.sum()), "table_inlier_fraction": round(plane[2], 3),
              "tool_height_over_target_m": dict(zip(kinds, np.round(sol.x[12:], 4).tolist()))}
    for name, heldout in (("fit", False), ("heldout", True)):
        e = anchor_error(sol.x, focal, plane, heldout)
        if len(e):
            d = np.linalg.norm(e, axis=1) * 100
            report[f"{name}_anchor_cm"] = {"median": round(float(np.median(d)), 1), "p90": round(float(np.percentile(d, 90)), 1),
                                           "per_axis_std": np.round(e.std(0) * 100, 1).tolist(), "n": len(e)}
    for name, rows in (("fit", fit), ("heldout", ~fit)):
        if rows.any():
            e = np.linalg.norm(error(sol.x, focal, plane, rows), axis=1)
            report[f"{name}_px"] = {"median": round(float(np.median(e)), 1), "p90": round(float(np.percentile(e, 90)), 1),
                                    "n": int(rows.sum())}
    table = c.backproject(bg_pixels, bg_depth)[:, 2]
    report["table_height_m"] = round(float(np.median(table[np.abs(table - np.median(table)) < 0.02])), 4)
    data = {k: (np.asarray(v).tolist() if not isinstance(v, float) else v) for k, v in vars(c).items()}
    json.dump(data | {"report": report}, open(a.out, "w"), indent=1)
    print(json.dumps(data | {"report": report}, indent=1))


if __name__ == "__main__":
    main()
