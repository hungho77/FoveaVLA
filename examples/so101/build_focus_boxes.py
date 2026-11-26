"""Subtask-focused interaction boxes: each subtask looks at ONE target with a fixed-size window.

build_boxes.py takes the union of the gripper and object boxes, so the box grows and shrinks with the distance between
the hand and the object and inherits every frame-to-frame error of the gripper labels. Here the window does not contain
the gripper on purpose: it sits on the object the current subtask is about (the hand enters it when it gets there), it
has one size, and it moves to the next target at the subtask boundary over a short "saccade".

  task    subtask -> focus
  pot     pick -> pot | place, retreat -> orange hollow cube
  stack   pull -> drawer handle | pick -> red cube | place, retreat -> blue cube
  drawer  pull -> drawer handle | pick -> red cube | place -> where the cube ends up in the drawer | push -> drawer front

Subtask boundaries come from the robot state, not from a VLM (Cosmos3-Nano's reasoner was tried for this, both on the
whole episode and on 1 s windows, and did not localize the steps):
  grasp, release  the longest gripper "hold" segment (build_boxes.grasp_window)
  pull -> pick    stack: end of the last closed-jaw segment before the grasp (the drawer is pulled with closed jaws);
                  otherwise the departure towards the cube: the last frame before the grasp at which the arm joints are
                  still >= DEPART of their largest distance to the grasp pose.
Objects are static while they are a focus, so each one is the median of its LocateAnything detections (detect_objects.py)
in the frames where it is not occluded. The drawer handle and the drawer front are not detected reliably (three identical
drawers): they are the median gripper box (cleaned Qwen3-VL labels) while the arm pulls / pushes.

Output: the same files as build_boxes.py (episode_XXX.parquet, summary.csv), so build_store.py and the QA scripts work
unchanged.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from build_boxes import CLOSED_MAX, clean_track, grasp_window, interpolate, segments

W, H = 640, 360
DEPART = 0.9
FOCUS = {"pot": {"pick": "pot", "place": "target", "retreat": "target"},
         "stack": {"pull": "handle", "pick": "cube", "place": "target", "retreat": "target"},
         "drawer": {"pull": "handle", "pick": "cube", "place": "dest", "push": "front"}}
COLUMN = {"handle": "drawer", "front": "drawer", "dest": "target"}  # role column the focus is stored under
ALL_PHASES = ("pull", "pick", "place", "push", "retreat")


def median_box(det, key, lo, hi, small=True):
    """Median detection of `key` in frames [lo, hi), after dropping wrong sizes and locations."""
    v = np.asarray([d[key] for i, d in det.items() if lo <= i < hi and key in d], float).reshape(-1, 4)
    if small:
        v = v[(v[:, 2] - v[:, 0]) * (v[:, 3] - v[:, 1]) < 0.03]
    if len(v) < 2:
        return None
    c = (v[:, :2] + v[:, 2:]) / 2
    keep = np.abs(c - np.median(c, 0)).max(1) < 0.05
    return np.median(v[keep], 0) if keep.sum() >= 2 else None


def window(b, side):
    """Square window (in pixels) centered on box b, at least `side` wide and 1.5x the object, kept inside the image."""
    cx, cy = (b[0] + b[2]) / 2 * W, (b[1] + b[3]) / 2 * H
    s = min(max(side, 1.5 * max((b[2] - b[0]) * W, (b[3] - b[1]) * H)), H)
    cx, cy = np.clip(cx, s / 2, W - s / 2), np.clip(cy, s / 2, H - s / 2)
    return np.array([(cx - s / 2) / W, (cy - s / 2) / H, (cx + s / 2) / W, (cy + s / 2) / H], np.float32)


def split_phases(task, state, t_grasp, t_release):
    n = len(state)
    ph = np.empty(n, object)
    after = "push" if task == "drawer" else "retreat"
    s = 0
    if "pull" in FOCUS[task]:
        closed = [(a, b) for a, b in segments(state[:t_grasp, 5] < CLOSED_MAX) if b - a >= 15] if task == "stack" else []
        if closed:
            s = int(closed[-1][1])
        else:
            d = np.linalg.norm(state[:t_grasp + 1, :5] - state[t_grasp, :5], axis=1)
            s = int(np.where(d >= DEPART * d.max())[0][-1])
    ph[:s], ph[s:t_grasp], ph[t_grasp:t_release], ph[t_release:] = "pull", "pick", "place", after
    return ph, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True, help="labels_raw: task and Qwen3-VL gripper labels")
    ap.add_argument("--det", required=True, help="detect_objects.py output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--side", type=float, default=150, help="window side in pixels of the 640x360 top view")
    ap.add_argument("--saccade", type=int, default=12, help="frames over which the window moves to the next target")
    ap.add_argument("--episodes", default=None, help="comma-separated; default: every labelled episode")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
                    for f in sorted(glob.glob(f"{a.root}/data/chunk-*/*.parquet"))])
    only = {int(x) for x in a.episodes.split(",")} if a.episodes else None
    summary = []
    for p in sorted(glob.glob(f"{a.labels}/episode_*.json")):
        L = json.load(open(p))
        e, n, task = L["episode_index"], L["length"], L.get("task", "drawer")
        if only is not None and e not in only:
            continue
        dp = f"{a.det}/episode_{e:03d}.json"
        if not os.path.exists(dp):
            summary.append(dict(episode=e, task=task, error="no detections"))
            continue
        det = {int(k): v for k, v in json.load(open(dp))["det"].items()}
        state = np.stack(df[df.episode_index == e].sort_values("frame_index")["observation.state"].values)
        win = grasp_window(state[:, 5])
        if win is None:
            summary.append(dict(episode=e, task=task, error="no holding segment"))
            continue
        t_grasp, t_release = win
        ph, s = split_phases(task, state, t_grasp, t_release)
        frames = sorted((int(k), v) for k, v in L["frames"].items())
        bases = [v["base"] for _, v in frames if v.get("base")]
        grip = interpolate(clean_track(frames, "gripper", np.median(np.asarray(bases), axis=0) if bases else None), n)

        def gripper_median(phase, lo, hi):
            idx = np.where(ph == phase)[0]
            idx = idx[int(len(idx) * lo):max(int(len(idx) * hi), int(len(idx) * lo) + 1)]
            g = grip[idx]
            g = g[np.isfinite(g).all(1)]
            return np.median(g, 0) if len(g) >= 5 else None

        need = set(FOCUS[task].values())
        obj = {}
        if "pot" in need:
            obj["pot"] = median_box(det, "pot", 0, n, small=False)
        if "cube" in need:
            obj["cube"] = median_box(det, "cube", max(t_grasp - 150, 0), t_grasp - 10)
            if obj["cube"] is None:
                obj["cube"] = median_box(det, "cube", 0, t_grasp)
        if "target" in need:
            obj["target"] = median_box(det, "target", 0, t_grasp)
            if obj["target"] is None:
                obj["target"] = median_box(det, "target", 0, n)
            if obj["target"] is None and task == "stack":  # the red cube rests on the blue one after the release
                obj["target"] = median_box(det, "cube", t_release + 15, t_release + 120)
        if "dest" in need:
            obj["dest"] = median_box(det, "cube", t_release + 15, t_release + 120)
            if obj["dest"] is None:
                g = grip[max(t_release - 15, 0):t_release + 15]
                g = g[np.isfinite(g).all(1)]
                obj["dest"] = np.median(g, 0) if len(g) >= 5 else None
        if "handle" in need:
            obj["handle"] = gripper_median("pull", 0.6, 1.0)
        if "front" in need:
            obj["front"] = gripper_median("push", 0.3, 0.8)

        names = np.array([FOCUS[task][x] for x in ph])
        box = np.full((n, 4), np.nan, np.float32)
        for k in need:
            if obj.get(k) is not None:
                box[names == k] = window(obj[k], a.side)
        for i in np.where(names[1:] != names[:-1])[0] + 1:
            lo, hi = max(i - a.saccade // 2, 0), min(i + a.saccade // 2, n - 1)
            if np.isfinite(box[lo]).all() and np.isfinite(box[hi]).all():
                w = ((np.arange(lo, hi + 1) - lo) / (hi - lo))[:, None]
                box[lo:hi + 1] = (1 - w) * box[lo] + w * box[hi]
        valid = np.isfinite(box).all(1)
        out = pd.DataFrame(dict(frame_index=np.arange(n), phase=ph.astype(str), x1=box[:, 0], y1=box[:, 1],
                                x2=box[:, 2], y2=box[:, 3], valid=valid))
        for k in sorted(need):  # the focused object itself, for the QA sheets
            col = COLUMN.get(k, k)
            for j, c in enumerate(("x1", "y1", "x2", "y2")):
                v = np.where(names == k, obj[k][j] if obj.get(k) is not None else np.nan, np.nan)
                out[f"{col}_{c}"] = np.where(np.isfinite(v), v, out[f"{col}_{c}"]) if f"{col}_{c}" in out else v
        out.to_parquet(f"{a.out}/episode_{e:03d}.parquet")
        summary.append(dict(episode=e, task=task, n=n, valid_frac=round(float(valid.mean()), 3),
                            **{k: round(float((ph == k).mean()), 3) for k in ALL_PHASES if k in FOCUS[task]},
                            split=s, grasp=int(t_grasp), release=int(t_release),
                            missing=",".join(sorted(k for k in need if obj.get(k) is None))))
    sm = pd.DataFrame(summary)
    sm.to_csv(f"{a.out}/summary.csv", index=False)
    print(sm.to_string())


if __name__ == "__main__":
    main()
