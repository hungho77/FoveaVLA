"""Subtask-focused interaction boxes: each subtask looks at ONE target with a fixed-size window.

build_boxes.py takes the union of the gripper and object boxes, so the box grows and shrinks with the distance between
the hand and the object and inherits every frame-to-frame error of the gripper labels. Here the window does not contain
the gripper on purpose: it sits on the object the current subtask is about (the hand enters it when it gets there) and
it has one size per focus. At a subtask boundary the window moves to the next target over --saccade frames, eased,
starting AT the boundary: the grasp or release is finished before the eye leaves it.

  task    subtask -> focus
  pot     pick -> pot | place, retreat -> orange hollow cube
  stack   pull -> drawer handle | pick -> red cube | place, retreat -> blue cube
  drawer  pull -> drawer handle | pick -> red cube | place -> where the cube ends up in the drawer | push -> drawer front

Subtask boundaries come from the robot state, not from a VLM (Cosmos3-Nano's reasoner was tried for this, both on the
whole episode and on 1 s windows, and did not localize the steps):
  grasp, release  the longest gripper "hold" segment (build_boxes.grasp_window)
  pull -> pick    stack: end of the last closed-jaw segment before the grasp (the drawer is pulled with closed jaws);
                  otherwise the departure towards the cube: the start of the final descent of d(t), the arm-joint
                  distance to the grasp pose (the last frame with d >= DEPART * the largest d of the 2 s before it,
                  ignoring pauses close to the cube). The home pose can be farther from the cube than the handle, so
                  the largest d of the whole episode is not a usable reference.
Objects are static while they are a focus, so each one is the median of its LocateAnything detections (detect_objects.py)
in the frames where it is not occluded. The drawer handle, the drawer front and the place inside the drawer are not
detected reliably (three identical translucent drawers, and the Qwen3-VL gripper boxes often sit on the arm instead of
the jaws): they are where the gripper tip is while the arm pulls / releases / pushes. The tip pixel comes from the arm
joints through a quadratic ridge fit (TipModel) on the moments where the tip position is known: at the grasp it is on
the red cube, at the release on the place target. Leave-one-out error of that fit: ~15 px median on 640x360.
The same fit checks the detections: a cube / target detected more than OFF px from where the hand grasps / releases is
replaced by the tip position (summary column from_tip).
The window size does not depend on the detected box: one size per focus, the same in every episode. build_store.py
crops 1.5x the window (context_scale), so the default 100 px window is a 150 px foveal crop of the 640x360 view.

Output: the same files as build_boxes.py (episode_XXX.parquet, summary.csv), so build_store.py and the QA scripts work
unchanged.
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd

from build_boxes import CLOSED_MAX, grasp_window, segments

W, H = 640, 360
DEPART = 0.9
FOCUS = {"pot": {"pick": "pot", "place": "target", "retreat": "target"},
         "stack": {"pull": "handle", "pick": "cube", "place": "target", "retreat": "target"},
         "drawer": {"pull": "handle", "pick": "cube", "place": "dest", "push": "front"}}
# Foci located through the gripper tip: phase, and the part of the phase whose median tip position is the focus.
# Following the tip frame by frame was tried and dropped: away from grasp / release poses the fit is off by ~30 px
# median (>75 px for a quarter of the frames, checked against the carried cube), so the window wandered off the drawer.
TIP = {"handle": ("pull", 0.6, 1.0), "front": ("push", 0.3, 0.8)}
COLUMN = {"handle": "drawer", "front": "drawer", "dest": "target"}  # role column the focus is stored under
OFF = 75  # px: a detected focus farther than this from the gripper tip at the grasp / release is replaced by the tip
# Window side as a multiple of --side: the pot is ~125 px wide, and the tip-located foci need room for the error of
# the tip fit and for the travel of the drawer.
SCALE = {"pot": 1.25, "handle": 1.2, "front": 1.2, "dest": 1.2}
ALL_PHASES = ("pull", "pick", "place", "push", "retreat")


def median_box(det, key, lo, hi, small=True):
    """Median detection of `key` in frames [lo, hi), after dropping wrong sizes and locations."""
    v = np.asarray([d[key] for i, d in det.items() if lo <= i < hi and key in d], float).reshape(-1, 4)
    area = (v[:, 2] - v[:, 0]) * (v[:, 3] - v[:, 1])
    if small:
        v = v[area < 0.03]
    elif len(v):  # the pot: boxes that also take in the arm are larger than the pot alone
        v = v[area <= 1.2 * np.percentile(area, 20)]
    if len(v) < 2:
        return None
    c = (v[:, :2] + v[:, 2:]) / 2
    keep = np.abs(c - np.median(c, 0)).max(1) < 0.05
    return np.median(v[keep], 0) if keep.sum() >= 2 else None


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
            before = np.array([d[max(t - 60, 0):t + 1].max() for t in range(len(d))])
            s = int(np.where((d >= DEPART * before) & (d >= 0.5 * d.max()))[0][-1])
    ph[:s], ph[s:t_grasp], ph[t_grasp:t_release], ph[t_release:] = "pull", "pick", "place", after
    return ph, s


class TipModel:
    """Arm joints (state[:5]) -> gripper-tip pixel in the top view, fitted on (joints, known tip position) pairs."""

    def __init__(self, x, y, lam=0.1, drop=60):
        x, y = np.asarray(x, float), np.asarray(y, float)
        self.mu, self.sd = x.mean(0), x.std(0)
        keep = np.ones(len(x), bool)
        for _ in range(2):  # second fit without the pairs the first one misses by > `drop` px (wrong detections)
            f = self._feats(x[keep])
            self.w = np.linalg.solve(f.T @ f + lam * np.eye(f.shape[1]), f.T @ y[keep])
            self.err = np.linalg.norm(self(x) - y, axis=1)
            keep = self.err < drop

    def _feats(self, x):
        z = (np.atleast_2d(x) - self.mu) / self.sd
        pairs = itertools.combinations_with_replacement(range(z.shape[1]), 2)
        return np.c_[np.ones(len(z)), z, np.stack([z[:, i] * z[:, j] for i, j in pairs], 1)]

    def __call__(self, x):
        return self._feats(x) @ self.w


def center(b):
    return [(b[0] + b[2]) / 2 * W, (b[1] + b[3]) / 2 * H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True, help="labels_raw: the task of each episode")
    ap.add_argument("--det", required=True, help="detect_objects.py output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--side", type=float, default=100, help="window side in pixels of the 640x360 top view")
    ap.add_argument("--saccade", type=int, default=30, help="frames over which the window moves to the next target")
    ap.add_argument("--episodes", default=None, help="comma-separated: write only these (the tip fit uses all)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
                    for f in sorted(glob.glob(f"{a.root}/data/chunk-*/*.parquet"))])
    only = {int(x) for x in a.episodes.split(",")} if a.episodes else None
    summary, eps, pairs = [], [], []
    for p in sorted(glob.glob(f"{a.labels}/episode_*.json")):
        L = json.load(open(p))
        e, n, task = L["episode_index"], L["length"], L.get("task", "drawer")
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
        obj = {"pot": median_box(det, "pot", 0, n, small=False) if task == "pot" else None,
               "cube": median_box(det, "cube", max(t_grasp - 150, 0), t_grasp - 10)}
        if obj["cube"] is None:
            obj["cube"] = median_box(det, "cube", 0, t_grasp)
        if task != "drawer":
            obj["target"] = median_box(det, "target", 0, t_grasp)
            if obj["target"] is None:
                obj["target"] = median_box(det, "target", 0, n)
            if obj["target"] is None and task == "stack":  # the red cube rests on the blue one after the release
                obj["target"] = median_box(det, "cube", t_release + 15, t_release + 120)
        # Known tip positions: on the cube (in the pot for the pot task) at the grasp, on the target at the release.
        at_grasp = obj["pot"] if task == "pot" else obj["cube"]
        if at_grasp is not None:
            pairs.append((state[t_grasp:t_grasp + 5, :5].mean(0), center(at_grasp)))
        at_release = obj.get("target")
        if task == "drawer":  # the cube lies where it was dropped until the drawer is pushed
            at_release = median_box(det, "cube", t_release + 15, t_release + 120)
        if at_release is not None and t_release < n - 1:
            pairs.append((state[max(t_release - 5, 0):t_release, :5].mean(0), center(at_release)))
        eps.append(dict(e=e, n=n, task=task, state=state, ph=ph, s=s, t_grasp=t_grasp, t_release=t_release, obj=obj))

    tip = TipModel(*zip(*pairs))
    print(f"tip fit: {len(pairs)} pairs, error median {np.median(tip.err):.1f} px, "
          f"{(tip.err >= 60).sum()} pairs dropped", flush=True)

    def point(xy):
        u, v = np.clip(xy[0] / W, 0, 1), np.clip(xy[1] / H, 0, 1)
        return np.array([u, v, u, v])

    for ep in eps:
        e, n, task, state, ph, obj = (ep[k] for k in ("e", "n", "task", "state", "ph", "obj"))
        if only is not None and e not in only:
            continue
        need = set(FOCUS[task].values())
        t_grasp, t_release = ep["t_grasp"], ep["t_release"]
        at = {"cube": tip(state[t_grasp:t_grasp + 5, :5].mean(0))[0],
              "target": tip(state[max(t_release - 5, 0):t_release, :5].mean(0))[0]}
        at["dest"], at["pot"] = at["target"], at["cube"]
        fixed = []
        for k in need & set(at):
            # A detection far from where the hand grasps / releases is another object (e.g. the blue cube returned
            # as "red cube" while the arm hides the red one): look where the hand goes instead.
            far = obj.get(k) is not None and np.abs(np.subtract(center(obj[k]), at[k])).max() > OFF
            if (obj.get(k) is None or far) and (k in ("cube", "pot", "dest") or t_release < n - 1):
                obj[k] = point(at[k])
                fixed.append(k)
        names = np.array([FOCUS[task][x] for x in ph])
        ctr = np.full((n, 2), np.nan)  # window center in pixels
        side = np.full(n, np.nan)
        for k in need:
            if obj.get(k) is not None:
                ctr[names == k], side[names == k] = center(obj[k]), a.side * SCALE.get(k, 1)
        for k, (phase, lo, hi) in TIP.items():
            idx = np.where(ph == phase)[0]
            part = idx[int(len(idx) * lo):max(int(len(idx) * hi), int(len(idx) * lo) + 1)]
            if k in need and len(part):
                obj[k] = point(np.median(tip(state[part, :5]), 0))
                ctr[idx], side[idx] = center(obj[k]), a.side * SCALE.get(k, 1)
        # Saccades: from the boundary on, ease from where the window was to where the new subtask wants it.
        for i in np.where(names[1:] != names[:-1])[0] + 1:
            if not (np.isfinite(ctr[i - 1]).all() and np.isfinite(ctr[i]).all()):
                continue
            hi = min(i + a.saccade, n)
            w = (np.arange(i, hi) - i + 1) / a.saccade
            w = (w * w * (3 - 2 * w))[:, None]
            start = np.r_[ctr[i - 1], side[i - 1]]
            moved = (1 - w) * start + w * np.c_[ctr[i:hi], side[i:hi]]
            ctr[i:hi], side[i:hi] = moved[:, :2], moved[:, 2]
        cx, cy = np.clip(ctr[:, 0], side / 2, W - side / 2), np.clip(ctr[:, 1], side / 2, H - side / 2)
        box = np.stack([(cx - side / 2) / W, (cy - side / 2) / H, (cx + side / 2) / W, (cy + side / 2) / H], 1)
        box = box.astype(np.float32)
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
                            split=ep["s"], grasp=int(ep["t_grasp"]), release=int(ep["t_release"]),
                            missing=",".join(sorted(k for k in need if obj.get(k) is None)),
                            from_tip=",".join(sorted(set(fixed) - {"dest"}))))
    sm = pd.DataFrame(summary).sort_values("episode")
    sm.to_csv(f"{a.out}/summary.csv", index=False)
    print(sm.to_string())


if __name__ == "__main__":
    main()
