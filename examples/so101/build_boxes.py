"""Raw Qwen3-VL labels -> per-frame FoveaVLA interaction boxes for the SO101 drawer/red-block task.

Per episode:
1. Clean each role track (gripper, cube, drawer): drop gripper boxes that sit on the arm mount, drop isolated jumps,
   interpolate gaps of up to MAX_GAP frames, and leave longer gaps invalid.
2. Phases from the gripper joint (state[5]) plus geometry:
   - grasp = start of the longest "holding" segment (gripper ~16, the cube width) after the gripper first opens;
     release = end of that segment;
   - before the grasp, one change point splits "pull" (open drawer: gripper near drawer) from "pick" (gripper near cube);
   - [grasp, release] is "place"; after release is "push" (close drawer).
3. Interaction box = union of the phase's roles, as in configs/role_policy_v1.yaml:
   pull/pick/push -> gripper + object (object = drawer for pull/push, cube for pick); place -> gripper + cube + drawer.

Output: one parquet per episode with columns frame_index, phase, box (xyxy normalized, NaN if invalid), valid.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

MAX_GAP = 18  # frames (0.6 s at 30 fps)
SPIKE = 0.15  # normalized center distance
# The gripper rests at 10-16 at the start of some episodes, opens above OPEN_MIN, and holds the cube at ~16.
OPEN_MIN, HOLD_LO, HOLD_HI = 24.0, 14.0, 18.5
ROLES = {"pull": ("gripper", "drawer"), "pick": ("gripper", "cube"), "place": ("gripper", "cube", "drawer"),
         "push": ("gripper", "drawer")}


def _area(b):
    return max(b[2] - b[0], 0) * max(b[3] - b[1], 0)


def _inter(a, b):
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def clean_track(frames, key, base_med=None):
    """frames: sorted list of (i, dict). Returns dict i -> box for kept labels."""
    pts = {}
    for i, d in frames:
        b = d.get(key)
        if b is None:
            continue
        if key == "gripper" and base_med is not None and _inter(b, base_med) > 0.7 * _area(b):
            continue
        if key == "gripper" and (b[2] - b[0]) * (b[3] - b[1]) > 0.12:  # whole-arm boxes
            continue
        pts[i] = np.asarray(b, np.float32)
    # Drop isolated spikes: a label far from both its neighbours while the neighbours agree with each other.
    idx = sorted(pts)
    ctr = {i: (pts[i][:2] + pts[i][2:]) / 2 for i in idx}
    keep = {}
    for k, i in enumerate(idx):
        if 0 < k < len(idx) - 1:
            p, q = ctr[idx[k - 1]], ctr[idx[k + 1]]
            far = min(np.abs(ctr[i] - p).max(), np.abs(ctr[i] - q).max())
            if far > SPIKE and np.abs(p - q).max() < far / 2:
                continue
        keep[i] = pts[i]
    return keep


def interpolate(keep, n):
    out = np.full((n, 4), np.nan, np.float32)
    idx = sorted(keep)
    for a, b in zip(idx, idx[1:] + [None]):
        out[a] = keep[a]
        if b is not None and b - a <= MAX_GAP:
            w = (np.arange(a + 1, b) - a)[:, None] / (b - a)
            out[a + 1:b] = (1 - w) * keep[a] + w * keep[b]
    if idx:  # hold the first/last label for up to MAX_GAP frames at the episode ends
        out[max(0, idx[0] - MAX_GAP):idx[0]] = keep[idx[0]]
        out[idx[-1] + 1:idx[-1] + 1 + MAX_GAP] = keep[idx[-1]]
    return out


def segments(mask):
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def grasp_window(grip):
    opened = np.where(grip > OPEN_MIN)[0]
    if not len(opened):
        return None
    t_open = int(opened[0])
    holds = [(a, b) for a, b in segments((grip >= HOLD_LO) & (grip <= HOLD_HI)) if a >= t_open and b - a >= 30]
    if not holds:
        return None
    a, b = max(holds, key=lambda s: s[1] - s[0])
    return int(a), int(b)


def phases(grip, g_box, cube, drawer, window):
    n = len(grip)
    t_grasp, t_release = window
    # Gripper-center-to-box distance (0 inside the box). Center-to-center distance over-weights the large drawer box
    # and put the whole pre-grasp segment into "pick" when the arm starts near the cube.
    gc = (g_box[:t_grasp, :2] + g_box[:t_grasp, 2:]) / 2
    to_box = lambda b: np.linalg.norm(np.maximum(np.maximum(b[:, :2] - gc, gc - b[:, 2:]), 0), axis=1)
    dd, dc = to_box(drawer[:t_grasp]), to_box(cube[:t_grasp])
    ok = np.isfinite(dd) & np.isfinite(dc)
    # cost(s) = sum_{t<s} dd + sum_{t>=s} dc over frames where both distances exist
    a = np.where(ok, dd, 0).cumsum()
    b = np.where(ok, dc, 0)[::-1].cumsum()[::-1]
    cost = np.concatenate([[0], a]) + np.concatenate([b, [0]])
    s = int(np.argmin(cost))
    ph = np.empty(n, object)
    ph[:s], ph[s:t_grasp], ph[t_grasp:t_release], ph[t_release:] = "pull", "pick", "place", "push"
    return ph, dict(split=s, grasp=int(t_grasp), release=int(t_release))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
                    for f in sorted(glob.glob(f"{a.root}/data/chunk-*/*.parquet"))])
    summary = []
    for p in sorted(glob.glob(f"{a.labels}/episode_*.json")):
        L = json.load(open(p))
        e, n = L["episode_index"], L["length"]
        frames = sorted((int(k), v) for k, v in L["frames"].items())
        bases = [v["base"] for _, v in frames if v.get("base")]
        base_med = np.median(np.asarray(bases), axis=0) if bases else None
        tr = {k: interpolate(clean_track(frames, k, base_med), n) for k in ("gripper", "cube", "drawer")}
        st = df[df.episode_index == e].sort_values("frame_index")
        grip = np.stack(st["observation.state"].values)[:, 5]
        window = grasp_window(grip)
        if window is None:
            summary.append(dict(episode=e, error="no holding segment"))
            continue
        # The cube lies still until it is grasped, and the VLM often misses it while the arm works the drawer:
        # use the median pre-grasp detection for every frame before the grasp.
        pre = tr["cube"][:max(window[0] - 30, 1)]
        pre = pre[np.isfinite(pre).all(1)]
        if len(pre):
            tr["cube"][:window[0]] = np.median(pre, axis=0)
        ph, info = phases(grip, tr["gripper"], tr["cube"], tr["drawer"], window)
        # While the cube is held its box follows the gripper; fill missing cube boxes from the gripper there.
        held = ph == "place"
        miss = held & np.isnan(tr["cube"][:, 0])
        tr["cube"][miss] = tr["gripper"][miss]
        box = np.full((n, 4), np.nan, np.float32)
        for i in range(n):
            bs = [tr[r][i] for r in ROLES[ph[i]]]
            if all(np.isfinite(b).all() for b in bs):
                bs = np.stack(bs)
                box[i] = [bs[:, 0].min(), bs[:, 1].min(), bs[:, 2].max(), bs[:, 3].max()]
        valid = np.isfinite(box).all(1)
        out = pd.DataFrame(dict(frame_index=np.arange(n), phase=ph.astype(str), x1=box[:, 0], y1=box[:, 1],
                                x2=box[:, 2], y2=box[:, 3], valid=valid))
        for r in ("gripper", "cube", "drawer"):
            for j, c in enumerate("xyxy"):
                out[f"{r}_{c}{1 + j // 2}"] = tr[r][:, j]
        out.to_parquet(f"{a.out}/episode_{e:03d}.parquet")
        summary.append(dict(episode=e, n=n, valid_frac=round(float(valid.mean()), 3),
                            **{k: round(float((ph == k).mean()), 3) for k in ROLES}, **info))
    s = pd.DataFrame(summary)
    s.to_csv(f"{a.out}/summary.csv", index=False)
    print(s.to_string())


if __name__ == "__main__":
    main()
