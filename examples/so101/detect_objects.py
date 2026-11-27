"""Object boxes from LocateAnything-3B for the subtask-focused interaction boxes (build_focus_boxes.py).

Every --stride-th frame of the top camera is upscaled 2x and sent to LocateAnything with the task's object names (the
"detect" prompt: the single-instance grounding prompt returns wrong boxes on these frames). A name is kept only when the
model returns exactly one box for it. LocateAnything cannot box the gripper on its own (it returns the whole arm, the
mount, or the leader arm), so the gripper is not asked for here.

Needs the NVlabs/Eagle checkout (Embodied/locateanything_worker.py) and the nvidia/LocateAnything-3B weights:
  python detect_objects.py --root <dataset> --tasks <labels_raw dir> --out <det dir> --eagle <Eagle/Embodied> \
      --model <LocateAnything-3B dir> [--episodes 0-50,108-158] [--shard 0/3]
Output: <out>/episode_XXX.json = {"episode", "task", "n", "stride", "det": {frame: {object: [x1, y1, x2, y2]}}}
"""
import argparse
import glob
import json
import os
import re
import sys

import av
import pandas as pd
from PIL import Image

NAMES = {"pot": {"cube": "red cube", "pot": "steel pot", "target": "orange cube"},
         "drawer": {"cube": "red cube"},
         "stack": {"cube": "red cube", "target": "blue cube"}}


def decode(path, t0, n):
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
            out[i] = fr.to_ndarray(format="rgb24")
    for i, x in enumerate(out):
        if x is None:
            out[i] = out[i - 1] if i else next(y for y in out if y is not None)
    return out


def parse_episodes(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tasks", required=True, help="labels_raw dir: episode_XXX.json with the 'task' key")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eagle", required=True, help="NVlabs/Eagle/Embodied checkout")
    ap.add_argument("--model", required=True, help="nvidia/LocateAnything-3B directory")
    ap.add_argument("--episodes", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--stride", type=int, default=5)
    a = ap.parse_args()
    sys.path.insert(0, a.eagle)
    from locateanything_worker import LocateAnythingWorker

    os.makedirs(a.out, exist_ok=True)
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.root}/meta/episodes/chunk-*/*.parquet"))])
    todo = parse_episodes(a.episodes) if a.episodes else sorted(eps.episode_index.tolist())
    k, m = (int(x) for x in a.shard.split("/"))
    todo = [e for e in todo if e % m == k and not os.path.exists(f"{a.out}/episode_{e:03d}.json")]
    la = LocateAnythingWorker(a.model)
    key = "videos/observation.images.top"
    for e in todo:
        task = json.load(open(f"{a.tasks}/episode_{e:03d}.json")).get("task", "drawer")
        names = NAMES[task]
        row = eps[eps.episode_index == e].iloc[0]
        fr = decode(f"{a.root}/{key}/chunk-{int(row[key + '/chunk_index']):03d}/file-{int(row[key + '/file_index']):03d}.mp4",
                    float(row[key + "/from_timestamp"]), int(row["length"]))
        h, w = fr[0].shape[:2]
        det = {}
        for i in range(0, len(fr), a.stride):
            im = Image.fromarray(fr[i]).resize((w * 2, h * 2), Image.BICUBIC)
            ans = la.detect(im, list(names.values()), verbose=False)["answer"]
            r = {}
            for lab, body in re.findall(r"<ref>(.*?)</ref>((?:<box>.*?</box>)+)", ans):
                bs = re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", body)
                for obj, name in names.items():
                    if name == lab and len(bs) == 1:
                        r[obj] = [int(x) / 1000 for x in bs[0]]
            det[i] = r
        json.dump(dict(episode=e, task=task, n=len(fr), stride=a.stride, det=det),
                  open(f"{a.out}/episode_{e:03d}.json", "w"))
        print(e, task, len(fr), {o: sum(o in d for d in det.values()) for o in names}, "of", len(det), flush=True)


if __name__ == "__main__":
    main()
