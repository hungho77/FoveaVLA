"""Qwen3-VL box labels for the SO101 left-arm top camera (FoveaVLA port, raw stage).

Writes one JSON per episode: {"task": ..., "frames": {frame_index: {"base": xyxy|None, "gripper": ..., <objects>}}},
coordinates normalized to [0, 1] on the 640x360 top image. The objects asked for depend on the episode's task
(TASKS): drawer/red block, red cube pot -> orange hollow cube, or drawer + stack the red cube on the blue cube. Post-processing (smoothing, phases, role union,
future offset) happens in build_boxes.py.
"""
import argparse, glob, json, os, re
import av, numpy as np, pandas as pd, torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Phases are not asked for: the VLM's subtask guesses were unreliable in QA, so build_boxes.py derives them from the
# gripper state and box geometry. "base" is asked for only so gripper boxes that land on the arm mount can be rejected.
PREAMBLE = "The blue robot arm is mounted on a base at the image border; its far end carries a two-finger gripper. "
ARM = (
    '"base" is where the arm is mounted. "gripper" is only the two jaws at the free end '
    "of the arm, farthest along the arm from the base; use null if the jaws are out of view or hidden."
)
DRAWER = "the top drawer of the clear plastic drawer unit, including its front"
# Scene objects per task, matched on the episode's task string. The drawer entry reproduces the original single-task
# prompt exactly, so its labels stay comparable with the drawer/red-block run.
TASKS = {
    "pot": ("steel pot", {"cube": "the small solid red cube, not the orange hollow cube", "pot": "the steel pot",
                          "target": "the orange hollow cube"}),
    "drawer": ("top drawer", {"cube": "the red cube", "drawer": DRAWER}),
    "stack": ("blue cube", {"cube": "the red cube", "drawer": DRAWER, "target": "the blue cube"}),
}
ALL_KEYS = ("base", "gripper", "cube", "drawer", "pot", "target")


def task_of(task_string):
    hits = [name for name, (needle, _) in TASKS.items() if needle in task_string]
    if len(hits) != 1:
        raise ValueError(f"cannot map task {task_string!r} to one of {list(TASKS)}")
    return hits[0]


def build_prompt(objects):
    fields = ", ".join(f'"{k}": [x1,y1,x2,y2] or null' for k in objects)
    described = " ".join(f'"{k}" is {v}.' for k, v in objects.items())
    return (f'{PREAMBLE}Output JSON only: {{"base": [x1,y1,x2,y2], "gripper": [x1,y1,x2,y2] or null, {fields}}}. '
            f"{ARM} {described}")


PROMPT = build_prompt(TASKS["drawer"][1])
KEYS = ("base", "gripper", *TASKS["drawer"][1])


def episode_frames(root, ep_row, stride):
    k = "videos/observation.images.top"
    f = f"{root}/{k}/chunk-{int(ep_row[k + '/chunk_index']):03d}/file-{int(ep_row[k + '/file_index']):03d}.mp4"
    t0, n = float(ep_row[k + "/from_timestamp"]), int(ep_row["length"])
    want = set(range(0, n, stride))
    with av.open(f) as c:
        st = c.streams.video[0]
        c.seek(int(max(t0 - 1.0, 0) / st.time_base), stream=st, backward=True)
        for fr in c.decode(st):
            i = int(round((fr.time - t0) * 30))
            if i < 0:
                continue
            if i >= n:
                break
            if i in want:
                yield i, fr.to_image()


def parse(txt, keys=KEYS):
    m = re.search(r"\{.*\}", txt, re.S)
    out = dict.fromkeys(keys)
    if not m:
        return out
    try:
        js = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    for key in keys:
        b = js.get(key)
        if isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
            x1, y1, x2, y2 = (min(max(float(v) / 1000, 0.0), 1.0) for v in b)
            if x2 > x1 and y2 > y1:
                out[key] = [x1, y1, x2, y2]
    return out


def parse_episodes(spec):
    """"0-50,108,110-112" -> [0, ..., 50, 108, 110, 111, 112]."""
    ids = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        ids += range(int(lo), int(hi or lo) + 1)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", default="all")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.root}/meta/episodes/chunk-*/*.parquet"))])
    ids = eps.episode_index.tolist() if a.episodes == "all" else parse_episodes(a.episodes)
    proc = AutoProcessor.from_pretrained(a.model)
    proc.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    for e in ids:
        path = f"{a.out}/episode_{e:03d}.json"
        if os.path.exists(path):
            continue
        row = eps[eps.episode_index == e].iloc[0]
        task = task_of(row["tasks"][0])
        objects = TASKS[task][1]
        prompt, keys = build_prompt(objects), ("base", "gripper", *objects)
        res, buf = {}, []

        def flush():
            msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}] for _, im in buf]
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                gen = model.generate(**inp, max_new_tokens=200, do_sample=False)
            for (i, _), t in zip(buf, proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)):
                res[i] = parse(t, keys)
            buf.clear()

        for i, im in episode_frames(a.root, row, a.stride):
            buf.append((i, im))
            if len(buf) == a.batch:
                flush()
        if buf:
            flush()
        with open(path + ".tmp", "w") as f:
            json.dump({"episode_index": e, "length": int(row["length"]), "stride": a.stride, "task": task, "frames": res}, f)
        os.replace(path + ".tmp", path)
        print(f"episode {e}: {len(res)} frames labeled", flush=True)


if __name__ == "__main__":
    main()
