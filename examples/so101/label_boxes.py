"""Qwen3-VL box + subtask labels for the SO101 drawer/red-block top camera (FoveaVLA port, raw stage).

Writes one JSON per episode: {frame_index: {"base": xyxy|None, "gripper": ..., "cube": ..., "drawer": ...}},
coordinates normalized to [0, 1] on the 640x360 top image. Post-processing (smoothing, phases, role union,
future offset) happens in build_boxes.py.
"""
import argparse, glob, json, os, re
import av, numpy as np, pandas as pd, torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Phases are not asked for: the VLM's subtask guesses were unreliable in QA, so build_boxes.py derives them from the
# gripper state and box geometry. "base" is asked for only so gripper boxes that land on the arm mount can be rejected.
PROMPT = (
    "The blue robot arm is mounted on a base at the image border; its far end carries a two-finger gripper. "
    'Output JSON only: {"base": [x1,y1,x2,y2], "gripper": [x1,y1,x2,y2] or null, "cube": [x1,y1,x2,y2] or null, '
    '"drawer": [x1,y1,x2,y2] or null}. "base" is where the arm is mounted. "gripper" is only the two jaws at the free end '
    "of the arm, farthest along the arm from the base; use null if the jaws are out of view or hidden. \"cube\" is the red "
    'cube. "drawer" is the top drawer of the clear plastic drawer unit, including its front.'
)
KEYS = ("base", "gripper", "cube", "drawer")


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


def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    out = dict.fromkeys(KEYS)
    if not m:
        return out
    try:
        js = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    for key in KEYS:
        b = js.get(key)
        if isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
            x1, y1, x2, y2 = (min(max(float(v) / 1000, 0.0), 1.0) for v in b)
            if x2 > x1 and y2 > y1:
                out[key] = [x1, y1, x2, y2]
    return out


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
    ids = eps.episode_index.tolist() if a.episodes == "all" else [int(x) for x in a.episodes.split(",")]
    proc = AutoProcessor.from_pretrained(a.model)
    proc.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    for e in ids:
        path = f"{a.out}/episode_{e:03d}.json"
        if os.path.exists(path):
            continue
        row = eps[eps.episode_index == e].iloc[0]
        res, buf = {}, []

        def flush():
            msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": PROMPT}]}] for _, im in buf]
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                gen = model.generate(**inp, max_new_tokens=200, do_sample=False)
            for (i, _), t in zip(buf, proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)):
                res[i] = parse(t)
            buf.clear()

        for i, im in episode_frames(a.root, row, a.stride):
            buf.append((i, im))
            if len(buf) == a.batch:
                flush()
        if buf:
            flush()
        with open(path + ".tmp", "w") as f:
            json.dump({"episode_index": e, "length": int(row["length"]), "stride": a.stride, "frames": res}, f)
        os.replace(path + ".tmp", path)
        print(f"episode {e}: {len(res)} frames labeled", flush=True)


if __name__ == "__main__":
    main()
