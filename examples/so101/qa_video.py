"""QA videos of the built boxes: per episode an MP4 (every 2nd frame, 15 fps) and a 16-frame sheet under <out>/<task>/.
Left: top view with the role boxes and the white interaction box
right: the foveal crop the model gets (grey when the
frame has no box and the full image is used).

  python qa_video.py <dataset root> <boxes dir> <out dir> <episodes, comma-separated>
"""
import sys
import glob
import subprocess
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from build_store import rd, CROP_ARGS, decode

root, box_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
episodes = [int(x) for x in sys.argv[4].split(",")]
eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/chunk-*/*.parquet"))])
summ = pd.read_csv(f"{box_dir}/summary.csv").set_index("episode")
W, H = 480, 270
COL = (("gripper", "yellow"), ("cube", "red"), ("drawer", "lime"), ("pot", "cyan"), ("target", "blue"))
for e in episodes:
    f = f"{box_dir}/episode_{e:03d}.parquet"
    if not os.path.exists(f):
        continue
    B = pd.read_parquet(f)
    n = len(B)
    task = summ.loc[e, "task"]
    row = eps[eps.episode_index == e].iloc[0]
    k = "videos/observation.images.top"
    frames, _ = decode(f"{root}/{k}/chunk-{int(row[k + '/chunk_index']):03d}/file-{int(row[k + '/file_index']):03d}.mp4",
                       float(row[k + "/from_timestamp"]), n, "rgb24")
    os.makedirs(f"{out}/{task}", exist_ok=True)
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W + H}x{H}",
                          "-r", "15", "-i", "-", "-c:v", "libx264", "-crf", "30", "-preset", "fast", "-pix_fmt", "yuv420p",
                          f"{out}/{task}/ep{e:03d}.mp4"], stdin=subprocess.PIPE)
    sheet_at = set(np.linspace(0, n - 1, 16).astype(int).tolist())
    tiles = []
    for i in range(n):
        if i % 2 and i not in sheet_at:
            continue
        b = B.iloc[i]
        img = frames[i]
        small = Image.fromarray(img).resize((W, H))
        d = ImageDraw.Draw(small)
        for r, c in COL:
            if f"{r}_x1" not in b:
                continue
            rb = b[[f"{r}_x1", f"{r}_y1", f"{r}_x2", f"{r}_y2"]].to_numpy(float)
            if np.isfinite(rb).all():
                d.rectangle([rb[0] * W, rb[1] * H, rb[2] * W, rb[3] * H], outline=c, width=1)
        if b["valid"]:
            ib = b[["x1", "y1", "x2", "y2"]].to_numpy(float)
            d.rectangle([ib[0] * W, ib[1] * H, ib[2] * W, ib[3] * H], outline="white", width=3)
            crop = rd.render_square_crop(img, rd.exact_crop_bbox(img.shape, ib, CROP_ARGS.context_scale), H)
        else:
            crop = np.full((H, H, 3), 60, np.uint8)
            ci = Image.fromarray(crop)
            ImageDraw.Draw(ci).text((70, 128), "NO BOX (full image)", fill="white")
            crop = np.asarray(ci)
        d.rectangle([0, 0, W, 13], fill="black")
        d.text((2, 1), f"{task} ep{e} f{i}/{n} {b['phase']}{'' if b['valid'] else ' (invalid)'}", fill="white")
        t = np.concatenate([np.asarray(small), crop], 1)
        if i % 2 == 0:
            p.stdin.write(t.tobytes())
        if i in sheet_at:
            tiles.append(t)
    p.stdin.close()
    p.wait()
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    Image.fromarray(np.concatenate([np.concatenate(tiles[j:j + 4], 1) for j in range(0, len(tiles), 4)], 0)).save(
        f"{out}/{task}/ep{e:03d}.jpg", quality=80)
    print(e, flush=True)
