"""Contact sheet of raw labels: boxes (gripper yellow, cube red, drawer green) + phase text on top frames."""
import sys, json, glob
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from label_boxes import episode_frames
root, lab_dir, out, every = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/chunk-*/*.parquet"))])
rows = []
for p in sorted(glob.glob(f"{lab_dir}/episode_*.json")):
    L = json.load(open(p)); fr = {int(k): v for k, v in L["frames"].items()}
    keep = sorted(fr)[::every]
    row = eps[eps.episode_index == L["episode_index"]].iloc[0]
    tiles = []
    for i, im in episode_frames(root, row, 1):
        if i not in keep: continue
        im = im.resize((320, 180)); d = ImageDraw.Draw(im)
        for k, c in [("base", "magenta"), ("gripper", "yellow"), ("cube", "red"), ("drawer", "lime")]:
            b = fr[i][k]
            if b: d.rectangle([b[0]*320, b[1]*180, b[2]*320, b[3]*180], outline=c, width=2)
        d.rectangle([0, 0, 320, 14], fill="black"); d.text((2, 1), f"ep{L['episode_index']} f{i} {fr[i].get('phase') or ''}", fill="white")
        tiles.append(np.array(im))
    while len(tiles) % 5: tiles.append(np.zeros_like(tiles[0]))
    rows += [np.concatenate(tiles[j:j+5], 1) for j in range(0, len(tiles), 5)]
Image.fromarray(np.concatenate(rows, 0)).save(out, quality=85)
