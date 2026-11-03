"""QA sheet of the built interaction boxes: per frame the role boxes (gripper yellow, cube red, drawer green), the
interaction box (white, phase in the caption) and, next to it, the exact foveal crop the model would get."""
import sys, glob
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
from build_store import rd, CROP_ARGS, decode

root, box_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
episodes = [int(x) for x in sys.argv[4].split(",")]
per_ep = int(sys.argv[5]) if len(sys.argv) > 5 else 8
eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/chunk-*/*.parquet"))])
rows = []
for e in episodes:
    B = pd.read_parquet(f"{box_dir}/episode_{e:03d}.parquet")
    n = len(B)
    want = set(np.linspace(0, n - 1, per_ep).astype(int).tolist())
    tiles = []
    row = eps[eps.episode_index == e].iloc[0]
    k = "videos/observation.images.top"
    frames, _ = decode(f"{root}/{k}/chunk-{int(row[k + '/chunk_index']):03d}/file-{int(row[k + '/file_index']):03d}.mp4",
                       float(row[k + "/from_timestamp"]), n, "rgb24")
    for i in sorted(want):
        img = frames[i]
        im = Image.fromarray(img)
        b = B.iloc[i]
        small = im.resize((320, 180))
        d = ImageDraw.Draw(small)
        for r, c in (("gripper", "yellow"), ("cube", "red"), ("drawer", "lime")):
            rb = b[[f"{r}_x1", f"{r}_y1", f"{r}_x2", f"{r}_y2"]].to_numpy(float)
            if np.isfinite(rb).all():
                d.rectangle([rb[0] * 320, rb[1] * 180, rb[2] * 320, rb[3] * 180], outline=c, width=1)
        if b["valid"]:
            ib = b[["x1", "y1", "x2", "y2"]].to_numpy(float)
            d.rectangle([ib[0] * 320, ib[1] * 180, ib[2] * 320, ib[3] * 180], outline="white", width=3)
            crop = rd.render_square_crop(img, rd.exact_crop_bbox(img.shape, ib, CROP_ARGS.context_scale), 180)
        else:
            crop = np.zeros((180, 180, 3), np.uint8)
        d.rectangle([0, 0, 320, 13], fill="black")
        d.text((2, 1), f"ep{e} f{i} {b['phase']}{'' if b['valid'] else ' (invalid)'}", fill="white")
        tiles.append(np.concatenate([np.asarray(small), crop], 1))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows += [np.concatenate(tiles[j:j + 4], 1) for j in range(0, len(tiles), 4)]
Image.fromarray(np.concatenate(rows, 0)).save(out, quality=88)
