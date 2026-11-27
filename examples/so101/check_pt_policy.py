"""JAX policy vs PyTorch policy (run C, converted) on one held-out frame with the same noise."""
import glob, json, sys, time
import numpy as np, pandas as pd
import build_store as bs
import eval_foveavla as ev
from foveavla.policies import policy_config
from foveavla.training import config as _config

root, jax_ck, pt_ck = sys.argv[1:4]
cfg = _config.get_config("foveavla_so101_single")
info = json.load(open(f"{root}/meta/info.json"))["features"]["observation.images.top_depth"]["info"]
prompt = pd.read_parquet(f"{root}/meta/tasks.parquet").index[0]
eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/chunk-*/*.parquet"))])
e, t = 2, 40
r = eps[eps.episode_index == e].iloc[0]
vids = {}
for cam, fmt in (("top", "rgb24"), ("top_depth", "gray12le")):
    k = f"videos/observation.images.{cam}"
    p = f"{root}/{k}/chunk-{int(r[k + '/chunk_index']):03d}/file-{int(r[k + '/file_index']):03d}.mp4"
    vids[cam], _ = bs.decode(p, float(r[k + "/from_timestamp"]), t + 1, fmt)
df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"]) for f in sorted(glob.glob(f"{root}/data/chunk-*/*.parquet"))])
state = np.stack(df[df.episode_index == e].sort_values("frame_index")["observation.state"].to_numpy()).astype(np.float32)
depth, dvalid = bs.dequantize(vids["top_depth"][t], info)
obs = ev.obs_for(vids["top"][t], depth, dvalid, state[t], prompt, None)
noise = np.random.default_rng(0).standard_normal((cfg.model.action_horizon, cfg.model.action_dim)).astype(np.float32)
out = {}
for name, ck in (("pytorch", pt_ck), ("jax", jax_ck)):
    pol = policy_config.create_trained_policy(cfg, ck)
    t0 = time.time()
    o = pol.infer(dict(obs), noise=noise)
    out[name] = o
    print(name, f"{time.time() - t0:.0f}s", "box", np.round(o["future_interaction_box"], 1).tolist(), bool(o["future_interaction_box_valid"]), flush=True)
    del pol
a, b = np.asarray(out["jax"]["actions"]), np.asarray(out["pytorch"]["actions"])
print("actions shape", a.shape, "max|jax|", np.abs(a).max(), "max|diff|", np.abs(a - b).max(), "mean|diff|", np.abs(a - b).mean())
print("jax[0]", np.round(a[0], 2).tolist()); print("pt [0]", np.round(b[0], 2).tolist())
