"""Open-loop eval of a π0.5 SO101 baseline (pi05_so101_drawer_* of hungho77/openpi) on the held-out episodes.

Same queries and metrics as eval_foveavla.py: every REPLAN frames, action-chunk MAE against the recorded absolute
actions. Inputs are the store's letterboxed top (and wrist) frames, i.e. exactly what the baselines trained on.

Run inside an openpi checkout that has these configs (see README.md).
"""
import argparse, json, os
import numpy as np, pandas as pd

from openpi.policies import policy_config
from openpi.training import config as _config

HELDOUT = (2, 15, 17, 28, 34, 44)
REPLAN = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    policy = policy_config.create_trained_policy(_config.get_config(a.config), a.checkpoint)
    top = np.load(f"{a.store}/full.npy", mmap_mode="r")
    wrist = np.load(f"{a.store}/wrist.npy", mmap_mode="r")
    meta = pd.read_parquet(f"{a.store}/meta.parquet", columns=["index", "episode_index", "frame_index", "state", "action"])
    prompt = json.load(open(f"{a.store}/store.json"))["prompt"]
    rows = []
    for e in HELDOUT:
        m = meta[meta.episode_index == e].sort_values("frame_index")
        idx = m["index"].to_numpy()
        state = np.stack(m["state"].to_numpy()).astype(np.float32)
        action = np.stack(m["action"].to_numpy()).astype(np.float32)
        n = len(m)
        for t in range(0, n, REPLAN):
            out = policy.infer({"observation/image": np.asarray(top[idx[t]]), "observation/wrist_image": np.asarray(wrist[idx[t]]),
                                "observation/state": state[t], "prompt": prompt})
            chunk = np.asarray(out["actions"])
            err = np.abs(chunk - action[np.minimum(np.arange(t, t + len(chunk)), n - 1)])
            rows.append(dict(mode="na", episode=e, t=t, mae=float(err.mean()), mae_exec=float(err[:REPLAN].mean()),
                             **{f"mae_j{j}": float(err[:, j].mean()) for j in range(6)}))
        print(f"episode {e} done", flush=True)
    R = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    R.to_csv(f"{a.out}/queries.csv", index=False)
    summary = {k: float(R[k].mean()) for k in ["mae", "mae_exec"] + [f"mae_j{j}" for j in range(6)]} | {"queries": len(R)}
    json.dump({"config": a.config, "checkpoint": a.checkpoint, "heldout": list(HELDOUT), "replan": REPLAN,
               "summary": summary}, open(f"{a.out}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
