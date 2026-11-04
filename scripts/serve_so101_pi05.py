"""Serve a π0.5 SO101 drawer baseline (model A: top + wrist, model B: top only) with openpi.

Needs openpi with code/openpi_so101drawer.patch applied (configs pi05_so101_drawer_top_wrist / _top) on PYTHONPATH.
A checkpoint directory with model.safetensors runs in PyTorch on --device; one with params/ runs in JAX.
Requests use the same keys as serve_so101.py (raw top/wrist frames, state, prompt).
"""

import argparse
import dataclasses
import importlib.util
import logging
from pathlib import Path
import sys
import types

# openpi's data loader imports LeRobot at module level; serving never reads a dataset, so stub it when absent.
if importlib.util.find_spec("lerobot") is None:
    for name in ("lerobot", "lerobot.common", "lerobot.common.datasets", "lerobot.common.datasets.lerobot_dataset"):
        sys.modules.setdefault(name, types.ModuleType(name))

from openpi.policies import policy_config  # noqa: E402
from openpi.training import config as openpi_config  # noqa: E402

from foveavla.policies.so101_pi05_policy import SO101Pi05Policy  # noqa: E402
from foveavla.serving import websocket_policy_server  # noqa: E402

CONFIGS = {"pi05_so101_drawer_top_wrist": True, "pi05_so101_drawer_top": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda", help="torch device for PyTorch checkpoints, e.g. cuda or cpu")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    cfg = openpi_config.get_config(args.config)
    asset_id = cfg.data.repo_id
    if not (args.checkpoint / f"assets/{asset_id}/norm_stats.json").is_file():
        parser.error(f"checkpoint must contain assets/{asset_id}/norm_stats.json")
    # torch.compile needs triton, which the Jetson CUDA torch environment does not ship.
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(cfg, args.checkpoint, pytorch_device=args.device)
    policy = SO101Pi05Policy(policy, use_wrist=CONFIGS[args.config])
    # Same capability keys as the FoveaVLA servers, so one client can pick its cameras from any port's metadata.
    metadata = policy.metadata | {
        "model_name": args.config,
        "requires_wrist_camera": CONFIGS[args.config],
        "requires_metric_depth": False,
    }
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
