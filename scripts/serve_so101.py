"""Serve a FoveaVLA-SO101 checkpoint over WebSocket; depth comes from the top RGB-D sensor, not stereo."""

import argparse
import logging
from pathlib import Path

from foveavla_client import foveation

from foveavla.policies import policy_config
from foveavla.policies.so101_foveation_policy import SO101FoveationPolicy
from foveavla.serving import websocket_policy_server
from foveavla.training import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="foveavla_so101_single", choices=config.CONFIG_NAMES)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/so101_drawer_red_block/foveavla_single"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    cfg = config.get_config(args.model)
    asset_id = cfg.data.repo_id
    if not (args.checkpoint / "params").is_dir() or not (args.checkpoint / f"assets/{asset_id}/norm_stats.json").is_file():
        parser.error(f"checkpoint must contain params/ and assets/{asset_id}/norm_stats.json")
    metadata = foveation.with_legacy_fields(cfg.policy_metadata)
    policy = SO101FoveationPolicy(policy_config.create_trained_policy(cfg, args.checkpoint), metadata)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
