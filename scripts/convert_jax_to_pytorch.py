"""Convert a JAX FoveaVLA (or π₀.₅) checkpoint to the PyTorch `FoveaVLAPytorch` layout.

FoveaVLA's parameter tree is π₀.₅'s, so this reuses the slicing functions of openpi's
`examples/convert_jax_model_to_pytorch.py` (pass the openpi checkout with --openpi-root or OPENPI_ROOT).

    python scripts/convert_jax_to_pytorch.py --config-name foveavla_so101_single \
        --checkpoint-dir checkpoints/foveavla_so101_single/so101_bs32_20k/19999 \
        --output-dir checkpoints/foveavla_so101_single/so101_bs32_20k_pytorch/19999
"""

import dataclasses
import importlib.util
import os
import pathlib
import shutil

import numpy as np
import safetensors.torch
import torch
import tyro

import foveavla.training.config as _config
from foveavla.models_pytorch.foveavla_pytorch import FoveaVLAPytorch


@dataclasses.dataclass
class Args:
    config_name: str
    checkpoint_dir: str
    output_dir: str
    openpi_root: str = os.environ.get("OPENPI_ROOT", str(pathlib.Path(__file__).resolve().parents[3] / "openpi"))
    precision: str = "float32"


class _PaliGemmaShape:
    vision_config = type("V", (), dict(hidden_size=1152, num_hidden_layers=27, num_attention_heads=16,
                                       intermediate_size=4304, patch_size=14, projection_dim=2048))()
    text_config = type("T", (), dict(hidden_size=2048, num_hidden_layers=18, num_attention_heads=8, head_dim=256,
                                     intermediate_size=16384))()


def jax_to_state_dict(checkpoint_dir: str, openpi_root: str) -> dict[str, torch.Tensor]:
    spec = importlib.util.spec_from_file_location(
        "openpi_convert", pathlib.Path(openpi_root) / "examples/convert_jax_model_to_pytorch.py"
    )
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    import openpi.models.gemma

    params = conv.slice_initial_orbax_checkpoint(checkpoint_dir=checkpoint_dir, restore_precision="float32")
    paligemma, expert = conv.slice_paligemma_state_dict(params["paligemma_params"], _PaliGemmaShape())
    # openpi decides "adaRMS (π₀.₅) or plain RMSNorm" from the checkpoint path; FoveaVLA always uses adaRMS.
    gemma = conv.slice_gemma_state_dict(
        expert, openpi.models.gemma.get_config("gemma_300m"), num_expert=1, checkpoint_dir="pi05", pi05=True
    )
    proj = {}
    for key in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        p = params["projection_params"][key]
        proj[f"{key}.weight"] = torch.from_numpy(np.asarray(p["kernel"])).T.contiguous()
        proj[f"{key}.bias"] = torch.from_numpy(np.asarray(p["bias"]))
    out = {**paligemma, **gemma, **proj}
    return {k: (v if isinstance(v, torch.Tensor) else torch.from_numpy(np.asarray(v))) for k, v in out.items()}


def main(args: Args):
    config = _config.get_config(args.config_name)
    model = FoveaVLAPytorch(config.model, precision=args.precision)
    state = jax_to_state_dict(os.path.abspath(args.checkpoint_dir), args.openpi_root)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # PaliGemma's lm_head is tied to the input embedding, so it loads with it.
    current = model.state_dict()
    loaded = {current[k].data_ptr() for k in state if k in current}
    missing = [k for k in missing if current[k].data_ptr() not in loaded]
    if missing or unexpected:
        raise RuntimeError(f"missing={missing[:10]} unexpected={unexpected[:10]}")
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_model(model, out / "model.safetensors")
    assets = pathlib.Path(args.checkpoint_dir) / "assets"
    if assets.exists():
        shutil.copytree(assets, out / "assets", dirs_exist_ok=True)
    print(f"saved {out / 'model.safetensors'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
