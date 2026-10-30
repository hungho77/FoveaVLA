"""Check that FoveaVLAPytorch reproduces the JAX FoveaVLA on one checkpoint and one real batch.

Both models run in float32 with the same noise and flow time and no augmentation (train=False). Compared:
per-step action loss, per-sample box loss, the 10-step sampled action chunk, and the greedily decoded box.

    JAX_PLATFORMS=cpu python scripts/compare_jax_pytorch.py --config-name foveavla_so101_single \
        --checkpoint-dir checkpoints/foveavla_so101_single/so101_bs32_20k/19999
"""

import dataclasses
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import torch
import tyro

import foveavla.models.foveavla as _jax_foveavla
import foveavla.models.model as _model
import foveavla.training.config as _config
import foveavla.training.data_loader as _data_loader
from foveavla.models_pytorch.foveavla_pytorch import FoveaVLAPytorch


@dataclasses.dataclass
class Args:
    config_name: str
    checkpoint_dir: str
    batch_size: int = 2
    openpi_root: str = str(pathlib.Path(__file__).resolve().parents[3] / "openpi")


def jax_outputs(model, batch, noise, time):
    obs = _model.Observation.from_dict(jax.tree.map(jnp.asarray, batch))
    actions = jnp.asarray(batch["actions"])
    obs = _model.preprocess_observation(None, obs, train=False, image_keys=model.image_keys)
    t = time[:, None, None]
    x_t = t * noise + (1 - t) * actions
    u_t = noise - actions
    prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs, train=False)
    suffix_tokens, suffix_mask, suffix_ar, adarms = model.embed_suffix(obs, x_t, time)
    action_prefix = model._action_prefix_mask(prefix_mask, prefix_ar)  # noqa: SLF001
    mask = _jax_foveavla.make_joint_attn_mask(prefix_mask, prefix_ar, action_prefix, suffix_mask, suffix_ar)
    positions = _jax_foveavla.make_joint_positions(prefix_mask, action_prefix, suffix_mask)
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=mask, positions=positions, adarms_cond=[None, adarms]
    )
    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
    action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
    bbox_loss, _ = model._compute_bbox_loss(obs, prefix_out)  # noqa: SLF001
    return np.asarray(action_loss), np.asarray(bbox_loss)


def main(args: Args):
    config = _config.get_config(args.config_name)
    model_cfg = dataclasses.replace(config.model, dtype="float32")
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_rlds_dataset(
        data_config, model_cfg.action_horizon, args.batch_size, shuffle=True, seed=1
    )
    dataset = _data_loader.transform_iterable_dataset(dataset, data_config, is_batched=True)
    batch = next(iter(dataset))
    batch = {k: v for k, v in batch.items() if k != "prompt"}

    rng = np.random.default_rng(0)
    noise = rng.standard_normal(np.asarray(batch["actions"]).shape).astype(np.float32)
    time = np.array([0.3, 0.8][: args.batch_size] + [0.5] * max(0, args.batch_size - 2), np.float32)
    sample_noise = rng.standard_normal(noise.shape).astype(np.float32)

    params = _model.restore_params(f"{args.checkpoint_dir}/params", restore_type=np.ndarray, dtype=jnp.float32)
    jmodel = model_cfg.load(params)
    del params
    j_action, j_bbox = jax_outputs(jmodel, batch, jnp.asarray(noise), jnp.asarray(time))
    jobs = _model.Observation.from_dict(jax.tree.map(jnp.asarray, batch))
    prefix = jmodel.prepare_inference_prefix(jobs)
    j_actions = np.asarray(jmodel.sample_actions_from_prepared_prefix(jax.random.key(0), prefix, noise=jnp.asarray(sample_noise)))
    j_box, j_valid = (np.asarray(x) for x in jmodel.sample_future_bbox_from_prepared_prefix(prefix))
    del jmodel, prefix

    spec = importlib.util.spec_from_file_location("convert", pathlib.Path(__file__).parent / "convert_jax_to_pytorch.py")
    convert = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(convert)
    tmodel = FoveaVLAPytorch(model_cfg, precision="float32")
    missing, unexpected = tmodel.load_state_dict(convert.jax_to_state_dict(args.checkpoint_dir, args.openpi_root), strict=False)
    assert not unexpected and all(k.endswith("lm_head.weight") for k in missing), (missing, unexpected)
    tmodel.eval()
    tobs = _model.Observation.from_dict(batch)
    tobs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)), tobs)
    tactions = torch.from_numpy(np.asarray(batch["actions"], np.float32))
    with torch.no_grad():
        losses, metrics = tmodel(tobs, tactions, noise=torch.from_numpy(noise), time=torch.from_numpy(time), train=False)
        prefix_t = tmodel.prepare_inference_prefix(tobs)
        t_actions = tmodel.sample_actions(tobs, noise=torch.from_numpy(sample_noise), prefix=prefix_t).numpy()
        t_box, t_valid = (x.numpy() for x in tmodel.sample_future_bbox(tobs, prefix=prefix_t))
    j_total = j_action + model_cfg.bbox_loss_weight * j_bbox[:, None]
    t_total = losses.numpy()

    def report(name, a, b):
        err = np.abs(a - b).max()
        rel = err / max(np.abs(a).max(), 1e-12)
        print(f"{name:28s} max|jax|={np.abs(a).max():.5g}  max|diff|={err:.3g}  rel={rel:.3g}")

    report("loss per step (action+box)", j_total, t_total)
    report("action loss (mean)", np.array(j_action.mean()), np.array(metrics["action_loss"].item()))
    report("box loss (mean)", np.array(j_bbox.mean()), np.array(metrics["bbox_loss"].item()))
    report("sampled actions (10 steps)", j_actions, t_actions)
    print("decoded box jax  ", np.round(j_box, 4).tolist(), j_valid.tolist())
    print("decoded box torch", np.round(t_box, 4).tolist(), t_valid.tolist())


if __name__ == "__main__":
    main(tyro.cli(Args))
