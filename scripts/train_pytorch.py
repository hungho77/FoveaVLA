"""PyTorch training for FoveaVLA, mirroring scripts/train.py (JAX).

Same configs, data pipeline, loss (flow matching + 0.1 x future-box cross-entropy), optimizer (AdamW, global-norm
clipping), warmup + cosine LR schedule and EMA as the JAX trainer. The model is `FoveaVLAPytorch`, built on openpi's
PyTorch π₀.₅ (requires transformers 4.53.2 with openpi's `transformers_replace`).

Single GPU:
    python scripts/train_pytorch.py foveavla_so101_single --exp-name run \
        --pytorch-weight-path /path/to/pi05_base_pytorch
Multi-GPU (SO101 store only; rank r draws batches with data seed r, and rank 0 matches the JAX trainer):
    torchrun --standalone --nproc_per_node=2 scripts/train_pytorch.py foveavla_so101_single --exp-name run ...

Checkpoints: <checkpoint_dir>/<step>/model.safetensors holds the EMA weights when EMA is on (the JAX trainer also
saves EMA params as `params`), assets/ the norm stats, and train_state/ the raw weights and optimizer for --resume.
"""

import contextlib
import dataclasses
import logging
import os
import pathlib
import platform
import queue
import shutil
import threading
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist

import foveavla.models.model as _model
from foveavla.models_pytorch.foveavla_pytorch import FoveaVLAPytorch
from foveavla.models_pytorch.foveavla_pytorch import load_weights
import foveavla.shared.normalize as _normalize
import foveavla.training.config as _config
import foveavla.training.data_loader as _data_loader


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname).1s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return world_size, rank, device


def lr_at(schedule, step: int) -> float:
    """optax.warmup_cosine_decay_schedule, as used by CosineDecaySchedule."""
    warmup, peak, decay_steps, end = schedule.warmup_steps, schedule.peak_lr, schedule.decay_steps, schedule.decay_lr
    if step < warmup:
        init = peak / (warmup + 1)
        return init + (peak - init) * step / warmup
    progress = min(1.0, (step - warmup) / max(1, decay_steps - warmup))
    return end + (peak - end) * 0.5 * (1 + np.cos(np.pi * progress))


class BatchIterator:
    """FoveaVLA's batched dataset + transforms, converted to torch on a background thread."""

    def __init__(self, config: _config.TrainConfig, data_config, *, batch_size: int, seed: int, device):
        if data_config.so101_store_dir is None and dist.is_initialized() and dist.get_world_size() > 1:
            raise NotImplementedError("Multi-process data loading is only implemented for the SO101 store.")
        dataset = _data_loader.create_rlds_dataset(
            data_config, config.model.action_horizon, batch_size, shuffle=True, seed=seed
        )
        self._dataset = _data_loader.transform_iterable_dataset(dataset, data_config, is_batched=True)
        self._device = device
        self._queue = queue.Queue(maxsize=3)
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            for batch in self._dataset:
                observation = _model.Observation.from_dict(batch)
                pin = (lambda x: x.pin_memory()) if torch.cuda.is_available() else (lambda x: x)  # noqa: E731
                observation = jax.tree.map(lambda x: pin(torch.from_numpy(np.asarray(x))), observation)
                actions = pin(torch.from_numpy(np.asarray(batch["actions"], dtype=np.float32)))
                self._queue.put((observation, actions))

    def __next__(self):
        observation, actions = self._queue.get()
        move = lambda x: x.to(self._device, non_blocking=True)  # noqa: E731
        return jax.tree.map(move, observation), move(actions)


def save_checkpoint(config, step, model, optimizer, ema, data_config):
    final = config.checkpoint_dir / str(step)
    tmp = config.checkpoint_dir / f"tmp_{step}"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "train_state").mkdir(parents=True)
    names = [n for n, _ in model.named_parameters()]
    if ema is not None:
        safetensors.torch.save_file({n: t.contiguous() for n, t in zip(names, ema)}, tmp / "model.safetensors")
    safetensors.torch.save_model(model, tmp / ("train_state/model.safetensors" if ema is not None else "model.safetensors"))
    torch.save(optimizer.state_dict(), tmp / "train_state/optimizer.pt")
    torch.save({"step": step, "config": dataclasses.asdict(config)}, tmp / "train_state/metadata.pt")
    if data_config.norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(tmp / "assets" / data_config.asset_id, data_config.norm_stats)
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    # Keep the latest checkpoint and every multiple of keep_period, as the JAX checkpoint manager does.
    for d in config.checkpoint_dir.iterdir():
        if d.is_dir() and d.name.isdigit() and int(d.name) != step:
            if config.keep_period is None or int(d.name) % config.keep_period != 0:
                shutil.rmtree(d)
    logging.info(f"saved checkpoint {final}")


def latest_step(checkpoint_dir: pathlib.Path) -> int | None:
    steps = [int(d.name) for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    return max(steps) if steps else None


def main(config: _config.TrainConfig):
    init_logging()
    # float32 training fills most of an 80GB GPU, so avoid fragmentation; must be set before CUDA initializes.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    world_size, rank, device = setup_ddp()
    is_main = rank == 0
    if config.batch_size % (world_size * config.pytorch_gradient_accumulation_steps):
        raise ValueError(
            f"batch size {config.batch_size} is not divisible by {world_size} processes x "
            f"{config.pytorch_gradient_accumulation_steps} accumulation steps"
        )
    torch.manual_seed(config.seed + rank)
    np.random.seed(config.seed + rank)

    ckpt_dir = config.checkpoint_dir
    if config.resume:
        start = latest_step(ckpt_dir) if ckpt_dir.exists() else None
        if start is None:
            raise FileNotFoundError(f"nothing to resume in {ckpt_dir}")
    else:
        if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
            if not config.overwrite:
                raise FileExistsError(f"{ckpt_dir} exists; pass --overwrite or --resume")
            if is_main:
                shutil.rmtree(ckpt_dir)
        start = None
    if dist.is_initialized():
        dist.barrier()
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if config.wandb_enabled:
            import wandb

            wandb.init(name=config.exp_name, project=config.project_name, config=dataclasses.asdict(config))

    data_config = config.data.create(config.assets_dirs, config.model)
    batches = BatchIterator(
        config, data_config, batch_size=config.batch_size // world_size, seed=rank, device=device
    )

    fp32 = config.pytorch_training_precision == "float32"
    model = FoveaVLAPytorch(config.model, precision=config.pytorch_training_precision)
    if start is not None:
        load_weights(model, ckpt_dir / str(start) / ("train_state/model.safetensors"
                                                     if config.ema_decay is not None else "model.safetensors"))
    elif config.pytorch_weight_path is not None:
        load_weights(model, pathlib.Path(config.pytorch_weight_path) / "model.safetensors")
    else:
        logging.warning("no --pytorch-weight-path: training from random initialization")
    model.to(device)
    model.gradient_checkpointing_enable()
    params = [p for p in model.parameters()]

    opt_cfg = config.optimizer
    optimizer = torch.optim.AdamW(
        params, lr=lr_at(config.lr_schedule, 0), betas=(opt_cfg.b1, opt_cfg.b2), eps=opt_cfg.eps,
        weight_decay=opt_cfg.weight_decay,
        # The fused kernel updates in place; the default foreach path allocates parameter-sized temporaries.
        fused=device.type == "cuda",
    )
    ema = None
    if config.ema_decay is not None:
        ema = [p.detach().to(torch.float32).clone() for p in params]
    step = 0
    if start is not None:
        optimizer.load_state_dict(torch.load(ckpt_dir / str(start) / "train_state/optimizer.pt", map_location=device))
        if ema is not None:
            saved = safetensors.torch.load_file(str(ckpt_dir / str(start) / "model.safetensors"), device=str(device))
            ema = [saved[n].to(torch.float32) for n, _ in model.named_parameters()]
        step = start + 1
        logging.info(f"resumed from step {start}")

    ddp = model
    if world_size > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)
    model.train()
    autocast = (
        # No cast cache: it would keep a bf16 copy of every weight alive through the backward pass.
        torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
        if fp32 and device.type == "cuda"
        else contextlib.nullcontext()
    )

    if is_main:
        n_params = sum(p.numel() for p in params)
        logging.info(
            f"FoveaVLA PyTorch on {platform.node()}: {n_params / 1e9:.2f}B params, precision={config.pytorch_training_precision}, "
            f"world={world_size}, batch={config.batch_size}, accumulation={config.pytorch_gradient_accumulation_steps}, steps={config.num_train_steps}, ema={config.ema_decay}"
        )

    infos, t0 = [], time.time()
    while step < config.num_train_steps:
        observation, actions = next(batches)
        lr = lr_at(config.lr_schedule, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        # Gradient accumulation over equal micro-batches gives the same mean loss and gradient as the full batch.
        n_micro = config.pytorch_gradient_accumulation_steps
        loss, micro_metrics = 0.0, []
        for i in range(n_micro):
            micro = lambda x: x.chunk(n_micro)[i]  # noqa: B023, E731
            sync = contextlib.nullcontext() if i == n_micro - 1 or world_size == 1 else ddp.no_sync()
            with sync, autocast:
                losses, metrics = ddp(jax.tree.map(micro, observation), micro(actions), train=True)
                micro_loss = losses.mean() / n_micro
            with sync:
                micro_loss.backward()
            loss += micro_loss.detach()
            micro_metrics.append(metrics)
        metrics = {k: torch.stack([m[k].float() for m in micro_metrics]).mean() for k in micro_metrics[0]}
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=opt_cfg.clip_gradient_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            with torch.no_grad():
                # In chunks, so bf16 weights are upcast a few at a time rather than all at once.
                for i in range(0, len(params), 64):
                    chunk = [p.detach().to(torch.float32) for p in params[i : i + 64]]
                    torch._foreach_lerp_(ema[i : i + 64], chunk, 1 - config.ema_decay)  # noqa: SLF001

        if is_main:
            infos.append({"loss": loss, "grad_norm": grad_norm.detach(), **metrics})
            if step - (start or -1) <= 2 and device.type == "cuda":
                logging.info(f"step {step}: peak GPU memory {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB")
            if step % config.log_interval == 0:
                avg = {k: torch.stack([i[k].float() for i in infos]).mean().item() for k in infos[0]}
                avg["lr"] = lr
                avg["s_per_step"] = (time.time() - t0) / len(infos)
                logging.info(f"Step {step}: " + ", ".join(f"{k}={v:.4g}" for k, v in avg.items()))
                if config.wandb_enabled:
                    import wandb

                    wandb.log(avg, step=step)
                infos, t0 = [], time.time()
            if (step % config.save_interval == 0 and step > (start or 0)) or step == config.num_train_steps - 1:
                save_checkpoint(config, step, model, optimizer, ema, data_config)
        step += 1

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main(_config.cli())
