# PyTorch training

`scripts/train_pytorch.py` trains FoveaVLA in PyTorch from the same configs and data pipeline as the JAX trainer
(`scripts/train.py`). The model is `foveavla.models_pytorch.foveavla_pytorch.FoveaVLAPytorch`, a port of
`foveavla.models.foveavla.FoveaVLA` built on openpi's PyTorch π₀.₅.

## Requirements

- `openpi` importable, with transformers 4.53.2 and openpi's `transformers_replace` copied over it. This is the same
  setup openpi's own PyTorch trainer needs.
- Initial weights in PyTorch layout. openpi's `pi05_base_pytorch` loads directly, because parameter names match
  `PI0Pytorch(pi05=True)`.

## Train

```bash
python scripts/train_pytorch.py foveavla_so101_single --exp-name run_pt \
    --pytorch-weight-path /path/to/pi05_base_pytorch            # add --resume to continue
torchrun --standalone --nproc_per_node=2 scripts/train_pytorch.py ...   # multi-GPU, SO101 store only
```

Everything below follows the JAX trainer:

| | |
|---|---|
| Loss | Flow matching + `bbox_loss_weight` × location-token cross-entropy, masked exactly as in JAX |
| Prefix | RGB views, then depth views (patch-masked SigLIP), then the prompt with the box tokens. The action branch cannot attend to the box tokens, and its positions skip them |
| Augmentation | augmax `ColorJitter(0.3, 0.4, 0.5)`, one draw per sample shared across views; depth-view dropout 0.1 |
| Optimizer | AdamW (b1, b2, eps, weight decay from the config), global-norm clip, warmup + cosine schedule |
| EMA | `ema_decay` with an fp32 shadow copy. `model.safetensors` holds the EMA weights, just as JAX saves EMA as `params` |
| Checkpoints | `<step>/model.safetensors`, `assets/`, and `train_state/` (raw weights + optimizer). Retention follows `keep_period` |

### Precision

`--pytorch-training-precision` controls how weights are stored:
- `bfloat16` (default): weights are stored in bf16, as openpi's PyTorch trainer does.
- `float32`: weights stay in fp32 and the forward runs under bf16 autocast. This matches the JAX trainer more
  closely but uses more memory.

## Convert a JAX checkpoint

```bash
python scripts/convert_jax_to_pytorch.py --config-name foveavla_so101_single \
    --checkpoint-dir checkpoints/foveavla_so101_single/<exp>/<step> --output-dir <out> --openpi-root /path/to/openpi
```

## Check against JAX

`scripts/compare_jax_pytorch.py` loads one JAX checkpoint into both implementations and runs them on a real batch, in
fp32 with fixed noise and flow time. It compares the per-step loss, the box loss, the sampled action chunk and the
decoded box.

## Inference

`FoveaVLAPytorch` provides:
- `prepare_inference_prefix(obs)`: encodes the observation once.
- `sample_actions(obs, prefix=...)`: 10 Euler steps.
- `sample_future_bbox(obs, prefix=...)`: greedy decode of the 4 location tokens. It leaves the prefix cache
  unchanged, so both calls can share one prefix.
