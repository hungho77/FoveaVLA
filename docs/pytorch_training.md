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
| Resume | Restores weights, optimizer and EMA. The data stream restarts from its seed, as in the JAX trainer |

### Precision and memory

`--pytorch-training-precision` controls how weights are stored:
- `float32` (default): fp32 weights, gradients, AdamW state and EMA, with the forward under bf16 autocast. This
  matches the JAX trainer.
- `bfloat16`: bf16 weights and AdamW state, as in openpi's PyTorch trainer. It uses about half the memory, but an
  update smaller than half a bf16 step leaves the weight unchanged. At the warmup learning rate (2.5e-6) only about
  2% of the weights move in a step, and at 5e-5 about a third. On the SO101 pilot it learned much more slowly than
  JAX.

In `float32` the weights, gradients, AdamW state and EMA take 54 GiB plus 13.5 GiB, which leaves little room for
activations on an 80GB GPU. `--pytorch-gradient-accumulation-steps N` splits each batch into N equal micro-batches.
The loss and gradient stay the same as for the full batch. `foveavla_so101_single` uses 4.

SO101 pilot, batch 32, one H100, first 100 steps from `pi05_base`, same data order as the JAX pilot:

| | s/step | Peak memory | Step 90 action loss | Step 90 box loss | Step 90 box IoU |
|---|---|---|---|---|---|
| JAX (fp32 weights, bf16 compute) | 1.8 | | 0.028 | 5.73 | 0.32 |
| PyTorch `float32`, 4 micro-batches | 3.0 | 72 GiB | 0.030 | 5.86 | 0.25 |
| PyTorch `bfloat16` | 2.2 | 47 GiB | 0.050 | 6.84 | 0.02 |

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

`policy_config.create_trained_policy` loads either kind of checkpoint. A directory with `model.safetensors` gets a
`PytorchPolicy`, which uses bf16 weights, the same transforms, and the same `infer()` outputs as the JAX `Policy`. On
a held-out SO101 frame, run C (JAX) and its PyTorch conversion with the same noise predict the same box. Their
actions differ by 0.04 on average, where the open-loop MAE is about 3.

`FoveaVLAPytorch` provides:
- `prepare_inference_prefix(obs)`: encodes the observation once.
- `sample_actions(obs, prefix=...)`: 10 Euler steps.
- `sample_future_bbox(obs, prefix=...)`: greedy decode of the 4 location tokens. It leaves the prefix cache
  unchanged, so both calls can share one prefix.
