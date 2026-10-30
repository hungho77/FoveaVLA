# Token-efficient variants

FoveaVLA-Single sends 1,024 image tokens to the language model per query (full and foveal RGB, plus their depth), and Dual sends 2,048. Two options reduce this budget without adding parameters, so the π₀.₅ base and released FoveaVLA checkpoints load unchanged:

| Option | Model config field | Effect per full view |
|---|---|---|
| Drop letterbox padding rows | `full_view_patch_rows=(3, 13)` | A 16:9 view letterboxed to 224×224 has content only in SigLIP patch rows 3–12; the 96 pure-padding patches are dropped after the position embedding, so kept patches keep their pretrained positions. 256 → 160 RGB tokens. |
| Pool depth tokens | `depth_token_pool=2` | Valid depth tokens are averaged over 2×2 patch windows after SigLIP. A window stays visible if any of its patches is valid. Full depth: 160 → 40; foveal depth: 256 → 64. |

Foveal crops are square and keep all 256 RGB tokens. Box coordinates are unchanged: they still refer to the padded 224×224 canvas.

## Configs and cost

`foveavla_single_efficient` and `foveavla_dual_efficient` enable both options. Per-query model FLOPs (batch 1, text prefix padded to 200 tokens; online stereo depth excluded):

| Config | Image tokens | SigLIP | LLM prefix | 10 flow steps | Box decode | Total |
|---|---|---|---|---|---|---|
| `foveavla_single` | 1,024 | 886 G | 5,072 G | 131 G | 12 G | 6.10 T |
| `foveavla_single_efficient` | 520 | 716 G | 2,930 G | 119 G | 12 G | 3.78 T (62%) |
| `foveavla_dual` | 2,048 | 1,771 G | 9,655 G | 155 G | 30 G | 11.61 T |
| `foveavla_dual_efficient` | 1,040 | 1,432 G | 5,142 G | 132 G | 29 G | 6.73 T (58%) |

Reproduce the table, and time each stage on a GPU:

```bash
uv run --no-sync scripts/profile_cost.py
uv run --no-sync scripts/profile_cost.py --measure --checkpoint=checkpoints/foveavla_single/params
```

FLOPs are matmul and convolution FLOPs counted from each stage's jaxpr, with scanned layer stacks multiplied by their depth. `--measure` reports median latency of the prefix, action, and box stages on the default JAX device; without `--checkpoint` it uses random weights, which time identically.

## Train

Train like the base models (see [training](training.md)):

```bash
uv run --no-sync scripts/train.py foveavla_single_efficient --exp-name=efficient_run
```

To warm-start from a released FoveaVLA checkpoint instead of π₀.₅ base, point `FOVEAVLA_BASE_CHECKPOINT` at its directory (it contains `params/` and `assets/droid/norm_stats.json`).

Each option can be ablated on its own by overriding a base config:

```bash
uv run --no-sync scripts/train.py foveavla_single --exp-name=trim_only --model.full-view-patch-rows 3 13
uv run --no-sync scripts/train.py foveavla_single --exp-name=depth_pool_only --model.depth-token-pool=2
```

## Serve

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 uv run --no-sync scripts/serve.py \
  --model=foveavla_single_efficient --checkpoint=checkpoints/foveavla_single_efficient/efficient_run/99999
```

Serving a released checkpoint under an efficient config (`--checkpoint=checkpoints/foveavla_single`) runs it zero-shot with the reduced token budget. This needs no retraining, but the model was not trained on reduced inputs, so treat it as a lower bound.

The robot client is unchanged.
