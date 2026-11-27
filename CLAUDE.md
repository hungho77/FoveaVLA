# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FoveaVLA: a π₀.₅ (OpenPI-derived) vision-language-action model that replaces the wrist camera with a learned
**foveation action**. Each query it predicts an action chunk *and* a box (4 location tokens) around the next
interaction region; at the next query that box is cropped from the full-resolution external image and fed back as a
second view alongside the full view (plus metric depth for both). Upstream is `Stanford-TML/FoveaVLA` (`origin`);
this checkout's work goes to the `hungho77/FoveaVLA` fork (`fork`), which adds an SO101 port and a PyTorch trainer.

## Commands

Environment: Linux, Python 3.11, uv. `uv sync --locked` (add `--group rlds` for DROID/RLDS training, which pulls
TensorFlow 2.15 + dlimp; `ml-dtypes`/`tensorstore` are pinned via `override-dependencies` to make that coexist with JAX).

```bash
# Tests (unittest, run from repo root; test_gripper_watchdog imports examples.droid.main)
uv run --no-sync python -m unittest discover tests
uv run --no-sync python -m unittest tests.test_efficient_tokens.LetterboxRowsTest.test_dropped_rows_hold_only_padding

# Lint (ruff config in pyproject: line-length 120, rules E4/E7/E9/F)
uv run --no-sync ruff check .

# JAX training / serving
uv run --no-sync scripts/train.py <config> --exp-name=NAME [--resume] [--batch-size=N --fsdp-devices=N]
XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 uv run --no-sync scripts/serve.py --model=<config> [--checkpoint=checkpoints/<config>/<exp>/<step>]

# PyTorch training (needs openpi importable with its transformers_replace; see docs/pytorch_training.md)
python scripts/train_pytorch.py foveavla_so101_single --exp-name NAME --pytorch-weight-path /path/to/pi05_base_pytorch
python scripts/convert_jax_to_pytorch.py --config-name ... --checkpoint-dir ... --output-dir ... --openpi-root ...
python scripts/compare_jax_pytorch.py ...   # JAX vs PyTorch loss/actions/box parity on one batch

# FLOP/latency profiling of configs
uv run --no-sync scripts/profile_cost.py [--measure --checkpoint=...]
```

SO101 scripts (`examples/so101/`) were run with the openpi virtualenv and
`PYTHONPATH=src:packages/foveavla-client/src:examples/so101`; see `examples/so101/README.md` for the full
label → boxes → store → train → eval pipeline.

## Architecture

**Layout**
- `src/foveavla/` – main package (JAX/Flax NNX model, training, policies, serving).
- `src/droid_foveation/` – offline DROID data prep: annotate interaction boxes (RoboInter), repack foveal crops into
  LMDB (`repack_direct.py`), parquet IO. Exposed as `droid-foveation-*` console scripts (with legacy `droid-roi-*` aliases).
- `packages/foveavla-client/` – uv workspace member, Python ≥3.10, no JAX/torch. Websocket client, msgpack-numpy
  protocol, and **`foveation.py` / `image_tools.py` shared by both server and robot client** — cropping/letterbox logic
  must stay identical between training, server and client, so change it here, not in a copy.
- `examples/droid/main.py` – real-robot DROID client (separate Python 3.10 env, see `docs/deployment.md`).
- `scripts/` – entry points (train, serve, data/depth preparation, checkpoint download/convert).

**Config registry** (`src/foveavla/training/config.py`): all named configs live in `_CONFIGS_DICT` —
`foveavla_single`, `foveavla_dual`, their `_efficient` variants (via `_efficient()`, drops letterbox patch rows and
2×2-pools depth tokens; no new params), and `foveavla_so101_single`. `train.py`, `train_pytorch.py`, `serve.py` and
`create_trained_policy` all resolve configs by name here; CLI overrides go through tyro. Data/weight paths come from
env vars read at import time (`FOVEAVLA_DATA_ROOT`, `FOVEAVLA_RLDS_DIR`, `FOVEAVLA_FOVEAL_DIR`,
`FOVEAVLA_DEPTH_{LEFT,RIGHT}_DIR`, `FOVEAVLA_DEPTH_FILTER`, `FOVEAVLA_BASE_CHECKPOINT`, `FOVEAVLA_SO101_STORE`).
`policy_metadata` on each config is shipped to the client and tells it which views/depth sources to use.

**Model** (`models/foveavla.py`, `foveavla_config.py`, `future_bbox.py`): prefix = RGB views, then depth views
(patch-masked SigLIP), then prompt + box location tokens. Loss = flow matching on actions + `bbox_loss_weight` ×
location-token CE. The action expert cannot attend to box tokens and its positions skip them. Image slot names keep
the π₀.₅ checkpoint names (`base_0_rgb`, `left_wrist_0_rgb`) even though the second slot holds the foveal crop — do
not rename them, or released checkpoints stop loading. Box coordinates are in the padded 224×224 letterboxed canvas.
`models_pytorch/foveavla_pytorch.py` is a parity port built on openpi's `PI0Pytorch(pi05=True)`.

**Data pipeline** (`training/data_loader.py` + datasets): DROID training joins RLDS episodes
(`droid_rlds_dataset.py`) with foveal-crop LMDBs (`droid_foveal_dataset.py`) and depth/stereo-geometry stores
(`droid_pointmap_store.py`, `droid_stereo_geometry_store.py`), filtered by sample-range JSONs from
`scripts/build_*_filter.py`. SO101 uses a self-contained 224×224 store (`so101_dataset.py`). Per-robot input/output
mapping lives in `policies/{droid,so101}_policy.py`; shared transforms in `transforms.py`.

**Normalization**: DROID configs use π₀.₅ base's `assets/droid/norm_stats.json` (downloaded with base weights and
saved into each checkpoint); SO101 uses its own stats from `scripts/compute_norm_stats.py`.
`tests/test_normalization_assets.py` guards which assets training vs. deployment pick.

**Inference/serving**: `policies/policy_config.create_trained_policy` returns a JAX `Policy` for orbax checkpoints or a
`PytorchPolicy` if the directory has `model.safetensors`. JAX `Policy` can return actions first and decode the box
afterwards (`infer_actions_with_pending_bbox` / `infer_future_bbox`) sharing one prefix cache. `serving/websocket_policy_server.py`
serves it; online depth for deployment comes from Fast-FoundationStereo (`shared/fast_foundation_*`,
installed by `scripts/setup_fast_foundation_depth.sh`).

**Backward compatibility**: fields were renamed `roi` → `foveal`/`box`; `foveation.canonicalize_fields` and env-var /
directory fallbacks (`FOVEAVLA_ROI_DIR`, `<data root>/roi`) keep released data working.
`tests/test_foveation_compatibility.py` covers this — keep legacy names readable when touching these paths.

## Docs

`docs/data.md` (data prep), `docs/training.md`, `docs/pytorch_training.md`, `docs/deployment.md`,
`docs/efficiency.md`. `docs/index.html` + assets is the project website, not developer docs.
