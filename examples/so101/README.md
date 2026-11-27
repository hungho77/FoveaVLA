# FoveaVLA on SO101 (drawer + red block)

These scripts port FoveaVLA-Single to one SO101 left-arm task,
[`hungho77/so101-left_arm_task_drawer_placing_red_squre_block`](https://huggingface.co/datasets/hungho77/so101-left_arm_task_drawer_placing_red_squre_block)
(LeRobot v3.0, 57 episodes, 30 fps, cameras `top`, `wrist` and `top_depth`). They also cover the two π0.5 baselines
the port is compared against. Checkpoints, eval outputs and the interaction boxes are on
[`twanghcmut/foveavla-so101-drawer-red-block`](https://huggingface.co/twanghcmut/foveavla-so101-drawer-red-block).

| run | config | repo | held-out MAE |
|---|---|---|---|
| A: π0.5, top + wrist | `pi05_so101_drawer_top_wrist` | [hungho77/openpi](https://github.com/hungho77/openpi) | 2.69 |
| B: π0.5, top only | `pi05_so101_drawer_top` | hungho77/openpi | 3.22 |
| C: FoveaVLA-Single, JAX | `foveavla_so101_single` | this repo | 3.39 |
| C': FoveaVLA-Single, PyTorch fp32 | `foveavla_so101_single` | this repo | 3.37 |

All four start from `pi05_base` and train at batch 32 for 20k steps. They hold out episodes 2, 15, 17, 28, 34 and 44
(`SO101_HELDOUT_EPISODES`) and skip 5 and 24, whose gripper never moves.

## Files

| file | step |
|---|---|
| `label_boxes.py` | Qwen3-VL-8B boxes (arm base, gripper, cube, drawer) on every 3rd top frame |
| `qa_sheet.py` | contact sheet of the raw labels |
| `build_boxes.py` | cleans the tracks, derives the phases (pull, pick, place, push) from the gripper joint, writes the per-frame interaction box |
| `qa_final.py` | contact sheet of the interaction boxes and the foveal crops they give |
| `build_store.py` | 224×224 store: letterboxed top, foveal crop, wrist, metric depth and its validity, plus `meta.parquet` |
| `eval_foveavla.py` | open-loop eval of C/C' on the held-out episodes, with crops from the model's own boxes (`self`) or the labels (`gt`) |
| `eval_pi05.py` | the same eval for A/B |
| `check_pt_policy.py` | JAX vs PyTorch policy on one frame with the same noise |

## Pipeline

Run from the repo root:

```bash
export PYTHONPATH=src:packages/foveavla-client/src:examples/so101
export DATA=/path/to/hungho77/so101-left_arm_task_drawer_placing_red_squre_block
export FOVEAVLA_SO101_STORE=/path/to/foveavla_store  # read by foveavla_so101_single
export FOVEAVLA_BASE_CHECKPOINT=$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_base
```

This reproduction ran the scripts with the openpi virtualenv (JAX, PyTorch, `av`, `pandas`, `transformers` for
Qwen3-VL) and the `PYTHONPATH` above.

**1. Interaction boxes.** You can skip this step by downloading the built boxes from `code/so101_port/boxes/` of the
HF repo into `examples/so101/boxes/`. To rebuild them:

```bash
python examples/so101/label_boxes.py --root $DATA --out labels_raw   # resumable: skips finished episodes
python examples/so101/qa_sheet.py $DATA labels_raw qa_raw.jpg 20
python examples/so101/build_boxes.py --root $DATA --labels labels_raw --out examples/so101/boxes
python examples/so101/qa_final.py $DATA examples/so101/boxes qa_0_1_10.jpg 0,1,10 8
```

Check the QA sheets before you build the store. The phase split and the box cleaning are heuristics tuned on this
task, so `examples/so101/boxes/summary.csv` lists any episode that has no holding segment.

**2. Store.**

```bash
python examples/so101/build_store.py --root $DATA --boxes examples/so101/boxes --out $FOVEAVLA_SO101_STORE --workers 6
```

**3. C (JAX) and C' (PyTorch).**

```bash
python scripts/compute_norm_stats.py --config-name foveavla_so101_single
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train.py foveavla_so101_single --exp-name=so101_bs32_20k \
    --no-wandb-enabled
python scripts/train_pytorch.py foveavla_so101_single --exp-name so101_bs32_20k_pt --no-wandb-enabled \
    --pytorch-weight-path /path/to/pi05_base_pytorch     # float32, 4 x 8 accumulated (docs/pytorch_training.md)
```

**4. A and B.** Run these in a checkout of [hungho77/openpi](https://github.com/hungho77/openpi) `main`, with
`SO101_DRAWER_STORE` set to the same store:

```bash
python scripts/compute_norm_stats.py --config-name pi05_so101_drawer_top_wrist   # and pi05_so101_drawer_top
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train.py pi05_so101_drawer_top_wrist --exp-name=so101_bs32_20k \
    --no-wandb-enabled
```

**5. Eval.** Each eval queries every 8 frames and reports action-chunk MAE against the recorded absolute actions. For C
and C', it also reports the IoU of the predicted future box.

```bash
python examples/so101/eval_foveavla.py --root $DATA --boxes examples/so101/boxes \
    --checkpoint checkpoints/foveavla_so101_single/so101_bs32_20k/19999 --out eval/C
# in the openpi checkout:
python /path/to/FoveaVLA/examples/so101/eval_pi05.py --config pi05_so101_drawer_top_wrist \
    --checkpoint checkpoints/pi05_so101_drawer_top_wrist/so101_bs32_20k/19999 --store $SO101_DRAWER_STORE --out eval/A
```

The eval runs on 6 held-out episodes of one task. It measures action error, not task success.
