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

## Three tasks in one model

`foveavla_so101_task3` trains one FoveaVLA-Single on the merged left-arm dataset `hungho77/so101-left-arm-task3`
(LeRobot v3.0, 159 episodes, 133,763 frames, cameras `top`, `wrist` and `top_depth`). Each frame is prompted with its
own task:

| episodes | task | phases → interaction box |
|---|---|---|
| 0–50 | red cube out of the steel pot into the orange hollow cube | pick: gripper + pot (the cube is inside it) → place: gripper + cube + target → retreat: gripper + target |
| 51–107 | the drawer/red-block task above (identical frames) | pull → pick → place → push, unchanged |
| 108–158 | open the drawer, stack the red cube on the blue cube | pull: gripper + drawer → pick: gripper + cube → place: gripper + cube + target → retreat: gripper + target |

`label_boxes.py` picks the objects to ask for from each episode's task string, and `build_boxes.py` picks the phases
and roles from the task recorded in the label file. The drawer prompt and drawer boxes are unchanged, so the
drawer labels can be copied over with episode index + 51 instead of labeling them again. The pot and target never
move on their own, so each episode uses its median box for them. The recipe follows the π0.5 run on the same data
(`pi05_so101_leftarm_task3` in hungho77/openpi): batch 128, 6,000 steps, cosine decay over the run, PyTorch float32
on one 80GB GPU as 16 micro-batches of 8. It trains on every episode except 56 and 75, the two drawer episodes whose
gripper never moves.

Labeling needs `transformers>=4.57` (Qwen3-VL), which the openpi environment does not have. Use a separate venv with
torch, transformers, av, pandas<3 and pyarrow<20 for it.

```bash
export DATA=/path/to/hungho77/so101-left-arm-task3
export FOVEAVLA_SO101_TASK3_STORE=/path/to/foveavla_store_task3
python examples/so101/label_boxes.py --root $DATA --out labels_task3 --episodes 0-50,108-158
python examples/so101/build_boxes.py --root $DATA --labels labels_task3 --out boxes_task3
python examples/so101/qa_final.py $DATA boxes_task3 qa_pot.jpg 0,17,34,50 8
python examples/so101/build_store.py --root $DATA --boxes boxes_task3 --out $FOVEAVLA_SO101_TASK3_STORE \
    --key-prefix so101_task3 --no-wrist
python scripts/compute_norm_stats.py --config-name foveavla_so101_task3
python scripts/train_pytorch.py foveavla_so101_task3 --exp-name task3_bs128_6k \
    --pytorch-weight-path /path/to/pi05_base_pytorch --no-wandb-enabled
```

### Subtask-focused boxes

The union boxes above jump in size (they stretch with the hand-to-object distance) and 15–19% of the frames have no box,
mostly because a gripper label was dropped. `build_focus_boxes.py` builds the boxes differently: every subtask looks at
one target with a fixed window that does not include the gripper (100 px of the 640×360 view for a cube, 120 px for a
drawer focus, 125 px for the pot; `build_store.py` crops 1.5× that). At a subtask boundary the window moves to the next
target over 30 frames, eased, starting at the boundary, so a grasp or release is finished before the eye leaves it.

| task | subtask → focus |
|---|---|
| pot | pick → pot · place, retreat → orange hollow cube |
| stack | pull → drawer handle · pick → red cube · place, retreat → blue cube |
| drawer | pull → drawer handle · pick → red cube · place → where the cube ends up in the drawer · push → drawer front |

Where each focus comes from:

- **Red and blue cube**: colour (`color_track.py`). More precise than a detector and it sees the cube inside the
  translucent drawer. Clusters much smaller than the cube are ignored (the arm's red servo wires, a blue sticker).
- **Pot, orange cube**: LocateAnything-3B (`detect_objects.py`, every 5th frame, median per episode).
- **Drawer handle and drawer front**: where the gripper tip is while the arm pulls / pushes. The tip pixel is a
  quadratic ridge fit from the five arm joints, trained on the moments where it is known: on the cube at the grasp, on
  the target at the release, and on the cube's colour while it is carried (~15 px median error on held-out episodes at
  grasp / release poses, ~23 px in transit). This is the weakest focus: there is no direct evidence for the handle.
- **Subtask boundaries**: robot state. Grasp and release from the gripper, pull → pick from the closed-jaw segment
  (stack) or from the arm leaving for the grasp pose (drawer).

In the pot task the top camera does not see the cube inside the pot in about half of the episodes (the arm starts
above it), so the top-only model cannot learn that grasp; train drawer + stack without it.

```bash
python examples/so101/detect_objects.py --root $DATA --tasks labels_task3 --out det_task3 \
    --eagle /path/to/Eagle/Embodied --model /path/to/LocateAnything-3B        # needs transformers 4.57, peft, decord
python examples/so101/color_track.py --root $DATA --out color_task3
python examples/so101/build_focus_boxes.py --root $DATA --labels labels_task3 --det det_task3 --color color_task3 \
    --out boxes_focus
python examples/so101/qa_video.py $DATA boxes_focus qa_focus 0,51,108
```

### Two tasks, held-out episodes

`foveavla_so101_task2` trains drawer + stack only (episodes 51–158) on a store built from the focus boxes, with the
recipe above. It holds out five episodes per task (`SO101_TASK2_HELDOUT_EPISODES`: 66, 68, 80, 86, 96 and 132, 139,
146, 152, 158) and skips 56, 75 and 136 (no grasp), which leaves 95 training episodes. The store still holds all 159
episodes; the config picks the ones to train on.

```bash
export FOVEAVLA_SO101_TASK2_STORE=/path/to/foveavla_store_focus
python examples/so101/build_store.py --root $DATA --boxes boxes_focus --out $FOVEAVLA_SO101_TASK2_STORE \
    --key-prefix so101_task3 --no-wrist
python scripts/compute_norm_stats.py --config-name foveavla_so101_task2
python scripts/train_pytorch.py foveavla_so101_task2 --exp-name task2_bs128_6k \
    --pytorch-weight-path /path/to/pi05_base_pytorch --no-wandb-enabled
python examples/so101/eval_foveavla.py --root $DATA --boxes boxes_focus --config foveavla_so101_task2 \
    --checkpoint checkpoints/foveavla_so101_task2/task2_bs128_6k/5999 --out eval/task2 \
    --episodes 66 68 80 86 96 132 139 146 152 158 --steps 400
```

### Motion stream

A single frame does not say whether the arm is moving, which way, or how long it has been still, and the open-loop
error of the focus-box model sits at those moments (the start of a motion, closing and opening the gripper).
`foveavla_so101_task2_motion` adds a third visual stream for it, modelled on the rods of the eye: wide, coarse,
colour-blind, sensitive to change.

- **Motion image** (`foveavla.shared.motion`): signed luminance difference between the letterboxed top view and the
  same view a few frames earlier, 3×3 blurred, differences under 4 grey levels set to zero (static frames differ by
  up to ~6 before that, a moving arm by 50–110). Grey means still.
- **Tokens**: the shared SigLIP encodes it; the model keeps the 10 patch rows with image content and mean-pools 2×2,
  so the stream costs 40 prefix tokens (the other four streams take 1,024). No parameters are added, and
  `pi05_base` loads unchanged. PyTorch model only.
- **Training**: the lag is drawn per sample from 4–16 frames (the dataset computes the image from the stored full
  views, so the store is the one above). The stream is dropped in 25% of the samples, which keeps the model usable
  without it and limits the "keep doing what you were doing" shortcut that history inputs invite.
- **Inference**: `SO101MotionInputs` builds the image from the full view of the previous `infer` call, so a client
  that replans every 4–16 frames needs no change. The first call, a call more than 3 s after the last one, or a
  sample with `"reset": True` sees a still image. A client can instead send `observation/exterior_image_1_left_prev`
  or a ready `observation/motion_image`; `observation/motion_image_mask: False` hides the stream.

```bash
python scripts/train_pytorch.py foveavla_so101_task2_motion --exp-name task2_motion_bs128_6k \
    --pytorch-weight-path /path/to/pi05_base_pytorch --no-wandb-enabled
python examples/so101/eval_foveavla.py ... --config foveavla_so101_task2_motion [--motion off] [--motion-lag 8]
```
