"""Report FoveaVLA per-query model cost: prefix tokens, FLOPs per inference stage, and optional latency.

FLOPs are matmul and convolution FLOPs counted from each stage's jaxpr, so no weights or accelerator are needed:

    uv run --no-sync scripts/profile_cost.py
    uv run --no-sync scripts/profile_cost.py --models foveavla_single foveavla_single_efficient

Add --measure to time the stages on the default JAX device (random weights unless --checkpoint is set; the
efficient configs have the same parameters, so a released checkpoint can be timed under every config):

    uv run --no-sync scripts/profile_cost.py --measure --checkpoint=checkpoints/foveavla_single/params

Online Fast-FoundationStereo depth runs in PyTorch before the model and is not included.
"""

import argparse
from collections.abc import Callable
import json
import math
from pathlib import Path
import statistics
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from foveavla.models import model as _model
from foveavla.shared import array_typing as at
from foveavla.training import config as _config


def _subjaxprs(params: dict):
    for value in params.values():
        for item in value if isinstance(value, list | tuple) else (value,):
            if hasattr(item, "eqns"):
                yield item
            elif hasattr(getattr(item, "jaxpr", None), "eqns"):
                yield item.jaxpr


def _jaxpr_flops(jaxpr) -> float:
    """Count matmul and convolution FLOPs, multiplying scanned layer stacks by their length."""
    total = 0.0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            lhs, rhs = (var.aval.shape for var in eqn.invars)
            (_, rhs_contract), (_, rhs_batch) = eqn.params["dimension_numbers"]
            rhs_free = math.prod(size for dim, size in enumerate(rhs) if dim not in (*rhs_contract, *rhs_batch))
            total += 2.0 * math.prod(lhs) * rhs_free
        elif eqn.primitive.name == "conv_general_dilated":
            out = eqn.outvars[0].aval.shape
            kernel = eqn.invars[1].aval.shape
            out_features = kernel[eqn.params["dimension_numbers"].rhs_spec[0]]
            total += 2.0 * math.prod(out) * math.prod(kernel) / out_features
        repeats = eqn.params.get("length", 1) if eqn.primitive.name == "scan" else 1
        branches = [_jaxpr_flops(sub) for sub in _subjaxprs(eqn.params)]
        if eqn.primitive.name == "cond":
            total += max(branches, default=0.0)
        else:
            total += repeats * sum(branches)
    return total


def _flops(fn: Callable, *args) -> float:
    # While loops count once; callers scale the flow loop by its step count.
    with at.disable_typechecking():
        closed = jax.make_jaxpr(fn)(*args)
    return _jaxpr_flops(closed.jaxpr)


def _as_bfloat16(tree):
    def cast(x):
        if jnp.issubdtype(x.dtype, jnp.floating):
            return (
                jax.ShapeDtypeStruct(x.shape, jnp.bfloat16)
                if isinstance(x, jax.ShapeDtypeStruct)
                else x.astype(jnp.bfloat16)
            )
        return x

    return jax.tree.map(cast, tree)


def _stage_functions(graphdef, num_steps: int) -> dict[str, Callable]:
    def method(name, **kwargs):
        def fn(state, *args):
            return getattr(nnx.merge(graphdef, state), name)(*args, **kwargs)

        return fn

    def embed(state, observation):
        model = nnx.merge(graphdef, state)
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=model.image_keys)
        return model.embed_prefix(observation)

    return {
        "embed": embed,
        "prefix": method("prepare_inference_prefix"),
        "action_step": method("sample_actions_from_prepared_prefix", num_steps=1),
        "actions": method("sample_actions_from_prepared_prefix", num_steps=num_steps),
        "bbox": method("sample_future_bbox_from_prepared_prefix"),
    }


def analyze(config_name: str, num_steps: int) -> dict:
    config = _config.get_config(config_name).model
    abstract_model = nnx.eval_shape(lambda: config.create(jax.random.key(0)))
    graphdef, state = nnx.split(abstract_model)
    state = _as_bfloat16(state)
    observation = config.inputs_spec(batch_size=1)[0]
    rng = jax.ShapeDtypeStruct((), jax.random.key(0).dtype)
    stages = _stage_functions(graphdef, num_steps)
    prefix = jax.eval_shape(stages["prefix"], state, observation)

    embed = _flops(stages["embed"], state, observation)
    prefix_total = _flops(stages["prefix"], state, observation)
    # The flow loop is a while loop with a traced trip count; count one step and scale it.
    action_step = _flops(stages["action_step"], state, rng, prefix)
    bbox = _flops(stages["bbox"], state, prefix)
    image_tokens = config.image_token_counts()
    return {
        "model": config_name,
        "image_tokens": sum(image_tokens.values()),
        "image_tokens_per_stream": image_tokens,
        "prefix_tokens": prefix.tokens.shape[1],
        "gflops": {
            "siglip_and_embed": embed / 1e9,
            "llm_prefix": (prefix_total - embed) / 1e9,
            "actions": num_steps * action_step / 1e9,
            "action_path": (prefix_total + num_steps * action_step) / 1e9,
            "bbox_decode": bbox / 1e9,
            "total": (prefix_total + num_steps * action_step + bbox) / 1e9,
        },
    }


def measure(config_name: str, num_steps: int, iters: int, checkpoint: Path | None) -> dict[str, float]:
    config = _config.get_config(config_name).model
    if checkpoint is not None:
        model = config.load(_model.restore_params(checkpoint, dtype=jnp.bfloat16))
        graphdef, state = nnx.split(model)
    else:
        graphdef = nnx.graphdef(nnx.eval_shape(lambda: config.create(jax.random.key(0))))
        state = jax.jit(lambda rng: _as_bfloat16(nnx.state(config.create(rng))))(jax.random.key(0))
    stages = {name: jax.jit(fn) for name, fn in _stage_functions(graphdef, num_steps).items()}
    observation = config.fake_obs()
    rng = jax.random.key(1)

    def timed(fn: Callable, *args) -> tuple[float, object]:
        result = jax.block_until_ready(fn(*args))
        samples = []
        for _ in range(iters):
            started = time.perf_counter()
            result = jax.block_until_ready(fn(*args))
            samples.append((time.perf_counter() - started) * 1000)
        return statistics.median(samples), result

    prefix_ms, prefix = timed(stages["prefix"], state, observation)
    actions_ms, _ = timed(stages["actions"], state, rng, prefix)
    bbox_ms, _ = timed(stages["bbox"], state, prefix)
    return {
        "prefix": prefix_ms,
        "actions": actions_ms,
        "action_path": prefix_ms + actions_ms,
        "bbox_decode": bbox_ms,
    }


def _print_table(rows: list[dict]) -> None:
    baseline = rows[0]
    header = f"{'model':<28}{'img tok':>8}{'prefix':>8}  " + "".join(f"{key:>18}" for key in baseline["gflops"])
    print("GFLOPs per query (batch 1; text prefix padded to max_token_len)")
    print(header)
    for row in rows:
        cells = "".join(
            f"{value:>10.1f} ({value / baseline['gflops'][key]:>4.0%})" for key, value in row["gflops"].items()
        )
        print(f"{row['model']:<28}{row['image_tokens']:>8}{row['prefix_tokens']:>8}  {cells}")
    if "latency_ms" in baseline:
        print(f"\nMedian latency in ms on {jax.devices()[0].device_kind}")
        print(f"{'model':<28}" + "".join(f"{key:>14}" for key in baseline["latency_ms"]))
        for row in rows:
            print(f"{row['model']:<28}" + "".join(f"{value:>14.1f}" for value in row["latency_ms"].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=_config.CONFIG_NAMES, default=list(_config.CONFIG_NAMES))
    parser.add_argument("--num-steps", type=int, default=10, help="Flow integration steps")
    parser.add_argument("--measure", action="store_true", help="Also time each stage on the default JAX device")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--checkpoint", type=Path, help="params/ directory used for --measure")
    parser.add_argument("--output", type=Path, help="Write the report as JSON")
    args = parser.parse_args()

    rows = []
    for name in args.models:
        row = analyze(name, args.num_steps)
        if args.measure:
            row["latency_ms"] = measure(name, args.num_steps, args.iters, args.checkpoint)
        rows.append(row)
    _print_table(rows)
    if args.output is not None:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
