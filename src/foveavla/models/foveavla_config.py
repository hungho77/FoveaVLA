"""FoveaVLA configuration: the π₀.₅ backbone with global/foveal RGB, depth, and future boxes."""

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from foveavla.models import gemma
from foveavla.models import model as _model
from foveavla.shared import array_typing as at

if TYPE_CHECKING:
    from foveavla.models.foveavla import FoveaVLA

PATCH_SIZE = 14
PATCH_GRID_SIZE = _model.IMAGE_RESOLUTION[0] // PATCH_SIZE
# Slots holding letterboxed full external views; the wrist-named slots hold square foveal crops.
FULL_VIEW_IMAGE_KEYS = ("base_0_rgb", "base_1_rgb")


@dataclasses.dataclass(frozen=True)
class FoveaVLAConfig(_model.BaseModelConfig):
    """Single and Dual share all weights; only the number of input views and boxes changes."""

    dtype: str = "bfloat16"
    paligemma_variant: gemma.Variant = "gemma_2b"
    action_expert_variant: gemma.Variant = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 16
    max_token_len: int = 200
    # Retain checkpoint slot names: wrist-named slots hold foveal crops, not wrist images.
    image_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb")
    depth_image_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb")
    depth_view_dropout_prob: float = 0.1
    future_bbox_count: int = 1
    bbox_loss_weight: float = 0.1
    # Efficiency options; the defaults reproduce the released models. Neither adds parameters, so the
    # π₀.₅ base and released FoveaVLA checkpoints load unchanged.
    # SigLIP patch rows [start, stop) kept for full views; the other rows are letterbox padding.
    full_view_patch_rows: tuple[int, int] | None = None
    # Mean-pool depth tokens over k×k patch windows before the language model.
    depth_token_pool: int = 1

    def __post_init__(self):
        if not self.depth_image_keys or not set(self.depth_image_keys).issubset(self.image_keys):
            raise ValueError("depth_image_keys must be a nonempty subset of image_keys")
        if not 0 <= self.depth_view_dropout_prob < 1:
            raise ValueError("depth_view_dropout_prob must be in [0, 1)")
        if self.future_bbox_count not in (1, 2):
            raise ValueError("FoveaVLA predicts one or two future boxes")
        if self.full_view_patch_rows is not None:
            start, stop = self.full_view_patch_rows
            if not 0 <= start < stop <= PATCH_GRID_SIZE:
                raise ValueError(f"full_view_patch_rows must satisfy 0 <= start < stop <= {PATCH_GRID_SIZE}")
        if self.depth_token_pool < 1:
            raise ValueError("depth_token_pool must be positive")
        for key in self.depth_image_keys:
            rows, columns = self.patch_grid(key)
            if rows % self.depth_token_pool or columns % self.depth_token_pool:
                raise ValueError(f"depth_token_pool must divide the {rows}x{columns} patch grid of {key}")

    def patch_rows(self, key: str) -> tuple[int, int] | None:
        """Kept SigLIP patch rows for an image slot, or None to keep all rows."""
        return self.full_view_patch_rows if key in FULL_VIEW_IMAGE_KEYS else None

    def patch_grid(self, key: str) -> tuple[int, int]:
        rows = self.patch_rows(key)
        return (PATCH_GRID_SIZE if rows is None else rows[1] - rows[0]), PATCH_GRID_SIZE

    def image_token_counts(self) -> dict[str, int]:
        """Language-model prefix tokens per visual stream, keyed by `rgb/<slot>` and `depth/<slot>`."""
        counts = {}
        for key in self.image_keys:
            rows, columns = self.patch_grid(key)
            counts[f"rgb/{key}"] = rows * columns
        for key in self.depth_image_keys:
            rows, columns = self.patch_grid(key)
            counts[f"depth/{key}"] = rows * columns // self.depth_token_pool**2
        return counts

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "FoveaVLA":
        from foveavla.models.foveavla import FoveaVLA

        return FoveaVLA(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        depth = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION], jnp.float32)
        image_mask = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        with at.disable_typechecking():
            observation = _model.Observation(
                images=dict.fromkeys(self.image_keys, image),
                image_masks=dict.fromkeys(self.image_keys, image_mask),
                depthmaps=dict.fromkeys(self.depth_image_keys, depth),
                depthmap_quality=dict.fromkeys(self.depth_image_keys, depth),
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
            )
        actions = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation, actions
