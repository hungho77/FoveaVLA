"""Checks for the token-efficiency options: full-view padding-row removal and depth token pooling."""

import unittest

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from foveavla_client import image_tools as client_image_tools

from foveavla.models import foveavla
from foveavla.models import foveavla_config
from foveavla.models import future_bbox
from foveavla.models import siglip
from foveavla.shared import fast_foundation_depth
from foveavla.shared import image_tools
from foveavla.training import config as _config

PATCH = foveavla_config.PATCH_SIZE


def _content_rows(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask.reshape(mask.shape[0], -1).any(axis=1))


class LetterboxRowsTest(unittest.TestCase):
    def test_dropped_rows_hold_only_padding(self):
        start, stop = future_bbox.letterbox_patch_rows(PATCH)
        self.assertEqual((start, stop), (3, 13))
        # RLDS training frames are 180x320; deployment frames are 720x1280.
        for height, width in ((180, 320), (720, 1280)):
            with self.subTest(source=(height, width)):
                source = np.full((height, width, 3), 255, np.uint8)
                for resized in (
                    np.asarray(image_tools.resize_with_pad(jnp.asarray(source), 224, 224)),
                    client_image_tools.resize_with_pad(source, 224, 224),
                ):
                    rows = _content_rows(resized[..., 0] > 0)
                    self.assertGreaterEqual(rows.min(), start * PATCH)
                    self.assertLess(rows.max(), stop * PATCH)
        *_, content = fast_foundation_depth._aligned_source_grid(1280, 720, 224, crop_bbox=None)
        rows = _content_rows(content)
        self.assertEqual((rows.min() // PATCH, rows.max() // PATCH + 1), (start, stop))


class TokenBudgetTest(unittest.TestCase):
    def test_image_token_counts(self):
        expected = {
            "foveavla_single": 1024,
            "foveavla_dual": 2048,
            "foveavla_single_efficient": 160 + 256 + 40 + 64,
            "foveavla_dual_efficient": 2 * (160 + 256 + 40 + 64),
        }
        for name, count in expected.items():
            with self.subTest(name=name):
                self.assertEqual(sum(_config.get_config(name).model.image_token_counts().values()), count)

    def test_released_configs_are_unchanged(self):
        for name in ("foveavla_single", "foveavla_dual"):
            model = _config.get_config(name).model
            self.assertIsNone(model.full_view_patch_rows)
            self.assertEqual(model.depth_token_pool, 1)

    def test_invalid_options_are_rejected(self):
        with self.assertRaises(ValueError):
            foveavla_config.FoveaVLAConfig(full_view_patch_rows=(0, 17))
        with self.assertRaises(ValueError):
            foveavla_config.FoveaVLAConfig(depth_token_pool=3)
        with self.assertRaises(ValueError):
            foveavla_config.FoveaVLAConfig(full_view_patch_rows=(3, 12), depth_token_pool=2)


class SiglipTokenRowsTest(unittest.TestCase):
    def test_kept_rows_reuse_full_grid_parameters(self):
        module = siglip.Module(num_classes=8, variant="mu/14", pool_type="none")
        image = jnp.zeros((2, 224, 224, 3))
        params = module.init(jax.random.key(0), image)
        self.assertEqual(params["params"]["pos_embedding"].shape[1], 256)

        full, _ = module.apply(params, image)
        trimmed, _ = module.apply(params, image, token_rows=(3, 13))
        self.assertEqual(full.shape, (2, 256, 8))
        self.assertEqual(trimmed.shape, (2, 160, 8))
        with self.assertRaises(ValueError):
            module.apply(params, image, token_rows=(3, 13), token_mask=jnp.ones((2, 256), bool))


class PoolMaskedTokensTest(unittest.TestCase):
    def test_masked_mean_over_windows(self):
        # A 2x4 grid pooled 2x2 gives two windows: columns 0-1 and 2-3.
        tokens = jnp.arange(8, dtype=jnp.float32).reshape(1, 8, 1)
        valid = jnp.asarray([[True, False, False, False, True, True, False, False]])
        pooled, pooled_valid = foveavla._pool_masked_tokens(tokens * valid[..., None], valid, 2, 4, 2)
        np.testing.assert_allclose(pooled[0, :, 0], [(0 + 4 + 5) / 3, 0.0])
        np.testing.assert_array_equal(pooled_valid, [[True, False]])

    def test_select_patch_rows(self):
        mask = jnp.arange(256).reshape(1, 256) % 16 == 0
        selected = foveavla._select_patch_rows(mask, (3, 13))
        self.assertEqual(selected.shape, (1, 160))
        self.assertEqual(int(selected.sum()), 10)


class EfficientPrefixTraceTest(unittest.TestCase):
    """Trace the full-size model abstractly: no weights are allocated."""

    def test_prefix_lengths_and_loss_shapes(self):
        for name in ("foveavla_single_efficient", "foveavla_dual_efficient"):
            with self.subTest(name=name):
                config = _config.get_config(name).model
                graphdef, state = nnx.split(nnx.eval_shape(lambda config=config: config.create(jax.random.key(0))))
                observation, actions = config.inputs_spec(batch_size=2)
                rng = jax.ShapeDtypeStruct((), jax.random.key(0).dtype)

                def prefix(state, observation):
                    return nnx.merge(graphdef, state).prepare_inference_prefix(observation)

                def loss(state, rng, observation, actions):
                    return nnx.merge(graphdef, state).compute_loss(rng, observation, actions, train=True)

                prefix_out = jax.eval_shape(prefix, state, observation)
                image_tokens = sum(config.image_token_counts().values())
                self.assertEqual(prefix_out.tokens.shape[1], image_tokens + config.max_token_len)
                self.assertEqual(jax.eval_shape(loss, state, rng, observation, actions).shape, (2, 16))


if __name__ == "__main__":
    unittest.main()
