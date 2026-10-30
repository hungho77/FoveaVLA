"""PyTorch FoveaVLA: a line-by-line port of `foveavla.models.foveavla.FoveaVLA` (JAX).

The backbone is openpi's PyTorch π₀.₅ (`openpi.models_pytorch.gemma_pytorch.PaliGemmaWithExpertModel`, which needs
transformers 4.53.2 with openpi's `transformers_replace` installed). Parameter names match openpi's `PI0Pytorch` with
`pi05=True`, so `pi05_base_pytorch/model.safetensors` loads directly and JAX FoveaVLA checkpoints convert with
`scripts/convert_jax_to_pytorch.py`.

What FoveaVLA adds over π₀.₅, all mirrored from the JAX model:
* metric depth for every view in `depth_image_keys`, colorized (fixed log range + Turbo) and encoded by the shared
  SigLIP with a patch mask (`_metric_depth_to_siglip_image`, `embed_image_masked`); training drops whole depth views
  with probability `depth_view_dropout_prob`;
* four PaliGemma location tokens (the t+8 interaction box) at the end of the prompt, autoregressive within the prefix
  (`token_ar_mask`), trained with cross-entropy over the 1024 location rows of the tied embedding table;
* action tokens never attend to the box tokens, and their positions skip them (`make_joint_attn_mask`,
  `make_joint_positions`);
* training-time color jitter is augmax's `ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)` (default hue 0.1,
  p 0.5; augmax's saturation step is a no-op), with no geometric augmentation.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from foveavla.models import future_bbox
import foveavla.models.foveavla as _jax_foveavla
import foveavla.models.gemma as _gemma
from foveavla.models.foveavla_config import FoveaVLAConfig
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel

_NEG = -2.3819763e38  # additive mask value used by openpi's PyTorch π₀.₅
_TURBO = torch.tensor(_jax_foveavla._TURBO_POLYNOMIAL_COEFFICIENTS, dtype=torch.float32)  # noqa: SLF001


def make_attn_mask(input_mask: torch.Tensor, mask_ar: torch.Tensor) -> torch.Tensor:
    """big_vision prefix-LM mask, as `foveavla.models.foveavla.make_attn_mask`."""
    mask_ar = mask_ar.expand_as(input_mask)
    cumsum = torch.cumsum(mask_ar.to(torch.int32), dim=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] & input_mask[:, :, None]
    return attn_mask & valid_mask


def make_joint_attn_mask(prefix_mask, prefix_ar_mask, action_prefix_mask, suffix_mask, suffix_ar_mask):
    """VLM/box branch over the prefix; the action branch sees the prefix minus the box tokens."""
    prefix_len = prefix_mask.shape[1]
    total = prefix_len + suffix_mask.shape[1]
    combined = torch.zeros(prefix_mask.shape[0], total, total, dtype=torch.bool, device=prefix_mask.device)
    combined[:, :prefix_len, :prefix_len] = make_attn_mask(prefix_mask, prefix_ar_mask)
    action_input_mask = torch.cat([action_prefix_mask, suffix_mask], dim=1)
    action_ar_mask = torch.cat([torch.zeros_like(action_prefix_mask), suffix_ar_mask], dim=1)
    combined[:, prefix_len:, :] = make_attn_mask(action_input_mask, action_ar_mask)[:, prefix_len:, :]
    return combined


def make_joint_positions(prefix_mask, action_prefix_mask, suffix_mask):
    prefix_positions = torch.cumsum(prefix_mask.to(torch.int64), dim=1) - 1
    suffix_positions = (
        action_prefix_mask.sum(dim=1, keepdim=True) + torch.cumsum(suffix_mask.to(torch.int64), dim=1) - 1
    )
    return torch.cat([prefix_positions, suffix_positions], dim=1)


def _location_classes_to_source_xyxy(classes: torch.Tensor) -> torch.Tensor:
    coords = classes.to(torch.float32) / (future_bbox.NUM_LOCATION_BINS - 1)
    y1, x1, y2, x2 = coords.unbind(-1)
    size, width, height = future_bbox.MODEL_IMAGE_SIZE, future_bbox.RAW_IMAGE_WIDTH, future_bbox.RAW_IMAGE_HEIGHT
    ratio = max(width / size, height / size)
    pad_y = (size - int(height / ratio)) // 2
    pad_x = (size - int(width / ratio)) // 2
    xyxy = torch.stack(
        [
            (x1 * size - pad_x) * ratio / width,
            (y1 * size - pad_y) * ratio / height,
            (x2 * size - pad_x) * ratio / width,
            (y2 * size - pad_y) * ratio / height,
        ],
        dim=-1,
    )
    return xyxy.clamp(0.0, 1.0)


def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter = (torch.minimum(a[..., 2:], b[..., 2:]) - torch.maximum(a[..., :2], b[..., :2])).clamp(min=0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp(min=0).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp(min=0).prod(-1)
    return inter / (area_a + area_b - inter).clamp(min=1e-8)


def compute_future_bbox_loss(bbox_logits, target_classes, target_mask):
    """Masked four-location-token cross-entropy and geometry metrics, as the JAX `compute_future_bbox_loss`."""
    bbox_logits = bbox_logits.to(torch.float32)
    target_mask = target_mask.to(torch.bool)
    batch_size = bbox_logits.shape[0]
    token_loss = -torch.gather(F.log_softmax(bbox_logits, dim=-1), -1, target_classes[..., None]).squeeze(-1)

    target_box_mask = target_mask.reshape(batch_size, -1, 4)
    box_valid = target_box_mask.all(dim=-1)
    valid_token_mask = box_valid[..., None].expand_as(target_box_mask).reshape(target_mask.shape)
    sample_valid = box_valid.any(dim=1)
    valid_token_count = valid_token_mask.sum(dim=1)
    raw_sample_loss = (token_loss * valid_token_mask).sum(dim=1) / valid_token_count.clamp(min=1)
    sample_loss = raw_sample_loss * batch_size / sample_valid.sum().clamp(min=1)

    with torch.no_grad():
        predictions = bbox_logits.argmax(dim=-1)
        prediction_classes = predictions.reshape(batch_size, -1, 4)
        iou = _box_iou_xyxy(
            _location_classes_to_source_xyxy(prediction_classes),
            _location_classes_to_source_xyxy(target_classes.reshape(batch_size, -1, 4)),
        )
        invalid = (prediction_classes[..., 0] >= prediction_classes[..., 2]) | (
            prediction_classes[..., 1] >= prediction_classes[..., 3]
        )
        valid_boxes = box_valid.sum().clamp(min=1)
        metrics = {
            "bbox_loss": sample_loss.mean(),
            "bbox_token_accuracy": ((predictions == target_classes) & valid_token_mask).sum()
            / valid_token_mask.sum().clamp(min=1),
            "bbox_iou": (iou * box_valid).sum() / valid_boxes,
            "bbox_invalid_rate": (invalid & box_valid).sum() / valid_boxes,
            "bbox_valid_fraction": box_valid.float().mean(),
        }
    return sample_loss, metrics


def posemb_sincos(pos: torch.Tensor, dim: int, min_period: float, max_period: float) -> torch.Tensor:
    # float32 throughout, as the JAX model computes it (openpi's PyTorch port uses float64, which differs by ~1e-4).
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float32, device=pos.device)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid = pos.to(torch.float32)[:, None] * (1.0 / period * 2 * math.pi)[None, :]
    return torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)


def _turbo_colormap(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    coefficients = _TURBO.to(value.device)
    result = torch.zeros(*value.shape, 3, dtype=torch.float32, device=value.device)
    for coefficient in coefficients.flip(1).T:
        result = result * value[..., None] + coefficient
    return result.clamp(0.0, 1.0)


def _metric_depth_to_siglip_image(depth_m: torch.Tensor, quality: torch.Tensor, patch_size: int = 14):
    """[B, 224, 224] metric Z-depth -> ([B, 224, 224, 3] in [-1, 1], [B, 256] patch-valid mask)."""
    depth_m = depth_m.to(torch.float32)
    quality = quality.to(torch.float32)
    if depth_m.ndim != 3 or depth_m.shape[1:] != (224, 224) or quality.shape != depth_m.shape:
        raise ValueError(f"Expected depth and quality [batch, 224, 224], got {depth_m.shape} and {quality.shape}")
    near, far = _jax_foveavla.METRIC_DEPTH_NEAR_M, _jax_foveavla.METRIC_DEPTH_FAR_M
    valid = (
        torch.isfinite(depth_m)
        & torch.isfinite(quality)
        & (depth_m >= near)
        & (depth_m <= far)
        & (quality >= _jax_foveavla.METRIC_DEPTH_QUALITY_MIN)
    )
    safe = torch.nan_to_num(depth_m, nan=near).clamp(near, far)
    normalized = torch.log(safe / near) / math.log(far / near)
    image = _turbo_colormap(1.0 - normalized) * 2.0 - 1.0
    image = torch.where(valid[..., None], image, torch.full_like(image, -1.0))
    b, h, w = valid.shape
    grid_h, grid_w = h // patch_size, w // patch_size
    patch_fraction = valid.to(torch.float32).reshape(b, grid_h, patch_size, grid_w, patch_size).mean(dim=(2, 4))
    patch_valid = patch_fraction.reshape(b, grid_h * grid_w) >= _jax_foveavla.METRIC_DEPTH_PATCH_MIN_VALID_FRACTION
    return image, patch_valid


def _rgb_to_hsv(rgb: torch.Tensor):
    value, argmax = rgb.max(dim=-1)
    rng = value - rgb.min(dim=-1).values
    second = torch.gather(rgb, -1, ((argmax + 1) % 3)[..., None]).squeeze(-1)
    third = torch.gather(rgb, -1, ((argmax + 2) % 3)[..., None]).squeeze(-1)
    safe_rng = torch.where(rng == 0, torch.ones_like(rng), rng)
    hue = torch.where(rng == 0, torch.zeros_like(rng), (2 * argmax + (second - third) / safe_rng) / 6)
    saturation = torch.where(value == 0, torch.zeros_like(value), rng / torch.where(value == 0, 1.0, value))
    return hue, saturation, value


def _hsv_to_rgb(hue, saturation, value):
    n = torch.tensor([5.0, 3.0, 1.0], device=hue.device)
    k = torch.remainder(n + hue[..., None] * 6, 6)
    return value[..., None] - value[..., None] * saturation[..., None] * torch.clamp(
        torch.minimum(k, 4 - k), 0, 1
    )


def draw_color_jitter(batch: int, device, brightness=0.3, contrast=0.4, hue=0.1, p=0.5) -> dict:
    """One augmax ColorJitter draw per sample. JAX FoveaVLA reuses the same per-sample keys for every camera, so
    one draw is applied to all views of a sample."""

    def uniform(strength):
        return (torch.rand(batch, 1, 1, device=device) * 2 - 1) * strength

    return {
        "brightness": uniform(brightness),
        "contrast": uniform(contrast),
        "hue": uniform(hue),
        "apply": torch.rand(batch, 1, 1, 1, device=device) < p,
    }


def color_jitter(image: torch.Tensor, params: dict) -> torch.Tensor:
    """augmax ColorJitter on [B, H, W, 3] images in [0, 1] (augmax's saturation step is a no-op, so none here)."""
    h, s, v = _rgb_to_hsv(image)
    amount = params["brightness"]
    v = torch.where(amount < 0, v * (1 + amount), v * (1 - amount) + amount)
    slant = torch.tan((params["contrast"] + 1.0) * (math.pi / 4))
    p1 = (slant - slant**2) / (2 * (1 - slant**2))
    p2 = 1 - p1
    v = torch.where(v < p1, v / slant, torch.where(v > p2, v / slant + 1 - 1 / slant, slant * (v - 0.5) + 0.5))
    out = _hsv_to_rgb(h + params["hue"], s, v)
    return torch.where(params["apply"], out, image)


class FoveaVLAPytorch(nn.Module):
    def __init__(self, config: FoveaVLAConfig, precision: str = "bfloat16"):
        super().__init__()
        if config.full_view_patch_rows is not None or config.depth_token_pool != 1:
            raise NotImplementedError("The PyTorch port does not implement the token-efficient options yet")
        self.config = config
        self.image_keys = tuple(config.image_keys)
        self.depth_image_keys = tuple(config.depth_image_keys)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config, action_expert_config, use_adarms=[False, True], precision=precision
        )
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.gradient_checkpointing = False
        torch.set_float32_matmul_precision("high")

    # Gradient checkpointing follows openpi's PI0Pytorch: the joint Gemma forward checkpoints every layer, and the
    # SigLIP passes here checkpoint per encoder layer.
    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

    def _ckpt(self, fn, *args):
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)
        return fn(*args)

    @property
    def _llm(self):
        return self.paligemma_with_expert.paligemma.language_model

    def _llm_dtype(self):
        return self._llm.layers[0].self_attn.q_proj.weight.dtype

    def embed_image(self, image: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """SigLIP + projector on [B, H, W, 3] images in [-1, 1]; with `token_mask` [B, 256], masked patches are
        zeroed and excluded from attention after every block, as the JAX SigLIP does."""
        paligemma = self.paligemma_with_expert.paligemma
        pixels = image.permute(0, 3, 1, 2).to(torch.float32)
        if token_mask is None:
            return self._ckpt(paligemma.model.get_image_features, pixels)
        vision = paligemma.model.vision_tower.vision_model
        hidden = vision.embeddings(pixels)
        keep = token_mask[..., None]
        hidden = torch.where(keep, hidden, torch.zeros_like(hidden))
        if vision.encoder.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden = hidden.to(torch.bfloat16)
        pair = token_mask[:, None, :, None] & token_mask[:, None, None, :]
        attn = torch.where(pair, 0.0, _NEG).to(hidden.dtype)
        for layer in vision.encoder.layers:
            hidden = self._ckpt(lambda h, m, layer=layer: layer(h, attention_mask=m)[0], hidden, attn)
            hidden = torch.where(keep, hidden, torch.zeros_like(hidden))
        hidden = vision.post_layernorm(hidden)
        hidden = torch.where(keep, hidden, torch.zeros_like(hidden))
        return paligemma.model.multi_modal_projector(hidden)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        emb = self._llm.embed_tokens(tokens)
        return emb * math.sqrt(emb.shape[-1])

    def preprocess(self, observation, *, train: bool):
        images = {}
        batch = observation.state.shape[0]
        jitter = draw_color_jitter(batch, observation.state.device) if train else None
        for key in self.image_keys:
            image = observation.images[key]
            if image.shape[1:3] != (224, 224):
                raise ValueError(f"Expected 224x224 inputs for {key}, got {tuple(image.shape)}")
            if train:
                image = color_jitter(image / 2.0 + 0.5, jitter) * 2.0 - 1.0
            images[key] = image
        masks = {
            key: observation.image_masks.get(key, torch.ones(batch, dtype=torch.bool, device=observation.state.device))
            for key in self.image_keys
        }
        return images, masks

    def embed_prefix(self, observation, images, image_masks, *, train: bool):
        tokens, input_mask, ar_mask = [], [], []
        for key in self.image_keys:
            emb = self.embed_image(images[key])
            tokens.append(emb)
            input_mask.append(image_masks[key][:, None].expand(-1, emb.shape[1]))
            ar_mask.append(torch.zeros(emb.shape[:2], dtype=torch.bool, device=emb.device))

        batch = observation.state.shape[0]
        device = observation.state.device
        if train and self.config.depth_view_dropout_prob:
            view_keep = torch.rand(batch, len(self.depth_image_keys), device=device) >= self.config.depth_view_dropout_prob
        else:
            view_keep = torch.ones(batch, len(self.depth_image_keys), dtype=torch.bool, device=device)
        for i, key in enumerate(self.depth_image_keys):
            depth_image, patch_valid = _metric_depth_to_siglip_image(
                observation.depthmaps[key], observation.depthmap_quality[key]
            )
            patch_valid = patch_valid & image_masks[key][:, None] & view_keep[:, i, None]
            tokens.append(self.embed_image(depth_image, token_mask=patch_valid))
            input_mask.append(patch_valid)
            ar_mask.append(torch.zeros_like(patch_valid))

        tokens.append(self._ckpt(self.embed_language_tokens, observation.tokenized_prompt))
        input_mask.append(observation.tokenized_prompt_mask.to(torch.bool))
        ar_mask.append(observation.token_ar_mask.to(torch.bool))
        tokens = [t.to(self._llm_dtype()) for t in tokens]
        return torch.cat(tokens, dim=1), torch.cat(input_mask, dim=1), torch.cat(ar_mask, dim=1)

    def embed_suffix(self, noisy_actions: torch.Tensor, timestep: torch.Tensor):
        action_tokens = self.action_in_proj(noisy_actions.to(self.action_in_proj.weight.dtype))
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = time_emb.to(self.time_mlp_in.weight.dtype)
        adarms_cond = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb))))
        batch, horizon = action_tokens.shape[:2]
        input_mask = torch.ones(batch, horizon, dtype=torch.bool, device=action_tokens.device)
        ar_mask = torch.zeros(batch, horizon, dtype=torch.bool, device=action_tokens.device)
        ar_mask[:, 0] = True
        return action_tokens.to(self._llm_dtype()), input_mask, ar_mask, adarms_cond

    def _location_rows(self) -> torch.Tensor:
        start = future_bbox.LOCATION_TOKEN_START_ID
        return self._llm.embed_tokens.weight[start : start + future_bbox.NUM_LOCATION_BINS]

    def _bbox_loss(self, observation, prefix_out):
        count = 4 * self.config.future_bbox_count
        prompt = observation.tokenized_prompt
        text_len = prompt.shape[1]
        idx = (
            observation.tokenized_prompt_mask.to(torch.int64).sum(dim=1, keepdim=True)
            - count
            + torch.arange(count, device=prompt.device)[None, :]
        )
        target_ids = torch.gather(prompt.to(torch.int64), 1, idx)
        target_mask = torch.gather(observation.token_loss_mask.to(torch.bool), 1, idx)
        hidden_idx = prefix_out.shape[1] - text_len + idx - 1
        hidden = torch.gather(prefix_out, 1, hidden_idx[..., None].expand(-1, -1, prefix_out.shape[-1]))
        logits = hidden.to(torch.float32) @ self._location_rows().to(torch.float32).T
        return compute_future_bbox_loss(logits, target_ids - future_bbox.LOCATION_TOKEN_START_ID, target_mask)

    def forward(self, observation, actions, *, noise=None, time=None, train: bool = True):
        """Returns (per-sample-per-step loss [B, H], metrics), as the JAX `compute_loss_with_metrics`."""
        images, image_masks = self.preprocess(observation, train=train)
        actions = actions.to(torch.float32)
        if noise is None:
            noise = torch.randn_like(actions)
        if time is None:
            beta = torch.distributions.Beta(torch.tensor(1.5, device=actions.device), torch.tensor(1.0, device=actions.device))
            time = beta.sample((actions.shape[0],)) * 0.999 + 0.001
        t = time[:, None, None]
        x_t = t * noise + (1 - t) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(observation, images, image_masks, train=train)
        suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(x_t, time)
        action_prefix_mask = prefix_mask & ~prefix_ar
        attn = make_joint_attn_mask(prefix_mask, prefix_ar, action_prefix_mask, suffix_mask, suffix_ar)
        positions = make_joint_positions(prefix_mask, action_prefix_mask, suffix_mask)
        attn_4d = torch.where(attn[:, None], 0.0, _NEG)

        (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=attn_4d,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix_tokens, suffix_tokens],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.config.action_horizon :].to(self.action_out_proj.weight.dtype))
        action_loss = (v_t.to(torch.float32) - u_t).square().mean(dim=-1)
        bbox_loss, metrics = self._bbox_loss(observation, prefix_out)
        metrics = {"action_loss": action_loss.detach().mean(), **metrics}
        return action_loss + self.config.bbox_loss_weight * bbox_loss[:, None], metrics

    @torch.no_grad()
    def prepare_inference_prefix(self, observation):
        images, image_masks = self.preprocess(observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(observation, images, image_masks, train=False)
        attn = torch.where(make_attn_mask(prefix_mask, prefix_ar)[:, None], 0.0, _NEG)
        positions = torch.cumsum(prefix_mask.to(torch.int64), dim=1) - 1
        self._llm.config._attn_implementation = "eager"  # noqa: SLF001
        (prefix_out, _), cache = self.paligemma_with_expert.forward(
            attention_mask=attn,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix_tokens, None],
            use_cache=True,
        )
        return prefix_out, prefix_mask, prefix_ar, cache

    @torch.no_grad()
    def sample_actions(self, observation, *, noise=None, num_steps: int = 10, prefix=None):
        prefix_out, prefix_mask, prefix_ar, cache = prefix if prefix is not None else self.prepare_inference_prefix(observation)
        batch = prefix_mask.shape[0]
        device = prefix_mask.device
        if noise is None:
            noise = torch.randn(batch, self.config.action_horizon, self.config.action_dim, device=device)
        action_prefix_mask = prefix_mask & ~prefix_ar
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        dt = -1.0 / num_steps
        x_t, time = noise.to(torch.float32), 1.0
        while time >= -dt / 2:
            suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(
                x_t, torch.full((batch,), time, device=device)
            )
            suffix_len = suffix_tokens.shape[1]
            full = torch.cat(
                [action_prefix_mask[:, None, :].expand(-1, suffix_len, -1), make_attn_mask(suffix_mask, suffix_ar)],
                dim=2,
            )
            positions = action_prefix_mask.sum(dim=1, keepdim=True) + torch.cumsum(suffix_mask.to(torch.int64), 1) - 1
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=torch.where(full[:, None], 0.0, _NEG),
                position_ids=positions,
                past_key_values=cache,
                inputs_embeds=[None, suffix_tokens],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.config.action_horizon :].to(self.action_out_proj.weight.dtype))
            x_t = x_t + dt * v_t.to(torch.float32)
            time += dt
        return x_t

    @torch.no_grad()
    def sample_future_bbox(self, observation, *, prefix=None):
        """Greedy decode of the t+8 box(es). Generated tokens are fed back each step with `use_cache=False`, so the
        prefix cache is left unchanged for action sampling."""
        prefix_out, prefix_mask, _, cache = prefix if prefix is not None else self.prepare_inference_prefix(observation)
        batch, prefix_len = prefix_mask.shape
        device = prefix_mask.device
        seq = torch.arange(prefix_len, device=device)
        last = torch.where(prefix_mask, seq[None], -1).max(dim=1).values
        hidden = prefix_out[torch.arange(batch, device=device), last]
        rows = self._location_rows().to(torch.float32)
        count = 4 * self.config.future_bbox_count
        classes, generated = [], []
        n_valid = prefix_mask.sum(dim=1, keepdim=True)
        for k in range(count):
            predicted = (hidden.to(torch.float32) @ rows.T).argmax(dim=-1)
            classes.append(predicted)
            if k == count - 1:
                break
            generated.append(self.embed_language_tokens(predicted[:, None] + future_bbox.LOCATION_TOKEN_START_ID))
            embeds = torch.cat(generated, dim=1).to(self._llm_dtype())
            n = embeds.shape[1]
            causal = torch.ones(n, n, dtype=torch.bool, device=device).tril()
            mask = torch.cat([prefix_mask[:, None, :].expand(-1, n, -1), causal[None].expand(batch, -1, -1)], dim=2)
            (out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=torch.where(mask[:, None], 0.0, _NEG),
                position_ids=n_valid + torch.arange(n, device=device)[None],
                past_key_values=cache,
                inputs_embeds=[embeds, None],
                use_cache=False,
            )
            hidden = out[:, -1]
        classes = torch.stack(classes, dim=-1).reshape(batch, self.config.future_bbox_count, 4)
        box = _location_classes_to_source_xyxy(classes)
        valid = (box[..., 0] < box[..., 2]) & (box[..., 1] < box[..., 3])
        if self.config.future_bbox_count == 1:
            return box[:, 0], valid[:, 0]
        return box, valid
