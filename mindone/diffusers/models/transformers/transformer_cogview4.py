# Copyright 2024 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional, Tuple, Union

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import mint, nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import logging
from ..attention import FeedForward
from ..attention_processor import Attention
from ..embeddings import CogView3CombinedTimestepSizeEmbeddings
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CogView4PatchEmbed(nn.Cell):
    def __init__(
        self,
        in_channels: int = 16,
        hidden_size: int = 2560,
        patch_size: int = 2,
        text_hidden_size: int = 4096,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = mint.nn.Linear(in_channels * patch_size**2, hidden_size)
        self.text_proj = mint.nn.Linear(text_hidden_size, hidden_size)

    def construct(self, hidden_states: ms.Tensor, encoder_hidden_states: ms.Tensor) -> ms.Tensor:
        batch_size, channel, height, width = hidden_states.shape
        post_patch_height = height // self.patch_size
        post_patch_width = width // self.patch_size

        hidden_states = hidden_states.reshape(
            batch_size, channel, post_patch_height, self.patch_size, post_patch_width, self.patch_size
        )
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
        hidden_states = self.proj(hidden_states)
        encoder_hidden_states = self.text_proj(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class CogView4AdaLayerNormZero(nn.Cell):
    def __init__(self, embedding_dim: int, dim: int) -> None:
        super().__init__()

        self.norm = mint.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm_context = mint.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.linear = mint.nn.Linear(embedding_dim, 12 * dim, bias=True)

    def construct(
        self, hidden_states: ms.Tensor, encoder_hidden_states: ms.Tensor, temb: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        norm_hidden_states = self.norm(hidden_states)
        norm_encoder_hidden_states = self.norm_context(encoder_hidden_states)

        emb = self.linear(temb)
        (
            shift_msa,
            c_shift_msa,
            scale_msa,
            c_scale_msa,
            gate_msa,
            c_gate_msa,
            shift_mlp,
            c_shift_mlp,
            scale_mlp,
            c_scale_mlp,
            gate_mlp,
            c_gate_mlp,
        ) = emb.chunk(12, dim=1)

        hidden_states = norm_hidden_states * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_msa.unsqueeze(1)) + c_shift_msa.unsqueeze(1)

        return (
            hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        )


class CogView4AttnProcessor:
    """
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        # move importing from __call__ to __init__ as it is not supported in construct()
        from ..embeddings import apply_rotary_emb

        self.apply_rotary_emb = apply_rotary_emb

    def apply_rotary_emb_for_image_part(
        self,
        hidden_state: ms.Tensor,
        image_rotary_emb: ms.Tensor,
        start_index: int,
        axis: int = 2,
        use_real_unbind_dim: int = -1,
    ) -> ms.Tensor:
        """
        Equivalence of expression(when axis=2):
            `hidden_state[:, :, start_index:, :] = self.apply_rotary_emb(hidden_state[:, :, start_index:, :], image_rotary_emb)`
        Rewrite it since implement above might call ops.ScatterNdUpdate which is super slow!
        """
        hidden_state_text, hidden_state_image = mint.split(
            hidden_state, (start_index, hidden_state.shape[axis] - start_index), dim=axis
        )
        hidden_state_image = self.apply_rotary_emb(
            hidden_state_image, image_rotary_emb, use_real_unbind_dim=use_real_unbind_dim
        )
        hidden_state = mint.cat([hidden_state_text, hidden_state_image], dim=axis)
        return hidden_state

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        image_rotary_emb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        batch_size, text_seq_length, embed_dim = encoder_hidden_states.shape
        batch_size, image_seq_length, embed_dim = hidden_states.shape
        hidden_states = mint.cat([encoder_hidden_states, hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        # key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        # value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        query = query.reshape(query.shape[:2] + (attn.heads, -1) + query.shape[3:]).swapaxes(1, 2)
        key = key.reshape(key.shape[:2] + (attn.heads, -1) + key.shape[3:]).swapaxes(1, 2)
        value = value.reshape(value.shape[:2] + (attn.heads, -1) + value.shape[3:]).swapaxes(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            # query[:, :, text_seq_length:, :] = self.apply_rotary_emb(
            #     query[:, :, text_seq_length:, :], image_rotary_emb, use_real_unbind_dim=-2
            # )
            # key[:, :, text_seq_length:, :] = self.apply_rotary_emb(
            #     key[:, :, text_seq_length:, :], image_rotary_emb, use_real_unbind_dim=-2
            # )
            query = self.apply_rotary_emb_for_image_part(
                query, image_rotary_emb, text_seq_length, use_real_unbind_dim=-2
            )
            key = self.apply_rotary_emb_for_image_part(key, image_rotary_emb, text_seq_length, use_real_unbind_dim=-2)

        # 4. Attention
        if attention_mask is not None:
            text_attention_mask = attention_mask.float()
            actual_text_seq_length = text_attention_mask.shape[1]
            new_attention_mask = mint.zeros((batch_size, text_seq_length + image_seq_length))
            new_attention_mask[:, :actual_text_seq_length] = text_attention_mask
            new_attention_mask = new_attention_mask.unsqueeze(2)
            attention_mask_matrix = new_attention_mask @ new_attention_mask.swapaxes(1, 2)
            attention_mask = (attention_mask_matrix > 0).unsqueeze(1).to(query.dtype)

        hidden_states = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.swapaxes(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # 5. Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.shape[1] - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


class CogView4TransformerBlock(nn.Cell):
    def __init__(
        self, dim: int = 2560, num_attention_heads: int = 64, attention_head_dim: int = 40, time_embed_dim: int = 512
    ) -> None:
        super().__init__()

        # 1. Attention
        self.norm1 = CogView4AdaLayerNormZero(time_embed_dim, dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            out_dim=dim,
            bias=True,
            qk_norm="layer_norm",
            elementwise_affine=False,
            eps=1e-5,
            processor=CogView4AttnProcessor(),
        )

        # 2. Feedforward
        self.norm2 = mint.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm2_context = mint.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def construct(
        self,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor,
        temb: Optional[ms.Tensor] = None,
        image_rotary_emb: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        **kwargs,
    ) -> ms.Tensor:
        # 1. Timestep conditioning
        (
            norm_hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.norm1(hidden_states, encoder_hidden_states, temb)

        # 2. Attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + attn_encoder_hidden_states * c_gate_msa.unsqueeze(1)

        # 3. Feedforward
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states) * (
            1 + c_scale_mlp.unsqueeze(1)
        ) + c_shift_mlp.unsqueeze(1)

        ff_output = self.ff(norm_hidden_states)
        ff_output_context = self.ff(norm_encoder_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + ff_output_context * c_gate_mlp.unsqueeze(1)

        return hidden_states, encoder_hidden_states


class CogView4RotaryPosEmbed(nn.Cell):
    def __init__(self, dim: int, patch_size: int, rope_axes_dim: Tuple[int, int], theta: float = 10000.0) -> None:
        super().__init__()

        self.dim = dim
        self.patch_size = patch_size
        self.rope_axes_dim = rope_axes_dim
        self.theta = theta

    def construct(self, hidden_states: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        batch_size, num_channels, height, width = hidden_states.shape
        height, width = height // self.patch_size, width // self.patch_size

        dim_h, dim_w = self.dim // 2, self.dim // 2
        h_inv_freq = 1.0 / (self.theta ** (mint.arange(0, dim_h, 2, dtype=ms.float32)[: (dim_h // 2)].float() / dim_h))
        w_inv_freq = 1.0 / (self.theta ** (mint.arange(0, dim_w, 2, dtype=ms.float32)[: (dim_w // 2)].float() / dim_w))
        h_seq = mint.arange(self.rope_axes_dim[0])
        w_seq = mint.arange(self.rope_axes_dim[1])
        freqs_h = mint.outer(h_seq, h_inv_freq)
        freqs_w = mint.outer(w_seq, w_inv_freq)

        h_idx = mint.arange(height)
        w_idx = mint.arange(width)
        inner_h_idx = h_idx * self.rope_axes_dim[0] // height
        inner_w_idx = w_idx * self.rope_axes_dim[1] // width

        freqs_h = freqs_h[inner_h_idx]
        freqs_w = freqs_w[inner_w_idx]

        # Create position matrices for height and width
        # [height, 1, dim//4] and [1, width, dim//4]
        freqs_h = freqs_h.unsqueeze(1)
        freqs_w = freqs_w.unsqueeze(0)
        # Broadcast freqs_h and freqs_w to [height, width, dim//4]
        freqs_h = freqs_h.broadcast_to((height, width, -1))
        freqs_w = freqs_w.broadcast_to((height, width, -1))

        # Concatenate along last dimension to get [height, width, dim//2]
        freqs = mint.cat([freqs_h, freqs_w], dim=-1)
        freqs = mint.cat([freqs, freqs], dim=-1)  # [height, width, dim]
        freqs = freqs.reshape(height * width, -1)
        return (freqs.cos(), freqs.sin())


class CogView4Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    r"""
    Args:
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        attention_head_dim (`int`, defaults to `40`):
            The number of channels in each head.
        num_attention_heads (`int`, defaults to `64`):
            The number of heads to use for multi-head attention.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        condition_dim (`int`, defaults to `256`):
            The embedding dimension of the input SDXL-style resolution conditions (original_size, target_size,
            crop_coords).
        pos_embed_max_size (`int`, defaults to `128`):
            The maximum resolution of the positional embeddings, from which slices of shape `H x W` are taken and added
            to input patched latents, where `H` and `W` are the latent height and width respectively. A value of 128
            means that the maximum supported height and width for image generation is `128 * vae_scale_factor *
            patch_size => 128 * 8 * 2 => 2048`.
        sample_size (`int`, defaults to `128`):
            The base resolution of input latents. If height/width is not provided during generation, this value is used
            to determine the resolution as `sample_size * vae_scale_factor => 128 * 8 => 1024`
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["CogView4TransformerBlock", "CogView4PatchEmbed", "CogView4PatchEmbed"]
    _skip_layerwise_casting_patterns = ["patch_embed", "norm", "proj_out"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: int = 16,
        num_layers: int = 30,
        attention_head_dim: int = 40,
        num_attention_heads: int = 64,
        text_embed_dim: int = 4096,
        time_embed_dim: int = 512,
        condition_dim: int = 256,
        pos_embed_max_size: int = 128,
        sample_size: int = 128,
        rope_axes_dim: Tuple[int, int] = (256, 256),
    ):
        super().__init__()

        # CogView4 uses 3 additional SDXL-like conditions - original_size, target_size, crop_coords
        # Each of these are sincos embeddings of shape 2 * condition_dim
        pooled_projection_dim = 3 * 2 * condition_dim
        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels

        # 1. RoPE
        self.rope = CogView4RotaryPosEmbed(attention_head_dim, patch_size, rope_axes_dim, theta=10000.0)

        # 2. Patch & Text-timestep embedding
        self.patch_embed = CogView4PatchEmbed(in_channels, inner_dim, patch_size, text_embed_dim)

        self.time_condition_embed = CogView3CombinedTimestepSizeEmbeddings(
            embedding_dim=time_embed_dim,
            condition_dim=condition_dim,
            pooled_projection_dim=pooled_projection_dim,
            timesteps_dim=inner_dim,
        )

        # 3. Transformer blocks
        self.transformer_blocks = nn.CellList(
            [
                CogView4TransformerBlock(inner_dim, num_attention_heads, attention_head_dim, time_embed_dim)
                for _ in range(num_layers)
            ]
        )

        # 4. Output projection
        self.norm_out = AdaLayerNormContinuous(inner_dim, time_embed_dim, elementwise_affine=False)
        self.proj_out = mint.nn.Linear(inner_dim, patch_size * patch_size * out_channels, bias=True)

        self.gradient_checkpointing = False

        self.patch_size = self.config.patch_size

    def construct(
        self,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor,
        timestep: ms.Tensor,
        original_size: ms.Tensor,
        target_size: ms.Tensor,
        crop_coords: ms.Tensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        attention_mask: Optional[ms.Tensor] = None,
        **kwargs,
    ) -> Union[ms.Tensor, Transformer2DModelOutput]:
        batch_size, num_channels, height, width = hidden_states.shape

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Patch & Timestep embeddings
        p = self.patch_size
        post_patch_height = height // p
        post_patch_width = width // p

        hidden_states, encoder_hidden_states = self.patch_embed(hidden_states, encoder_hidden_states)

        temb = self.time_condition_embed(timestep, original_size, target_size, crop_coords, hidden_states.dtype)
        temb = F.silu(temb)

        # 3. Transformer blocks
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, image_rotary_emb, attention_mask, **kwargs
            )

        # 4. Output norm & projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(batch_size, post_patch_height, post_patch_width, -1, p, p)
        output = hidden_states.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
