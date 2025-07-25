# Adapted from https://github.com/Tencent-Hunyuan/HunyuanVideo to work with MindSpore.
import logging
import os
from typing import List, Optional

import numpy as np

import mindspore as ms
from mindspore import Tensor, mint, nn, ops, tensor
from mindspore.communication import get_group_size

from mindone.diffusers.configuration_utils import ConfigMixin, register_to_config
from mindone.diffusers.models import ModelMixin

from ..acceleration import (
    GatherFowardSplitBackward,
    SplitFowardGatherBackward,
    get_sequence_parallel_group,
    init_alltoall,
)
from .activation_layers import get_activation_layer
from .attention import FlashAttentionVarLen, VanillaAttention  # , parallel_attention, get_cu_seqlens
from .embed_layers import PatchEmbed, TextProjection, TimestepEmbedder
from .mlp_layers import MLP, FinalLayer, MLPEmbedder
from .modulate_layers import ModulateDiT, apply_gate, modulate
from .norm_layers import LayerNorm, get_norm_layer
from .posemb_layers import RoPE
from .token_refiner import SingleTokenRefiner, rearrange_qkv

logger = logging.getLogger(__name__)


class MMDoubleStreamBlock(nn.Cell):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        attn_mode: str = "flash",
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        self.head_dim = head_dim
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_attn_qkv = mint.nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_proj = mint.nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        self.img_norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.txt_attn_qkv = mint.nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_proj = mint.nn.Linear(
            hidden_size,
            hidden_size,
            bias=qkv_bias,
        )

        self.txt_norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        if (sp_group := get_sequence_parallel_group()) is not None:
            self.sp_group_size = get_group_size(sp_group)
            self.alltoall = init_alltoall(
                2, 1, group=sp_group, split_count=self.sp_group_size
            )  # BSND, split at 2 dim and concat at 1 dim
            self.alltoall_out = init_alltoall(
                1, 2, group=sp_group, split_count=self.sp_group_size
            )  # BSND, split at 1 dim and concat at 2 dim
            if heads_num % self.sp_group_size != 0:
                raise ValueError(f"heads_num {heads_num} must be divisible by sp_group_size {self.sp_group_size}")
            heads_num = heads_num // self.sp_group_size
        else:
            self.sp_group_size = None
            self.alltoall = nn.Identity()
            self.alltoall_out = nn.Identity()

        if attn_mode == "vanilla":
            self.compute_attention = VanillaAttention(head_dim)
        elif attn_mode == "flash":
            self.compute_attention = FlashAttentionVarLen(heads_num, head_dim)
        else:
            raise NotImplementedError

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def construct(
        self,
        img: ms.Tensor,
        txt: ms.Tensor,
        vec: ms.Tensor,
        actual_seq_qlen: ms.Tensor = None,
        actual_seq_kvlen: ms.Tensor = None,
        freqs_cos: ms.Tensor = None,
        freqs_sin: ms.Tensor = None,
        attn_mask: ms.Tensor = None,
    ):
        """
        img: (B S_v HD), HD - hidden_size = (num_heads * head_dim)
        txt: (B S_t HD)
        vec: (B HD), projected representation of timestep and global text embed (from CLIP)
        actual_seq_qlen: []
        attn_mask: (B 1 S_v+S_t S_v+S_t)
        """
        # DOING: img -> M
        # txt = txt.to(self.param_dtype)
        # vec = vec.to(self.param_dtype)

        # AMP: in xx_mode, silu (input cast to bf16) -> linear (bf16), so output bf16
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(
            6, axis=-1
        )  # shift, scale, gate are all zeros initially
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(
            vec
        ).chunk(6, axis=-1)

        # Prepare image for attention.
        # AMP: img bf16, norm fp32, out bf16
        img_modulated = self.img_norm1(img)

        # AMP: matmul and add/sum ops, should be bf16
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)

        img_qkv = self.img_attn_qkv(img_modulated)
        # "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        img_q, img_k, img_v = rearrange_qkv(img_qkv, self.heads_num)

        # Apply QK-Norm if needed
        # AMP: img_q bf16, rms norm (fp32) output cast to bf16, out bf16
        img_q = self.img_attn_q_norm(img_q)  # .to(img_v)
        img_k = self.img_attn_k_norm(img_k)  # .to(img_v)

        # Apply RoPE if needed.
        if freqs_cos is not None:
            # AMP: img_q, img_k cast to fp32 inside, cast back in output, out bf16
            img_qq, img_kk = RoPE()(img_q, img_k, freqs_cos, freqs_sin, head_first=False)

            img_q, img_k = img_qq.to(img_q.dtype), img_kk.to(img_k.dtype)

        # Prepare txt for attention.
        # AMP: txt bf16, norm fp32, out bf16
        txt_modulated = self.txt_norm1(txt)

        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
        txt_qkv = self.txt_attn_qkv(txt_modulated)

        # "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        txt_q, txt_k, txt_v = rearrange_qkv(txt_qkv, self.heads_num)

        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q)  # .to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k)  # .to(txt_v)

        # sequence_parallel
        img_q = self.alltoall(img_q)  # B S_v/sp H D -> B S_v H/sp, D if sp is enabled
        img_k = self.alltoall(img_k)
        img_v = self.alltoall(img_v)
        txt_q = self.alltoall(txt_q)
        txt_k = self.alltoall(txt_k)
        txt_v = self.alltoall(txt_v)
        img_seq_len = img_q.shape[1]

        # Run actual attention.
        # input hidden states (B, S_v+S_t, H, D)
        q = ops.concat((img_q, txt_q), axis=1)
        k = ops.concat((img_k, txt_k), axis=1)
        v = ops.concat((img_v, txt_v), axis=1)
        # assert (
        #    cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
        # ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"

        # attention computation start

        attn = self.compute_attention(
            q,
            k,
            v,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
        )

        # attention computation end

        # output hidden states (B, S_v+S_t, H, D)
        img_attn, txt_attn = attn[:, :img_seq_len], attn[:, img_seq_len:]
        img_attn, txt_attn = self.alltoall_out(img_attn), self.alltoall_out(
            txt_attn
        )  # B S_v H/sp, D -> B S_v/sp H D if sp is enabled

        # Calculate the img bloks.
        # residual connection with gate. img = img + img_attn_proj * gate, for simplicity
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)),
            gate=img_mod2_gate,
        )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )

        return img, txt


class MMSingleStreamBlock(nn.Cell):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        attn_mode: str = "flash",
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim**-0.5

        # qkv and mlp_in
        self.linear1 = mint.nn.Linear(
            hidden_size,
            hidden_size * 3 + mlp_hidden_dim,
        )
        # proj and mlp_out
        self.linear2 = mint.nn.Linear(
            hidden_size + mlp_hidden_dim,
            hidden_size,
        )

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.pre_norm = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

        if (sp_group := get_sequence_parallel_group()) is not None:
            self.sp_group_size = get_group_size(sp_group)
            self.alltoall = init_alltoall(
                2, 1, group=sp_group, split_count=self.sp_group_size
            )  # BSND, split at 2 dim and concat at 1 dim
            self.alltoall_out = init_alltoall(
                1, 2, group=sp_group, split_count=self.sp_group_size
            )  # BSND, split at 1 dim and concat at 2 dim
            if heads_num % self.sp_group_size != 0:
                raise ValueError(f"heads_num {heads_num} must be divisible by sp_group_size {self.sp_group_size}")
            heads_num = heads_num // self.sp_group_size
        else:
            self.sp_group_size = None
            self.alltoall = nn.Identity()
            self.alltoall_out = nn.Identity()

        if attn_mode == "vanilla":
            self.compute_attention = VanillaAttention(head_dim)
        elif attn_mode == "flash":
            self.compute_attention = FlashAttentionVarLen(heads_num, head_dim)
        else:
            raise NotImplementedError

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def construct(
        self,
        x: ms.Tensor,
        vec: ms.Tensor,
        txt_len: int,
        actual_seq_qlen: ms.Tensor = None,
        actual_seq_kvlen: ms.Tensor = None,
        freqs_cos: ms.Tensor = None,
        freqs_sin: ms.Tensor = None,
        attn_mask: ms.Tensor = None,
    ) -> ms.Tensor:
        """
        attn_mask: (B 1 S_v+S_t S_v+S_t)
        """
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, axis=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = ops.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], axis=-1)

        # q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        q, k, v = rearrange_qkv(qkv, heads_num=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q)  # .to(v)
        k = self.k_norm(k)  # .to(v)
        if self.sp_group_size is not None:
            txt_len = txt_len // self.sp_group_size

        img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
        img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
        img_v, txt_v = v[:, :-txt_len, :, :], v[:, -txt_len:, :, :]

        # Apply RoPE if needed.
        if freqs_cos is not None:
            img_qq, img_kk = RoPE()(img_q, img_k, freqs_cos, freqs_sin, head_first=False)
            # assert (
            #    img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            # ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq.to(img_q.dtype), img_kk.to(img_k.dtype)

        # Compute attention.
        # sequence_parallel
        img_q = self.alltoall(img_q)  # B S_v/sp H D -> B S_v H/sp, D if sp is enabled
        img_k = self.alltoall(img_k)
        img_v = self.alltoall(img_v)
        txt_q = self.alltoall(txt_q)
        txt_k = self.alltoall(txt_k)
        txt_v = self.alltoall(txt_v)

        q = ops.concat((img_q, txt_q), axis=1)
        k = ops.concat((img_k, txt_k), axis=1)
        v = ops.concat((img_v, txt_v), axis=1)

        attn = self.compute_attention(
            q,
            k,
            v,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
        )
        if self.sp_group_size is not None:
            txt_len = txt_len * self.sp_group_size

        img_attn, txt_attn = attn[:, :-txt_len], attn[:, -txt_len:]
        img_attn, txt_attn = self.alltoall_out(img_attn), self.alltoall_out(txt_attn)
        # attention computation end
        attn = ops.concat((img_attn, txt_attn), axis=1)
        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(ops.concat((attn, self.mlp_act(mlp)), axis=2))
        return x + apply_gate(output, gate=mod_gate)


class HYVideoDiffusionTransformer(ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206

    Parameters
    ----------
    text_state_dim: int
        The text embedding dim of text encoder 1
    text_state_dim_2: int
        The text embedding dim of text encoder2
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    dtype: ms.dtype
        The dtype of the model, i.e. model parameter dtype
    use_recompute: bool, default=False
        Whether to use recompute.
    num_no_recompute: int or tuple(list) of int, default=0
        The number of blocks to not use recompute.
    """

    @register_to_config
    def __init__(
        self,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # For modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        use_conv2d_patchify: bool = False,
        attn_mode: str = "flash",
        dtype=None,
        use_recompute=False,
        num_no_recompute: int = 0,
        # TeaCache
        enable_teacache: bool = False,
        teacache_thresh: float = 0.1,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.use_conv2d_patchify = use_conv2d_patchify
        # self.dtype = dtype
        self.use_recompute = use_recompute
        self.num_no_recompute = num_no_recompute
        print("attn_mode: ", attn_mode)

        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2

        self.param_dtype = dtype

        if hidden_size % heads_num != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}")
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(f"Got {rope_dim_list} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # image projection
        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size, use_conv2d=use_conv2d_patchify, **factory_kwargs
        )

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                self.text_states_dim, hidden_size, heads_num, depth=2, attn_mode=attn_mode, **factory_kwargs
            )
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        # time modulation
        self.time_in = TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)

        # text modulation
        self.vector_in = MLPEmbedder(self.text_states_dim_2, self.hidden_size, **factory_kwargs)

        # guidance modulation
        self.guidance_in = (
            TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
            if guidance_embed
            else None
        )

        # double blocks
        self.double_blocks = nn.CellList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    attn_mode=attn_mode,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        # single blocks
        self.single_blocks = nn.CellList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    attn_mode=attn_mode,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

        if self.use_recompute:
            num_no_recompute = self.num_no_recompute
            if isinstance(num_no_recompute, int):
                num_no_recompute = (num_no_recompute, num_no_recompute)
            elif isinstance(num_no_recompute, (list, tuple)):
                assert (
                    len(num_no_recompute) == 2
                    and isinstance(num_no_recompute[0], int)
                    and isinstance(num_no_recompute[1], int)
                ), "Expect to have num_no_recompute as a list or tuple of two integers."

            num_blocks = len(self.single_blocks)
            assert (
                num_no_recompute[0] <= num_no_recompute[1] <= num_blocks
            ), f"num_no_recompute should be in [0, {num_blocks}], but got {num_no_recompute}"
            logger.info(f"Excluding {num_no_recompute[0]} single_blocks from the recomputation list.")
            for bidx, block in enumerate(self.single_blocks):
                if bidx < num_blocks - num_no_recompute[0]:
                    self.recompute(block)

            num_blocks = len(self.double_blocks)
            assert (
                num_no_recompute[1] <= num_blocks
            ), f"num_no_recompute should be in [0, {num_blocks}], but got {num_no_recompute}"
            logger.info(f"Excluding {num_no_recompute[1]} double_blocks from the recomputation list.")
            for bidx, block in enumerate(self.double_blocks):
                if bidx < num_blocks - num_no_recompute[1]:
                    self.recompute(block)

        # init sequence parallel
        if (sp_group := get_sequence_parallel_group()) is not None:
            logger.info(f"Initialize HyVideo Transformer model with sequence parallel group `{sp_group}`.")
            self.split_forward_gather_backward = SplitFowardGatherBackward(dim=1, grad_scale="down", group=sp_group)
            self.gather_forward_split_backward = GatherFowardSplitBackward(dim=1, grad_scale="up", group=sp_group)
        else:
            self.split_forward_gather_backward = nn.Identity()
            self.gather_forward_split_backward = nn.Identity()

        # TeaCache
        self._teacache = enable_teacache
        if self._teacache:
            self._rel_l1_thresh = teacache_thresh
            self._coef = [7.33226126e02, -4.01131952e02, 6.75869174e01, -3.14987800e00, 9.61237896e-02]
            # actual values depend on the execution mode and are initialized in `init_teacache`
            self._accum_rel_l1_distance = self._prev_mod_input = self._prev_residual = None

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute(parallel_optimizer_comm_recompute=True)
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        elif ms.get_context("mode") == ms.GRAPH_MODE:
            b.add_flags(output_no_recompute=True)

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def init_teacache(self, shape: tuple[int, int, int, int, int]):
        """
        (Re)initializes the teacache for caching intermediate computation results.
        Specifically, wraps variables with `Parameter` objects of fixed shape and dtype, as required by MindSpore's Graph mode.

        Args:
            shape: shape of the input latent tensor, in format [B C T H W].
        """
        if self._teacache:
            if ms.get_context("mode") == ms.GRAPH_MODE:
                seq_len = shape[2] * (shape[3] // 2) * (shape[4] // 2)
                self._accum_rel_l1_distance = ms.Parameter(
                    tensor(0, dtype=self.param_dtype), name="accum_rel_l1_distance"
                )
                self._prev_mod_input = ms.Parameter(
                    tensor(np.zeros((shape[0], seq_len, self.hidden_size)), dtype=self.param_dtype),
                    name="prev_mod_input",
                )
                if (sp_group := get_sequence_parallel_group()) is not None:  # Sequence Parallel case
                    seq_len //= get_group_size(sp_group)
                self._prev_residual = ms.Parameter(
                    tensor(np.zeros((shape[0], seq_len, self.hidden_size)), dtype=self.param_dtype),
                    name="prev_residual",
                )
            else:
                self._accum_rel_l1_distance = tensor(0, dtype=self.param_dtype)
                self._prev_mod_input = tensor(0, dtype=self.param_dtype)

    def _calc_teacache(self, img: Tensor, vec: Tensor) -> bool:
        img_mod1_shift, img_mod1_scale, *_ = self.double_blocks[0].img_mod(vec).chunk(6, dim=-1)
        normed_inp = self.double_blocks[0].img_norm1(img)
        modulated_inp = modulate(normed_inp, shift=img_mod1_shift, scale=img_mod1_scale)
        modulated_inp = self.gather_forward_split_backward(modulated_inp)  # sequence parallel

        x = mint.mean(mint.abs(modulated_inp - self._prev_mod_input)) / mint.mean(mint.abs(self._prev_mod_input))
        self._accum_rel_l1_distance += (
            self._coef[0] * mint.pow(x, 4)
            + self._coef[1] * mint.pow(x, 3)
            + self._coef[2] * mint.pow(x, 2)
            + self._coef[3] * x
            + self._coef[4]
        )
        # the first step will naturally fail as `self._rescale_func` will produce `NaN`
        if self._accum_rel_l1_distance < self._rel_l1_thresh:
            should_calc = False
        else:
            should_calc = True
            self._accum_rel_l1_distance = tensor(0, dtype=self.param_dtype)

        self._prev_mod_input = modulated_inp
        return should_calc

    def construct(
        self,
        x: ms.Tensor,
        t: ms.Tensor,  # Should be in range(0, 1000).
        text_states: ms.Tensor = None,
        text_mask: ms.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[ms.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[ms.Tensor] = None,
        freqs_sin: Optional[ms.Tensor] = None,
        guidance: ms.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        # actual_seq_len = None,
    ) -> ms.Tensor:
        """
        x: (B C T H W), video latent. dtype same as vae-precision, which is fp16 by default
        t: (B,), float32
        text_states: (B S_t D_t); S_t - seq len of padded text tokens, D_t: text feature dim, from LM text encoder,
            default: S_t=256, D_t = 4096; dtype same as text-encoder-precision, which is fp16 by default.
        text_mask: (B S_t), 1 - retain, 0 - drop;
        text_states_2: (B D_t2), from CLIP text encoder, global text feature (fuse 77 tokens), D_t2=768
        freqs_cos: (B S attn_head_dim) or (S attn_head_dim), S - seq len of the patchified video latent (T * H //2 * W//2)
        freqs_sin: (B S attn_head_dim) or (S attn_head_dim)
        guidance: (B,)
        """
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        # print(tt, th, tw)
        # Prepare modulation vectors.
        # AMP: t (fp16) -> sinusoidal (fp32) -> mlp (bf16), out bf16
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2.to(self.param_dtype))

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            # AMP: sinusoidal (fp32) -> mlp (bf16), out bf16
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        # AMP: img (fp16) -> conv bf16, out bf16
        img = self.img_in(img.to(self.param_dtype))
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            # TODO: remove cast after debug
            # txt = txt.to(self.param_dtype)
            # AMP: txt -> mask sum (fp32) -> c; txt (fp16/fp32) -> linear (bf16); out bf16
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]
        bs = img.shape[0]

        # TODO: for stable training in graph mode, better prepare actual_seq_qlen in data prepartion
        # import pdb; pdb.set_trace()
        max_seq_len = img_seq_len + txt_seq_len
        valid_text_len = text_mask.sum(axis=1)
        actual_seq_len = ops.zeros(bs * 2, dtype=ms.int32)
        for i in range(bs):
            valid_seq_len = valid_text_len[i] + img_seq_len
            actual_seq_len[2 * i] = i * max_seq_len + valid_seq_len
            actual_seq_len[2 * i + 1] = (i + 1) * max_seq_len

        # sequence parallel start
        img = self.split_forward_gather_backward(img)
        if freqs_cos is not None and freqs_sin is not None:
            if freqs_cos.ndim == 2:
                # (S, attn_head_dim)
                freqs_cos = self.split_forward_gather_backward(freqs_cos.unsqueeze(0))[0]
                freqs_sin = self.split_forward_gather_backward(freqs_sin.unsqueeze(0))[0]
            elif freqs_cos.ndim == 3:
                # (B, S, attn_head_dim)
                freqs_cos = self.split_forward_gather_backward(freqs_cos)[0]
                freqs_sin = self.split_forward_gather_backward(freqs_sin)[0]
            else:
                raise ValueError(
                    f"Expect that the n dimensions of freqs_cos(freqs_sin) is 2 or 3, but got {freqs_cos.ndim}"
                )

        txt = self.split_forward_gather_backward(txt)

        # TeaCache
        if self._teacache:
            should_calc = self._calc_teacache(img, vec)

        # --------------------- Pass through DiT blocks ------------------------
        if self._teacache and not should_calc:
            img += self._prev_residual
        else:
            if self._teacache:
                ori_img = img.clone()
            for _, block in enumerate(self.double_blocks):
                # AMP: img bf16, txt bf16, vec bf16, freqs fp32
                img, txt = block(
                    img,
                    txt,
                    vec,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    actual_seq_qlen=actual_seq_len,
                    actual_seq_kvlen=actual_seq_len,
                    # attn_mask=mask,
                )

            # Merge txt and img to pass through single stream blocks.
            x = ops.concat((img, txt), axis=1)
            img_seq_len = img.shape[1]
            if len(self.single_blocks) > 0:
                for _, block in enumerate(self.single_blocks):
                    x = block(
                        x,
                        vec,
                        txt_seq_len,
                        freqs_cos=freqs_cos,
                        freqs_sin=freqs_sin,
                        actual_seq_qlen=actual_seq_len,
                        actual_seq_kvlen=actual_seq_len,
                        # attn_mask=mask,
                    )
            # sequence parallel end

            img = x[:, :img_seq_len, ...]
            if self._teacache:
                self._prev_residual = img - ori_img
        img = self.gather_forward_split_backward(img)
        # print(img.shape)

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        img = self.unpatchify(img, tt, th, tw)

        return img

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]
        x = x.reshape((x.shape[0], t, h, w, c, pt, ph, pw))

        # x = torch.einsum("nthwcopq->nctohpwq", x)
        x = ops.transpose(x, (0, 4, 1, 5, 2, 6, 3, 7))

        imgs = x.reshape((x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def load_from_checkpoint(self, ckpt_path):
        """
        model param dtype
        """
        if ckpt_path.endswith(".pt"):
            import torch

            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            load_key = "module"
            sd = state_dict[load_key]
            # NOTE: self.dtype is get from parameter.dtype in real-time
            param_dtype = ms.float32 if self.dtype is None else self.dtype
            print("D--: get param dtype: ", param_dtype)
            parameter_dict = dict()

            for pname in sd:
                # np doesn't support bf16
                # print(pname, sd[pname].shape, sd[pname].dtype)
                np_val = sd[pname].cpu().detach().float().numpy()
                parameter_dict[pname] = ms.Parameter(ms.Tensor(np_val, dtype=param_dtype))

            # reshape conv3d weight to conv2d if use conv2d in PatchEmbed
            if self.use_conv2d_patchify:
                key_3d = "img_in.proj.weight"
                assert len(sd[key_3d].shape) == 5 and sd[key_3d].shape[-3] == 1  # c_out, c_in, 1, 2, 2
                conv3d_weight = parameter_dict.pop(key_3d)
                parameter_dict[key_3d] = ms.Parameter(conv3d_weight.value().squeeze(axis=-3), name=key_3d)

            param_not_load, ckpt_not_load = ms.load_param_into_net(self, parameter_dict, strict_load=True)
            logger.info(
                "Net params not load: {}, Total net params not loaded: {}".format(param_not_load, len(param_not_load))
            )
            logger.info(
                "Ckpt params not load: {}, Total ckpt params not loaded: {}".format(ckpt_not_load, len(ckpt_not_load))
            )

        elif ckpt_path.endswith(".ckpt"):
            parameter_dict = ms.load_checkpoint(ckpt_path)
            parameter_dict = dict(
                [k.replace("network.model.model.", "") if k.startswith("network.model.model.") else k, v]
                for k, v in parameter_dict.items()
            )
            parameter_dict = dict(
                [k.replace("network.model.", "") if k.startswith("network.model.") else k, v]
                for k, v in parameter_dict.items()
            )
            parameter_dict = dict(
                [k.replace("_backbone.", "") if "_backbone." in k else k, v] for k, v in parameter_dict.items()
            )
            if self.use_conv2d_patchify and "img_in.proj.weight" in parameter_dict:
                key_3d = "img_in.proj.weight"
                if (
                    len(parameter_dict[key_3d].shape) == 5 and parameter_dict[key_3d].shape[-3] == 1
                ):  # c_out, c_in, 1, 2, 2
                    conv3d_weight = parameter_dict.pop(key_3d)
                    parameter_dict[key_3d] = ms.Parameter(conv3d_weight.value().squeeze(axis=-3), name=key_3d)

            param_not_load, ckpt_not_load = ms.load_param_into_net(self, parameter_dict, strict_load=True)
            logger.info(
                "Net params not load: {}, Total net params not loaded: {}".format(param_not_load, len(param_not_load))
            )
            logger.info(
                "Ckpt params not load: {}, Total ckpt params not loaded: {}".format(ckpt_not_load, len(ckpt_not_load))
            )
        else:
            _, file_extension = os.path.splitext(ckpt_path)
            logger.info(
                f"Only support .pt or .ckpt file, but got {file_extension} file. The checkpoint loading will be skipped!!!"
            )

    def params_count(self):
        counts = {
            "double": sum(
                [
                    sum(p.numel() for p in block.img_attn_qkv.parameters())
                    + sum(p.numel() for p in block.img_attn_proj.parameters())
                    + sum(p.numel() for p in block.img_mlp.parameters())
                    + sum(p.numel() for p in block.txt_attn_qkv.parameters())
                    + sum(p.numel() for p in block.txt_attn_proj.parameters())
                    + sum(p.numel() for p in block.txt_mlp.parameters())
                    for block in self.double_blocks
                ]
            ),
            "single": sum(
                [
                    sum(p.numel() for p in block.linear1.parameters())
                    + sum(p.numel() for p in block.linear2.parameters())
                    for block in self.single_blocks
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]

        return counts


#################################################################################
#                             HunyuanVideo Configs                              #
#################################################################################

HUNYUAN_VIDEO_CONFIG = {
    "HYVideo-T/2": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
    },
    "HYVideo-T/2-cfgdistill": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
    "HYVideo-T/2-depth1": {
        "mm_double_blocks_depth": 1,
        "mm_single_blocks_depth": 1,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
}
