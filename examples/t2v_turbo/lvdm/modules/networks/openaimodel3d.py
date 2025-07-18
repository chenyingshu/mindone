# This code is adapted from https://github.com/Ji4chenLi/t2v-turbo
# with modifications to run on MindSpore.


from abc import abstractmethod

from lvdm.basics import (
    avg_pool_nd,
    conv_nd,
    linear,
    normalization,
    rearrange_in_gn5d_bs,
    rearrange_out_gn5d,
    zero_module,
)
from lvdm.common import GroupNormExtend
from lvdm.models.utils_diffusion import timestep_embedding
from lvdm.modules.attention import SpatialTransformer, TemporalTransformer

import mindspore as ms
from mindspore import mint, nn, ops, recompute
from mindspore.common.initializer import Zero, initializer


class TimestepBlock(nn.Cell):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def construct(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.SequentialCell, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def construct(self, x, emb, context=None, batch_size=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, batch_size)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, TemporalTransformer):
                # x = rearrange(x, "(b f) c h w -> b c f h w", b=batch_size)
                x = rearrange_in_gn5d_bs(x, batch_size)
                x = layer(x, context)
                # x = rearrange(x, "b c f h w -> (b f) c h w")
                x = rearrange_out_gn5d(x)
            else:
                x = layer(
                    x,
                )
        return x


class TimestepEmbedSequentialRecompute(nn.SequentialCell, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def construct(self, x, emb, context=None, batch_size=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = recompute(layer, x, emb, batch_size)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, TemporalTransformer):
                # x = rearrange(x, "(b f) c h w -> b c f h w", b=batch_size)
                x = rearrange_in_gn5d_bs(x, batch_size)
                x = layer(x, context)
                # x = rearrange(x, "b c f h w -> (b f) c h w")
                x = rearrange_out_gn5d(x)
            else:
                x = layer(x)
        return x


class Downsample(nn.Cell):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, dtype=ms.float32):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
                pad_mode="pad" if padding > 0 else "same",
                has_bias=True,
                dtype=dtype,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def construct(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Cell):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, dtype=ms.float32):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            pad_mode = "pad" if padding > 0 else "same"
            self.conv = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                padding=padding,
                pad_mode=pad_mode,
                has_bias=True,
                dtype=dtype,
            )

    def construct(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = ops.ResizeNearestNeighbor((x.shape[2], x.shape[3] * 2, x.shape[4] * 2))(x)
        elif self.dims == 2:
            x = ops.ResizeNearestNeighbor((x.shape[2] * 2, x.shape[3] * 2))(x)
        else:
            x = ops.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        use_conv=False,
        up=False,
        down=False,
        use_temporal_conv=False,
        tempspatial_aware=False,
        dtype=ms.float32,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_temporal_conv = use_temporal_conv

        self.in_layers = nn.SequentialCell(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1, pad_mode="pad", has_bias=True, dtype=dtype),
        ).to_float(dtype)

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, dtype=dtype)
            self.x_upd = Upsample(channels, False, dims, dtype=dtype)
        elif down:
            self.h_upd = Downsample(channels, False, dims, dtype=dtype)
            self.x_upd = Downsample(channels, False, dims, dtype=dtype)
        else:
            self.h_upd = nn.Identity()
            self.x_upd = nn.Identity()

        self.emb_layers = nn.SequentialCell(
            nn.SiLU(),
            nn.Dense(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        ).to_float(dtype)
        self.out_layers = nn.SequentialCell(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, pad_mode="pad", has_bias=True)),
        ).to_float(dtype)

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1, pad_mode="pad", has_bias=True, dtype=dtype
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1, has_bias=True, dtype=dtype)

        if self.use_temporal_conv:
            self.temopral_conv = TemporalConvBlock(
                self.out_channels, self.out_channels, dropout=0.1, spatial_aware=tempspatial_aware, dtype=dtype
            )

    def construct(self, x, emb, batch_size=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).astype(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = ops.expand_dims(emb_out, -1)
            # emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = ops.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        h = self.skip_connection(x) + h

        if self.use_temporal_conv and batch_size:
            # h = rearrange(h, "(b t) c h w -> b c t h w", b=batch_size)
            h = rearrange_in_gn5d_bs(h, batch_size)
            h = self.temopral_conv(h)
            # h = rearrange(h, "b c t h w -> (b t) c h w")
            h = rearrange_out_gn5d(h)
        return h


class TemporalConvBlock(nn.Cell):
    """
    Adapted from modelscope: https://github.com/modelscope/modelscope/blob/master/modelscope/models/multi_modal/video_synthesis/unet_sd.py
    """

    def __init__(self, in_channels, out_channels=None, dropout=0.0, spatial_aware=False, dtype=ms.float32):
        super(TemporalConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_shape = (3, 1, 1) if not spatial_aware else (3, 3, 3)
        padding_shape = (1, 1, 0, 0, 0, 0) if not spatial_aware else (1, 1, 1, 1, 1, 1)

        # conv layers
        self.conv1 = nn.SequentialCell(
            GroupNormExtend(32, in_channels),
            nn.SiLU(),
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_shape,
                padding=padding_shape,
                pad_mode="pad",
                has_bias=True,
                dtype=dtype,
            ).to_float(dtype),
        )
        self.conv2 = nn.SequentialCell(
            GroupNormExtend(32, out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(
                out_channels,
                in_channels,
                kernel_shape,
                padding=padding_shape,
                pad_mode="pad",
                has_bias=True,
                dtype=dtype,
            ).to_float(dtype),
        )
        self.conv3 = nn.SequentialCell(
            GroupNormExtend(32, out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(
                out_channels,
                in_channels,
                (3, 1, 1),
                padding=(1, 1, 0, 0, 0, 0),
                pad_mode="pad",
                has_bias=True,
                dtype=dtype,
            ).to_float(dtype),
        )
        self.conv4 = nn.SequentialCell(
            GroupNormExtend(32, out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(
                out_channels,
                in_channels,
                (3, 1, 1),
                padding=(1, 1, 0, 0, 0, 0),
                pad_mode="pad",
                has_bias=True,
                dtype=dtype,
            ).to_float(dtype),
        )

        # zero out the last layer params,so the conv block is identity
        self.conv4[-1].weight.set_data(initializer(Zero(), self.conv4[-1].weight.shape, self.conv4[-1].weight.dtype))
        self.conv4[-1].bias.set_data(initializer(Zero(), self.conv4[-1].bias.shape, self.conv4[-1].bias.dtype))

        # nn.init.zeros_(self.conv4[-1].weight)
        # nn.init.zeros_(self.conv4[-1].bias)

    def construct(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x + identity


class UNetModel(nn.Cell):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: in_channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        context_dim=None,
        use_scale_shift_norm=False,
        resblock_updown=False,
        num_heads=-1,
        num_head_channels=-1,
        transformer_depth=1,
        use_linear=False,
        use_checkpoint=False,
        temporal_conv=False,
        tempspatial_aware=False,
        temporal_attention=True,
        temporal_selfatt_only=True,
        use_relative_position=True,
        use_causal_attention=False,
        temporal_length=None,
        addition_attention=False,
        use_image_attention=False,
        temporal_transformer_depth=1,
        fps_cond=False,
        time_cond_proj_dim=None,
        dtype="fp32",
    ):
        super(UNetModel, self).__init__()
        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"
        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.temporal_attention = temporal_attention
        time_embed_dim = model_channels * 4
        self.use_checkpoint = use_checkpoint
        self.addition_attention = addition_attention
        self.use_image_attention = use_image_attention
        self.fps_cond = fps_cond
        self.time_cond_proj_dim = time_cond_proj_dim
        self.dtype = {"fp32": ms.float32, "fp16": ms.float16, "bf16": ms.bfloat16}[dtype]

        if use_checkpoint and self.training:
            tseq_class = TimestepEmbedSequentialRecompute
        else:
            tseq_class = TimestepEmbedSequential

        self.time_embed = nn.SequentialCell(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        ).to_float(self.dtype)
        if self.fps_cond:
            self.fps_embedding = nn.SequentialCell(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            ).to_float(self.dtype)
        if time_cond_proj_dim is not None:
            self.time_cond_proj = nn.Dense(time_cond_proj_dim, model_channels, has_bias=False).to_float(self.dtype)
        else:
            self.time_cond_proj = None

        input_blocks = nn.CellList(
            [
                tseq_class(
                    conv_nd(
                        dims, in_channels, model_channels, 3, padding=1, pad_mode="pad", has_bias=True, dtype=self.dtype
                    )
                ).to_float(self.dtype)
            ]
        )
        if self.addition_attention:
            init_attn = tseq_class(
                TemporalTransformer(
                    model_channels,
                    n_heads=8,
                    d_head=num_head_channels,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    use_checkpoint=use_checkpoint,
                    only_self_att=temporal_selfatt_only,
                    causal_attention=use_causal_attention,
                    relative_position=use_relative_position,
                    temporal_length=temporal_length,
                )
            ).to_float(self.dtype)

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        tempspatial_aware=tempspatial_aware,
                        use_temporal_conv=temporal_conv,
                        dtype=self.dtype,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_linear=use_linear,
                            use_checkpoint=use_checkpoint,
                            disable_self_attn=False,
                            img_cross_attention=self.use_image_attention,
                        ).to_float(self.dtype)
                    )
                    if self.temporal_attention:
                        layers.append(
                            TemporalTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=temporal_transformer_depth,
                                context_dim=context_dim,
                                use_linear=use_linear,
                                use_checkpoint=use_checkpoint,
                                only_self_att=temporal_selfatt_only,
                                causal_attention=use_causal_attention,
                                relative_position=use_relative_position,
                                temporal_length=temporal_length,
                            ).to_float(self.dtype)
                        )
                input_blocks.append(tseq_class(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                input_blocks.append(
                    tseq_class(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            dtype=self.dtype,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        layers = [
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                tempspatial_aware=tempspatial_aware,
                use_temporal_conv=temporal_conv,
                dtype=self.dtype,
            ),
            SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                use_linear=use_linear,
                use_checkpoint=use_checkpoint,
                disable_self_attn=False,
                img_cross_attention=self.use_image_attention,
            ).to_float(self.dtype),
        ]
        if self.temporal_attention:
            layers.append(
                TemporalTransformer(
                    ch,
                    num_heads,
                    dim_head,
                    depth=temporal_transformer_depth,
                    context_dim=context_dim,
                    use_linear=use_linear,
                    use_checkpoint=use_checkpoint,
                    only_self_att=temporal_selfatt_only,
                    causal_attention=use_causal_attention,
                    relative_position=use_relative_position,
                    temporal_length=temporal_length,
                ).to_float(self.dtype)
            )
        layers.append(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                tempspatial_aware=tempspatial_aware,
                use_temporal_conv=temporal_conv,
                dtype=self.dtype,
            )
        )
        middle_block = tseq_class(*layers).to_float(self.dtype)

        output_blocks = nn.CellList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        tempspatial_aware=tempspatial_aware,
                        use_temporal_conv=temporal_conv,
                        dtype=self.dtype,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_linear=use_linear,
                            use_checkpoint=use_checkpoint,
                            disable_self_attn=False,
                            img_cross_attention=self.use_image_attention,
                        ).to_float(self.dtype)
                    )
                    if self.temporal_attention:
                        layers.append(
                            TemporalTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=temporal_transformer_depth,
                                context_dim=context_dim,
                                use_linear=use_linear,
                                use_checkpoint=use_checkpoint,
                                only_self_att=temporal_selfatt_only,
                                causal_attention=use_causal_attention,
                                relative_position=use_relative_position,
                                temporal_length=temporal_length,
                            ).to_float(self.dtype)
                        )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            dtype=self.dtype,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype)
                    )
                    ds //= 2
                output_blocks.append(tseq_class(*layers))

        self.input_blocks = input_blocks
        self.init_attn = init_attn
        self.middle_block = middle_block
        self.output_blocks = output_blocks

        self.out = nn.SequentialCell(
            normalization(ch),
            nn.SiLU(),
            zero_module(
                conv_nd(
                    dims, model_channels, out_channels, 3, padding=1, pad_mode="pad", has_bias=True, dtype=self.dtype
                )
            ),
        ).to_float(self.dtype)

    def construct(self, x, timesteps, context, features_adapter=None, fps=16, timestep_cond=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=self.dtype)
        if timestep_cond is not None:
            t_emb = t_emb + self.time_cond_proj(timestep_cond)
        emb = self.time_embed(t_emb)

        if self.fps_cond:
            if type(fps) == int:
                fps = ops.full_like(timesteps, fps)
            fps_emb = timestep_embedding(fps, self.model_channels, repeat_only=False, dtype=self.dtype)
            emb += self.fps_embedding(fps_emb)

        b, _, t, _, _ = x.shape
        # repeat t times for context [(b t) 77 768] & time embedding
        context = context.repeat_interleave(repeats=t, dim=0)
        emb = emb.repeat_interleave(repeats=t, dim=0)

        # always in shape (b t) c h w, except for temporal layer
        # x = rearrange(x, "b c t h w -> (b t) c h w")
        x = rearrange_out_gn5d(x)

        h = x
        adapter_idx = 0
        hs = []
        for id, module in enumerate(self.input_blocks):
            h = module(h, emb=emb, context=context, batch_size=b)
            if id == 0 and self.addition_attention:
                h = self.init_attn(h, emb=emb, context=context, batch_size=b)
            # plug-in adapter features
            if ((id + 1) % 3 == 0) and features_adapter is not None:
                h = h + features_adapter[adapter_idx]
                adapter_idx += 1
            hs.append(h)
        if features_adapter is not None:
            assert len(features_adapter) == adapter_idx, "Wrong features_adapter"

        h = self.middle_block(h, emb=emb, context=context, batch_size=b)

        for i, module in enumerate(self.output_blocks):
            hs_pop = hs[-(i + 1)]
            h = mint.cat([h, hs_pop], dim=1)
            h = module(h, emb=emb, context=context, batch_size=b)

        h = h.astype(x.dtype)
        y = self.out(h)

        # reshape back to (b c t h w)
        # y = rearrange(y, "(b t) c h w -> b c t h w", b=b)
        y = rearrange_in_gn5d_bs(y, b)

        return y
