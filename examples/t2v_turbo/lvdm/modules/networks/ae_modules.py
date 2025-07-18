# This code is adapted from https://github.com/Ji4chenLi/t2v-turbo
# with modifications to run on MindSpore.

# pytorch_diffusion + derived encoder decoder
import math

import numpy as np
from lvdm.common import GroupNormExtend
from lvdm.distributions import DiagonalGaussianDistribution
from lvdm.modules.attention import LinearAttention
from utils.utils import instantiate_from_config

import mindspore as ms
from mindspore import mint, nn, ops


def nonlinearity(x):
    # swish
    return x * mint.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return GroupNormExtend(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Cell):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, has_bias=True)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, has_bias=True)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, has_bias=True)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, has_bias=True)

    def construct(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)  # bcl
        q = q.permute(0, 2, 1)  # bcl -> blc l=hw
        k = k.reshape(b, c, h * w)  # bcl

        w_ = mint.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = mint.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = mint.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f"attn_type {attn_type} unknown"
    # print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class Downsample(nn.Cell):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        self.in_channels = in_channels
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0, has_bias=True, pad_mode="valid"
            )

    def construct(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = ops.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = ops.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Upsample(nn.Cell):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        self.in_channels = in_channels
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
            )

    def construct(self, x):
        # x = ops.interpolate(x, scale_factor=2.0, mode="nearest")
        x = ops.ResizeNearestNeighbor((x.shape[2] * 2, x.shape[3] * 2))(x)
        if self.with_conv:
            x = self.conv(x)
        return x


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = mint.exp(mint.arange(half_dim, dtype=ms.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = mint.cat([mint.sin(emb), mint.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = mint.pad(emb, (0, 1, 0, 0))
    return emb


class ResnetBlock(nn.Cell):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )
        if temb_channels > 0:
            self.temb_proj = nn.Dense(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0, has_bias=True
                )

    def construct(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Encoder(nn.Cell):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.CellList()
        for i_level in range(self.num_resolutions):
            block = nn.CellList()
            attn = nn.CellList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Cell()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            pad_mode="pad",
            has_bias=True,
        )

    def construct(self, x):
        # timestep embedding
        temb = None

        # print(f'encoder-input={x.shape}')
        # downsampling
        hs = [self.conv_in(x)]
        # print(f'encoder-conv in feat={hs[0].shape}')
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # print(f'encoder-down feat={h.shape}')
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                # print(f'encoder-downsample (input)={hs[-1].shape}')
                hs.append(self.down[i_level].downsample(hs[-1]))
                # print(f'encoder-downsample (output)={hs[-1].shape}')

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        # print(f'encoder-mid1 feat={h.shape}')
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # print(f'encoder-mid2 feat={h.shape}')

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # print(f'end feat={h.shape}')
        return h


class Decoder(nn.Cell):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="vanilla",
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("AE working on z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.CellList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.CellList()
            attn = nn.CellList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Cell()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True)

    def construct(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # print(f'decoder-input={z.shape}')
        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)
        # print(f'decoder-conv in feat={h.shape}')

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # print(f'decoder-mid feat={h.shape}')

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
                # print(f'decoder-up feat={h.shape}')
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                # print(f'decoder-upsample feat={h.shape}')

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # print(f'decoder-conv_out feat={h.shape}')
        if self.tanh_out:
            h = mint.tanh(h)
        return h


class SimpleDecoder(nn.Cell):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = nn.CellList(
            [
                nn.Conv2d(in_channels, in_channels, 1, has_bias=True),
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    temb_channels=0,
                    dropout=0.0,
                ),
                ResnetBlock(
                    in_channels=2 * in_channels,
                    out_channels=4 * in_channels,
                    temb_channels=0,
                    dropout=0.0,
                ),
                ResnetBlock(
                    in_channels=4 * in_channels,
                    out_channels=2 * in_channels,
                    temb_channels=0,
                    dropout=0.0,
                ),
                nn.Conv2d(2 * in_channels, in_channels, 1, has_bias=True),
                Upsample(in_channels, with_conv=True),
            ]
        )
        # end
        self.norm_out = Normalize(in_channels)
        self.conv_out = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

    def construct(self, x):
        for i, layer in enumerate(self.model):
            if i in [1, 2, 3]:
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x


class UpsampleDecoder(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        ch,
        num_res_blocks,
        resolution,
        ch_mult=(2, 2),
        dropout=0.0,
    ):
        super().__init__()
        # upsampling
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.CellList()
        self.upsample_blocks = nn.CellList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            self.res_blocks.append(nn.CellList(res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                curr_res = curr_res * 2

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in, out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

    def construct(self, x):
        # upsampling
        h = x
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class LatentRescaler(nn.Cell):
    def __init__(self, factor, in_channels, mid_channels, out_channels, depth=2):
        super().__init__()
        # residual block, interpolate, residual block
        self.factor = factor
        self.conv_in = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )
        self.res_block1 = nn.CellList(
            [
                ResnetBlock(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    temb_channels=0,
                    dropout=0.0,
                )
                for _ in range(depth)
            ]
        )
        self.attn = AttnBlock(mid_channels)
        self.res_block2 = nn.CellList(
            [
                ResnetBlock(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    temb_channels=0,
                    dropout=0.0,
                )
                for _ in range(depth)
            ]
        )

        self.conv_out = nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=1,
            has_bias=True,
        )

    def construct(self, x):
        x = self.conv_in(x)
        for block in self.res_block1:
            x = block(x, None)
        x = ops.interpolate(
            x,
            size=(
                int(round(x.shape[2] * self.factor)),
                int(round(x.shape[3] * self.factor)),
            ),
        )
        x = self.attn(x)
        for block in self.res_block2:
            x = block(x, None)
        x = self.conv_out(x)
        return x


class MergedRescaleEncoder(nn.Cell):
    def __init__(
        self,
        in_channels,
        ch,
        resolution,
        out_ch,
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        ch_mult=(1, 2, 4, 8),
        rescale_factor=1.0,
        rescale_module_depth=1,
    ):
        super().__init__()
        intermediate_chn = ch * ch_mult[-1]
        self.encoder = Encoder(
            in_channels=in_channels,
            num_res_blocks=num_res_blocks,
            ch=ch,
            ch_mult=ch_mult,
            z_channels=intermediate_chn,
            double_z=False,
            resolution=resolution,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            out_ch=None,
        )
        self.rescaler = LatentRescaler(
            factor=rescale_factor,
            in_channels=intermediate_chn,
            mid_channels=intermediate_chn,
            out_channels=out_ch,
            depth=rescale_module_depth,
        )

    def construct(self, x):
        x = self.encoder(x)
        x = self.rescaler(x)
        return x


class MergedRescaleDecoder(nn.Cell):
    def __init__(
        self,
        z_channels,
        out_ch,
        resolution,
        num_res_blocks,
        attn_resolutions,
        ch,
        ch_mult=(1, 2, 4, 8),
        dropout=0.0,
        resamp_with_conv=True,
        rescale_factor=1.0,
        rescale_module_depth=1,
    ):
        super().__init__()
        tmp_chn = z_channels * ch_mult[-1]
        self.decoder = Decoder(
            out_ch=out_ch,
            z_channels=tmp_chn,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            in_channels=None,
            num_res_blocks=num_res_blocks,
            ch_mult=ch_mult,
            resolution=resolution,
            ch=ch,
        )
        self.rescaler = LatentRescaler(
            factor=rescale_factor,
            in_channels=z_channels,
            mid_channels=tmp_chn,
            out_channels=tmp_chn,
            depth=rescale_module_depth,
        )

    def construct(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Upsampler(nn.Cell):
    def __init__(self, in_size, out_size, in_channels, out_channels, ch_mult=2):
        super().__init__()
        assert out_size >= in_size
        num_blocks = int(np.log2(out_size // in_size)) + 1
        factor_up = 1.0 + (out_size % in_size)
        print(
            f"Building {self.__class__.__name__} with in_size: {in_size} --> out_size {out_size} and factor {factor_up}"
        )
        self.rescaler = LatentRescaler(
            factor=factor_up,
            in_channels=in_channels,
            mid_channels=2 * in_channels,
            out_channels=in_channels,
        )
        self.decoder = Decoder(
            out_ch=out_channels,
            resolution=out_size,
            z_channels=in_channels,
            num_res_blocks=2,
            attn_resolutions=[],
            in_channels=None,
            ch=in_channels,
            ch_mult=[ch_mult for _ in range(num_blocks)],
        )

    def construct(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Resize(nn.Cell):
    def __init__(self, in_channels=None, learned=False, mode="bilinear"):
        super().__init__()
        self.with_conv = learned
        self.mode = mode
        if self.with_conv:
            print(f"Note: {self.__class__.__name} uses learned downsampling and will ignore the fixed {mode} mode")
            raise NotImplementedError()
            assert in_channels is not None
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1, pad_mode="pad")

    def construct(self, x, scale_factor=1.0):
        if scale_factor == 1.0:
            return x
        else:
            x = ops.interpolate(x, mode=self.mode, align_corners=False, scale_factor=scale_factor)
        return x


class FirstStagePostProcessor(nn.Cell):
    def __init__(
        self,
        ch_mult: list,
        in_channels,
        pretrained_model: nn.Cell = None,
        reshape=False,
        n_channels=None,
        dropout=0.0,
        pretrained_config=None,
    ):
        super().__init__()
        if pretrained_config is None:
            assert pretrained_model is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.pretrained_model = pretrained_model
        else:
            assert pretrained_config is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.instantiate_pretrained(pretrained_config)

        self.do_reshape = reshape

        if n_channels is None:
            n_channels = self.pretrained_model.encoder.ch

        self.proj_norm = Normalize(in_channels, num_groups=in_channels // 2)
        self.proj = nn.Conv2d(
            in_channels, n_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

        blocks = []
        downs = []
        ch_in = n_channels
        for m in ch_mult:
            blocks.append(ResnetBlock(in_channels=ch_in, out_channels=m * n_channels, dropout=dropout))
            ch_in = m * n_channels
            downs.append(Downsample(ch_in, with_conv=False))

        self.model = nn.CellList(blocks)
        self.downsampler = nn.CellList(downs)

    def instantiate_pretrained(self, config):
        model = instantiate_from_config(config)
        self.pretrained_model = model.eval()
        # self.pretrained_model.train = False
        for param in self.pretrained_model.get_parameters():
            param.requires_grad = False

    def encode_with_pretrained(self, x):
        c = self.pretrained_model.encode(x)
        if isinstance(c, DiagonalGaussianDistribution):
            c = c.mode()
        return c

    def construct(self, x):
        z_fs = self.encode_with_pretrained(x)
        z = self.proj_norm(z_fs)
        z = self.proj(z)
        z = nonlinearity(z)

        for submodel, downmodel in zip(self.model, self.downsampler):
            z = submodel(z, temb=None)
            z = downmodel(z)

        if self.do_reshape:
            b, c, h, w = z.shape
            z = z.reshape(b, -1, c)
            # z = rearrange(z, "b c h w -> b (h w) c")
        return z
