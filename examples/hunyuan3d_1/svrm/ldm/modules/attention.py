# ref to `stable_diffusion_v2/ldm/modules/attention.py`

from inspect import isfunction
import math
import mindspore as ms
from mindspore import nn, mint, ops 
from mindone.utils.version_control import (
    # MS_VERSION,
    check_valid_flash_attention,
    # choose_flash_attention_dtype,
    is_old_ms_version,
)
import numpy as np

from mindone.models.modules.flash_attention import FLASH_IS_AVAILABLE, MSFlashAttention
XFORMERS_IS_AVAILBLE = FLASH_IS_AVAILABLE
from mindspore.common.initializer import initializer, XavierUniform
from .util import dtype_to_max

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    # return d() if isfunction(d) else d
    # this may lead to error in mindspore 2.1. use isinstance, and if return, return
    if isinstance(d, (ms.Tensor, int, float)):
        return d
    return d

def max_neg_value(t):
    return -dtype_to_max(t.dtype)


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor = ops.uniform(shape=tensor.shape, minval=-std, maxval=std)
    return tensor

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    # if flag:
    #     args = tuple(inputs) + tuple(params)
    #     return CheckpointFunction.apply(func, len(inputs), *args)
    # else:
    #     return func(*inputs)
    return func(*inputs)

# TODO:
# class CheckpointFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, run_function, length, *args):
#         ctx.run_function = run_function
#         ctx.input_tensors = list(args[:length])
#         ctx.input_params = list(args[length:])

#         with torch.no_grad():
#             output_tensors = ctx.run_function(*ctx.input_tensors)
#         return output_tensors

#     @staticmethod
#     def backward(ctx, *output_grads):
#         ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
#         with torch.enable_grad():
#             # Fixes a bug where the first op in run_function modifies the
#             # Tensor storage in place, which is not allowed for detach()'d
#             # Tensors.
#             shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
#             output_tensors = ctx.run_function(*shallow_copies)
#         input_grads = torch.autograd.grad(
#             output_tensors,
#             ctx.input_tensors + ctx.input_params,
#             output_grads,
#             allow_unused=True,
#         )
#         del ctx.input_tensors
#         del ctx.input_params
#         del output_tensors
#         return (None, None) + input_grads


# feedforward
class GEGLU(nn.Cell):
    def __init__(self, dim_in, dim_out, dtype=ms.float32):
        super().__init__()
        self.proj = nn.Dense(dim_in, dim_out * 2).to_float(dtype)

    def construct(self, x):
        x, gate = self.proj(x).chunk(2, axis=-1)
        return x * mint.nn.functional.gelu(gate)


class FeedForward(nn.Cell):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0., dtype=ms.float32):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.SequentialCell(
            nn.Dense(dim, inner_dim).to_float(dtype),
            nn.GELU(approximate=False).to_float(dtype)
            ) if not glu else GEGLU(dim, inner_dim, dtype=dtype)
        )

        self.net = nn.SequentialCell(
            project_in,
            nn.Dropout(p = dropout),
            nn.Dense(inner_dim, dim_out).to_float(dtype)
        )

    def construct(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    weight = initializer("zeros", module.weight.shape)
    bias_weight = initializer("zeros", module.bias.shape)
    module.weight.set_data(weight)
    module.bias.set_data(bias_weight)
    return module


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Cell):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, has_bias=False, pad_mode="pad")
        self.to_out = nn.Conv2d(hidden_dim, dim, 1, has_bias=True, pad_mode="pad")

    def construct(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)        
        # 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3
        qkv_b, _, qkv_h, qkv_w = qkv.shape
        q, k, v = qkv.reshape(qkv_b, 3, self.head, -1, qkv_h, qkv_w).permute((1, 0, 2, 4, 5)).reshape(3, qkv_b, self.heads, -1, qkv_h*qkv_w)

        k = k.softmax(dim=-1)  
        context = ops.einsum('bhdn,bhen->bhde', k, v)
        out = ops.einsum('bhde,bhdn->bhen', context, q)
        
        # 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w
        b, _, c, _ = out.shape
        out = out.reshape(b, self.heads, c, h*w).reshape(b, heads*c, h, w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Cell):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                has_bias=True, 
                                pad_mode="pad")
        self.k = nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                has_bias=True, 
                                pad_mode="pad")
        self.v = nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                has_bias=True, 
                                pad_mode="pad")
        self.proj_out = nn.Conv2d(in_channels,
                                    in_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    has_bias=True, 
                                    pad_mode="pad")

    def construct(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        # 'b c h w -> b (h w) c'
        q = q.reshape(q.shape[0], -1, q.shape[-1])
        # 'b c h w -> b c (h w)'
        k = k.reshape(k.shape[0], k.shape[1], -1)
        w_ = ops.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = mint.nn.functional.softmax(w_, dim=2)

        # attend to values
        # 'b c h w -> b c (h w)'
        v = v.reshape(v.shape[0], v.shape[1], -1)
        # 'b i j -> b j i'
        w_ = w_.swapaxes(1, 2)
        h_ = ops.einsum('bij,bjk->bik', v, w_)
        # 'b c (h w) -> b c h w', h=h
        h_ = h_.reshape(h.shape[0], h.shape[1], h, -1)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Dense(query_dim, inner_dim, has_bias=False)
        self.to_k = nn.Dense(context_dim, inner_dim, has_bias=False)
        self.to_v = nn.Dense(context_dim, inner_dim, has_bias=False)

        self.to_out = nn.SequentialCell(
            nn.Dense(inner_dim, query_dim),
            nn.Dropout(p = dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = ops.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            # 'b ... -> b (...)'
            mask = mask.reshape(mask.shape[0], -1)
            max_neg_value = -dtype_to_max(sim.dtype)
            # repeat(mask, 'b j -> (b h) () j', h=h)
            mask = mask.tile((h, 1))[:, None, :]
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = mint.nn.functional.softmax(sim, dim=-1)
        out = ops.einsum('b i j, b j d -> b i d', attn, v) # [b*h, n, d]
        # '(b h) n d -> b n (h d)', h=h
        _, out_n, out_d = out.shape
        out = out.reshape(-1, h, out_n, out_d).permute((0, 2, 1, 3)).reshape(-1, out_n, h*out_d)

        return self.to_out(out)


class FlashAttention(nn.Cell):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., dtype = ms.bfloat16):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        # self.dtype = dtype
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dropout = dropout
        self.to_q = nn.Dense(query_dim, inner_dim, has_bias=False)
        self.to_k = nn.Dense(context_dim, inner_dim, has_bias=False)
        self.to_v = nn.Dense(context_dim, inner_dim, has_bias=False)
        self.to_out = nn.SequentialCell(
            nn.Dense(inner_dim, query_dim),
            nn.Dropout(p = dropout)
        )

        self.flash_attention = MSFlashAttention(head_dim=dim_head, head_num=heads, input_layout="BNSD")
        self.flash_attention = MSFlashAttention(
                head_dim=dim_head,
                head_num=heads,
                attention_dropout=dropout,
                input_layout="BNSD",
                dtype=dtype,
            )

    def construct(self, x, context=None, mask=None):
        context = default(context, x)
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h), (q, k, v)) # q is [b, 3079, 16, 64]
        q = q.reshape(q.shape[0], q.shape[1], h, -1)
        k = k.reshape(k.shape[0], k.shape[1], h, -1)
        v = v.reshape(v.shape[0], v.shape[1], h, -1)
        # out = flash_attn_func(q, k, v, dropout_p=self.dropout, softmax_scale=None, causal=False, window_size=(-1, -1)) # out is same shape to q
        out = self.flash_attention(q, k, v) # out is same shape to q
        # 'b n h d -> b n (h d)', h=h
        out = out.reshape(out.shape[0], out.shape[1], -1)
        return self.to_out(out.float())

# class MemoryEfficientCrossAttention(nn.Cell):
#     # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
#     def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
#         super().__init__()
#         print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
#               f"{heads} heads.")
#         inner_dim = dim_head * heads
#         context_dim = default(context_dim, query_dim)

#         self.heads = heads
#         self.dim_head = dim_head

#         self.to_q = nn.Dense(query_dim, inner_dim, has_bias=False)
#         self.to_k = nn.Dense(context_dim, inner_dim, has_bias=False)
#         self.to_v = nn.Dense(context_dim, inner_dim, has_bias=False)
#         self.to_out = nn.SequentialCell(
#             nn.Dense(inner_dim, query_dim),
#             nn.Dropout(p=dropout),
#         )
#         self.attention_op: Optional[Any] = None

#     def construct(self, x, context=None, mask=None):
#         q = self.to_q(x)
#         context = default(context, x)
#         k = self.to_k(context)
#         v = self.to_v(context)

#         b, _, _ = q.shape
#         q, k, v = map(
#             lambda t: t.unsqueeze(3)
#             .reshape(b, t.shape[1], self.heads, self.dim_head)
#             .permute((0, 2, 1, 3))
#             .reshape(b * self.heads, t.shape[1], self.dim_head)
#             .contiguous(),
#             (q, k, v),
#         )

#         # actually compute the attention, what we cannot get enough of
#         out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

#         if exists(mask):
#             raise NotImplementedError
#         out = (
#             out.unsqueeze(0)
#             .reshape(b, self.heads, out.shape[1], self.dim_head)
#             .permute(0, 2, 1, 3)
#             .reshape(b, out.shape[1], self.heads * self.dim_head)
#         )
#         return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                    context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = Fp32LayerNorm(dim)
        self.norm2 = Fp32LayerNorm(dim)
        self.norm3 = Fp32LayerNorm(dim)
        self.checkpoint = checkpoint

    def construct(self, x, context=None):
        # return checkpoint(self._construct, (x, context), self.parameters(), self.checkpoint)
        return self._construct(x, context)

    def _construct(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

ATTENTION_MODES = {
    "softmax": CrossAttention,  # vanilla attention
    # "softmax-xformers": MemoryEfficientCrossAttention,  #TODO
    "softmax-flash": FlashAttention
}

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Fp32LayerNorm(nn.Cell):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine: bool = True, dtype=ms.float32):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.gamma = Parameter(initializer("ones", normalized_shape, dtype=dtype))
            self.beta = Parameter(initializer("zeros", normalized_shape, dtype=dtype))
        else:
            self.gamma = ops.ones(normalized_shape, dtype=dtype)
            self.beta = ops.zeros(normalized_shape, dtype=dtype)
        self.layer_norm = ops.LayerNorm(-1, -1, epsilon=eps)

    def construct(self, x: ms.Tensor):
        x, _, _ = self.layer_norm(x, self.gamma, self.beta)
        return x



class AdaNorm(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.adaLN_modulation = nn.SequentialCell(
            nn.SiLU(),
            nn.Dense(dim, 2 * dim, has_bias=True)
        )
        self.norm = Fp32LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def construct(self, x, c):  # x is fp32, c is fp16
        shift, scale = self.adaLN_modulation(c.float()).chunk(2, axis=1) # bf16
        x = modulate(self.norm(x), shift, scale) # fp32
        return x


class BasicTransformerBlockLRM(nn.Cell):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, \
                 checkpoint=True):
        super().__init__()

        # attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        attn_mode = "softmax-flash" if FLASH_IS_AVAILABLE else "softmax"
        assert attn_mode in ATTENTION_MODES
        attn_cls = ATTENTION_MODES[attn_mode]

        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, \
                              context_dim=context_dim) # cross-attn
        self.attn2 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, \
                              context_dim=None) # self-attn

        self.norm1 = Fp32LayerNorm(dim)
        self.norm2 = Fp32LayerNorm(dim)
        self.norm3 = Fp32LayerNorm(dim)
        
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.checkpoint = checkpoint

    def construct(self, x, context=None, cam_emb=None): # (float32, float32, bfloat16)
        # return checkpoint(self._construct, (x, context), self.parameters(), self.checkpoint)
        return self._construct(x, context)


    def _construct(self, x, context=None, cam_emb=None):

        x = self.attn1(self.norm1(x), context=context) + x  # cross-attn
        x = self.attn2(self.norm2(x), context=None) + x # self-attn
        x = self.ff(self.norm3(x)) + x

        return x


class ImgToTriplaneTransformer(nn.Cell):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, query_dim, n_heads, d_head, depth=1, dropout=0., context_dim=None, triplane_size=64):
        super().__init__()

        self.transformer_blocks = nn.CellList([
            BasicTransformerBlockLRM(query_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim) 
            for d in range(depth)])
        
        self.norm = Fp32LayerNorm(query_dim, eps=1e-6)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Dense): 
                weight = initializer("xavier_uniform", module.weight.shape)
                module.weight.set_data(weight)
                if module.bias is not None:                    
                    bias_weight = initializer("zeros", module.bias.shape)
                    module.bias.set_data(bias_weight)
            elif isinstance(module, Fp32LayerNorm):
                if module.bias is not None:       
                    bias_weight = initializer("zeros", module.bias.shape)
                    module.bias.set_data(bias_weight)
                if module.weight is not None:      
                    weight = initializer("ones", module.bias.shape)
                    module.weight.set_data(weight)
        self.apply(_basic_init)

    def construct(self, x, context=None, cam_emb=None):
        # note: if no context is given, cross-attention defaults to self-attention
        for block in self.transformer_blocks:
            x = block(x, context=context)  
        x = self.norm(x)
        return x




