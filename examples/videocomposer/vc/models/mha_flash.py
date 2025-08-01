# This code is adapted from https://github.com/ali-vilab/videocomposer
# with modifications to run on MindSpore.

import math
import time

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import Normal, initializer
from mindspore.ops import FlashAttention


class FlashAttentionBlock(nn.Cell):
    def __init__(self, dim, context_dim=None, num_heads=None, head_dim=None, batch_size=4):
        # consider head_dim first, then num_heads
        num_heads = dim // head_dim if head_dim else num_heads
        head_dim = dim // num_heads
        assert num_heads * head_dim == dim
        super(FlashAttentionBlock, self).__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = math.pow(head_dim, -0.25)

        # layers
        self.norm = nn.GroupNorm(32, dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, has_bias=True)
        if context_dim is not None:
            self.context_kv = nn.Dense(context_dim, dim * 2)
        self.proj = nn.Conv2d(dim, dim, 1, has_bias=True)

        if self.head_dim <= 128 and (self.head_dim % 8) == 0:
            new_scale = math.pow(head_dim, -0.5)  # noqa
            self.flash_attn = FlashAttention(softmax_scale=None, attention_dropout=0.0)

        # zero out the last layer params
        self.proj.weight.set_data(initializer("zeros", self.proj.weight.shape, self.proj.weight.dtype))
        # self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Dense):
            module.weight.set_data(initializer(Normal(sigma=0.15, mean=0.0), module.weight.shape, module.weight.dtype))
            if module.bias is not None:
                module.bias.set_data(initializer("zeros", module.bias.shape, module.bias.dtype))
        elif isinstance(module, nn.Conv2d):
            module.weight.set_data(initializer(Normal(sigma=0.15, mean=0.0), module.weight.shape, module.weight.dtype))
            if module.bias is not None:
                module.bias.set_data(initializer("zeros", module.bias.shape, module.bias.dtype))

    def construct(self, x, context=None):
        r"""x:       [B, C, H, W].
        context: [B, L, C] or None.
        """
        identity = x
        b, c, h, w, n, d = *x.shape, self.num_heads, self.head_dim

        # compute query, key, value
        x = self.norm(x)
        q, k, v = self.to_qkv(x).view(b, n * 3, d, h * w).chunk(3, axis=1)
        if context is not None:
            ck, cv = self.context_kv(context).reshape(b, -1, n * 2, d).transpose(0, 2, 3, 1).chunk(2, axis=1)
            k = ops.cat([ck, k], axis=-1)
            v = ops.cat([cv, v], axis=-1)
            cq = ops.zeros([b, n, d, 4], dtype=q.dtype)
            q = ops.cat([q, cq], axis=-1)

        qkv = ops.cat([q, k, v], axis=1)
        origin_dtype = qkv.dtype
        qkv = qkv.transpose(0, 3, 1, 2).reshape(b, -1, 3, n, d).half()
        out, _ = self.flash_attn(qkv)
        out.to(origin_dtype)

        if context is not None:
            out = out[:, :-4, :, :]
        out = out.transpose(0, 2, 3, 1).reshape(b, c, h, w)

        # output
        x = self.proj(out)
        return x + identity


if __name__ == "__main__":
    batch_size = 8
    flash_net = FlashAttentionBlock(dim=1280, context_dim=512, num_heads=None, head_dim=64, batch_size=batch_size)

    x = ops.randn([batch_size, 1280, 32, 32], dtype=ms.float32)
    context = ops.randn([batch_size, 4, 512], dtype=ms.float32)
    # context = None
    flash_net.set_train(False)

    # with amp.autocast(enabled=True):
    # warm up
    for i in range(5):
        y = flash_net(x, context)
    # torch.cuda.synchronize()
    s1 = time.time()
    for i in range(10):
        y = flash_net(x, context)
    # torch.cuda.synchronize()
    s2 = time.time()

    print(f"Average cost time {(s2-s1)*1000/10} ms")
