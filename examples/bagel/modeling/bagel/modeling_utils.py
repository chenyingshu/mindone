# Copyright (c) 2022 Facebook, Inc. and its affiliates.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: CC BY-NC 4.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under CC BY-NC 4.0, with the full license text
# available at https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt.
#
# This modified file is released under the same license.

import math

import numpy as np

import mindspore as ms
import mindspore.mint as mint
from mindspore import nn

from mindone.transformers.activations import ACT2FN


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# TimestepEmbedder
# Reference:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
class TimestepEmbedder(nn.Cell):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.SequentialCell(
            mint.nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            mint.nn.SiLU(),
            mint.nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = mint.exp(-math.log(max_period) * mint.arange(start=0, end=half, dtype=ms.float32) / half)
        args = t[:, None].float() * freqs[None]
        embedding = mint.cat([mint.cos(args), mint.sin(args)], dim=-1)
        if dim % 2:
            embedding = mint.cat([embedding, mint.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def construct(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(ms.bfloat16)
        t_emb = self.mlp(t_freq)
        return t_emb


class MLPconnector(nn.Cell):
    def __init__(self, in_dim: int, out_dim: int, hidden_act: str):
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = mint.nn.Linear(in_dim, out_dim)
        self.fc2 = mint.nn.Linear(out_dim, out_dim)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class PositionEmbedding(nn.Cell):
    def __init__(self, max_num_patch_per_side, hidden_size):
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size
        self.pos_embed = ms.Parameter(mint.zeros((max_num_patch_per_side**2, hidden_size)), requires_grad=False)
        self._init_weights()

    def _init_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.max_num_patch_per_side)
        self.pos_embed.set_data(ms.tensor(pos_embed).float())

    def construct(self, position_ids):
        return self.pos_embed[position_ids]
