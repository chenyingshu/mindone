"""Adapted from https://github.com/huggingface/transformers/tree/main/src/transformers/integrations/sdpa_attention.py."""
from math import sqrt
from typing import Optional

import mindspore as ms
from mindspore import mint, nn, ops

from ..utils import logging

logger = logging.get_logger(__name__)


def repeat_kv(hidden_states: ms.Tensor, n_rep: int) -> ms.Tensor:
    """
    This is the equivalent of mint.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape  # BNSD format
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand((batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def sdpa_attention_forward(
    module: nn.Cell,
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attention_mask: Optional[ms.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[ms.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask", None) is not None:
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    if is_causal:
        if attention_mask is not None:
            raise ValueError("Causal mode cannot be used with an explicit `attention_mask`")
        attention_mask = mint.ones((query.shape[-2], key.shape[-2]), dtype=ms.bool_).tril(diagonal=0)

    if attention_mask is not None:
        attention_mask = mint.logical_not(attention_mask)  # in MindSpore, 0 indicates retain, 1 indicates discard

    scaling = 1 / sqrt(query.shape[-1]) if scaling is None else scaling

    attn_output = ops.speed_fusion_attention(
        query,
        key,
        value,
        head_num=query.shape[1],
        input_layout="BNSD",
        atten_mask=attention_mask,
        scale=scaling,
        keep_prob=1 - dropout,
    )[0]
    attn_output = mint.transpose(attn_output, 1, 2).contiguous()

    return attn_output, None
