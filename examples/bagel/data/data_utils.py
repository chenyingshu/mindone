# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import math
import random
from typing import Callable

import numpy as np
from PIL import Image

import mindspore as ms
from mindspore import Tensor, mint, ops

_MIN_FP16 = ms.tensor(np.finfo(np.float16).min, dtype=ms.float16)
_mask_mod_signature = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


# refer to torch.nn.attention.flex_attention.or_masks and and_masks
def or_masks(*mask_mods: _mask_mod_signature) -> _mask_mod_signature:
    """Returns a mask_mod that's the union of provided mask_mods"""
    if not all(callable(arg) for arg in mask_mods):
        raise RuntimeError(f"All inputs should be callable mask_mods: {mask_mods}")

    def or_mask(b, h, q_idx, kv_idx):
        result = mint.zeros_like(b, dtype=ms.bool_)
        for mask in mask_mods:
            result = result | mask(b, h, q_idx, kv_idx)
        return result

    return or_mask


# refer to torch.nn.attention.flex_attention.and_masks
def and_masks(*mask_mods: _mask_mod_signature) -> _mask_mod_signature:
    """Returns a mask_mod that's the intersection of provided mask_mods"""
    if not all(callable(arg) for arg in mask_mods):
        raise RuntimeError(f"All inputs should be callable mask_mods: {mask_mods}")

    def and_mask(b, h, q_idx, kv_idx):
        result = mint.ones_like(b, dtype=ms.bool_)
        for mask in mask_mods:
            result = result & mask(b, h, q_idx, kv_idx)
        return result

    return and_mask


def create_sparse_mask(document_lens, split_lens, attn_modes):
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return ~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx]))

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ["full", "noise"] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == "noise" else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = ms.Tensor(full_and_noise_tmp)
    noise_seq_id = ms.Tensor(noise_tmp)

    document_id = mint.cat([mint.full((l,), i) for i, l in enumerate(document_lens, start=1)])

    return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)


def patchify(image, patch_size):
    p = patch_size
    c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(c, h // p, p, w // p, p)
    image = mint.einsum("chpwq->hwpqc", image)
    image = image.reshape(-1, p**2 * c)
    return image


def get_flattened_position_ids_extrapolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = mint.arange(0, num_patches_h)
    coords_w = mint.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten(start_dim=0)
    return pos_ids


def get_flattened_position_ids_interpolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    boundaries = mint.arange(1 / max_num_patches_per_side, 1.0, 1 / max_num_patches_per_side)
    fractional_coords_h = mint.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = mint.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = ops.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = ops.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = (bucket_coords_h[:, None] * max_num_patches_per_side + bucket_coords_w).flatten(start_dim=0)
    return pos_ids


def prepare_attention_mask_per_sample(split_lens, attn_modes):
    """
    nested_split_lens: A list of N lists of ints. Each int indicates the length of a split within
        a sample, where each sample contains multiple splits with different attn modes.
    nested_attn_modes: whether to use full attn in each split.
    """
    sample_len = sum(split_lens)
    attention_mask = mint.zeros((sample_len, sample_len), dtype=ms.bool_)

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ["causal", "full", "noise"]
        if attn_mode == "causal":
            attention_mask[csum : csum + s, csum : csum + s] = mint.ones((s, s), dtype=ms.bool_).tril()
            attention_mask[csum : csum + s, :csum] = 1
        else:
            attention_mask[csum : csum + s, csum : csum + s] = mint.ones((s, s), dtype=ms.bool_)
            attention_mask[csum : csum + s, :csum] = 1
        csum += s

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            attention_mask[:, csum : csum + s] = mint.zeros((sample_len, s), dtype=ms.bool_)
            attention_mask[csum : csum + s, csum : csum + s] = mint.ones((s, s), dtype=ms.bool_)
        csum += s

    attention_mask = mint.zeros_like(attention_mask, dtype=ms.float16).masked_fill_(
        ~attention_mask, _MIN_FP16
    )  # TODO: double check if the change works: float32/float("-inf") -> float16/_MIN_FP16

    return attention_mask


def split_integer_exp_decay(S, ng_sample_decay=1.0):
    if ng_sample_decay == 1.0:
        N = random.randint(1, S)
    else:
        base = (1 - ng_sample_decay) / (1 - math.pow(ng_sample_decay, S))
        p = [base * math.pow(ng_sample_decay, i) for i in range(S)]
        N = random.choices(list(range(1, S + 1)), p, k=1)[0]
    cumsum = [0] + sorted(random.sample(range(1, S), N - 1)) + [S]
    result = [cumsum[i + 1] - cumsum[i] for i in range(len(cumsum) - 1)]
    return result, cumsum


def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image


def add_special_tokens(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []

    if "<|im_start|>" not in all_special_tokens:
        new_tokens.append("<|im_start|>")

    if "<|im_end|>" not in all_special_tokens:
        new_tokens.append("<|im_end|>")

    if "<|vision_start|>" not in all_special_tokens:
        new_tokens.append("<|vision_start|>")

    if "<|vision_end|>" not in all_special_tokens:
        new_tokens.append("<|vision_end|>")

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    start_of_image = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end_of_image = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    new_token_ids = dict(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        start_of_image=start_of_image,
        end_of_image=end_of_image,
    )

    return tokenizer, new_token_ids, num_new_tokens


def len2weight(x, loss_reduction="square"):
    if x == 0:
        return x
    if loss_reduction == "token":
        return 1
    if loss_reduction == "sample":
        return 1 / x
    if loss_reduction == "square":
        return 1 / (x**0.5)
    raise NotImplementedError(loss_reduction)
