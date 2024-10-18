import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import mindspore as ms
from mindspore import nn, ops
from packaging import version

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import (
    # is_hqq_available,
    # is_optimum_quanto_available,
    is_quanto_available,
    logging,
)
from transformers.utils.deprecation import deprecate_kwarg


# if is_hqq_available():
#     from hqq.core.quantize import Quantizer as HQQQuantizer

logger = logging.get_logger(__name__)


class Cache(nn.Cell):
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def __init__(self):
        super().__init__()

    def update(
        self,
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`ms.Tensor`):
                The new key states to cache.
            value_states (`ms.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    # Deprecate in favor of max-cache-shape because we want to be specifc by what we mean with "max_length"
    # Prev some cache objects didn't have "max_length" (SlidingWindowCache or SinkCache) because the cache object technically handles
    # infinite amount of tokens. In the codebase what we really need to check is the max capacity of certain cache instances, so
    # we change naming to be more explicit
    def get_max_length(self) -> Optional[int]:
        logger.warning_once(
            "`get_max_cache()` is deprecated for all Cache classes. Use `get_max_cache_shape()` instead. "
            "Calling `get_max_cache()` will raise error from v4.48"
        )
        return self.get_max_cache_shape()

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        raise NotImplementedError("Make sure to implement `get_max_cache_shape` in a subclass.")

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_cache_shape()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reorder_cache(self, beam_idx: ms.Tensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] != []:
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx)
            if self.value_cache[layer_idx] != []:
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)

    @property
    def seen_tokens(self):
        logger.warning_once(
            "The `seen_tokens` attribute is deprecated and will be removed in v4.41. Use the `cache_position` "
            "model input instead."
        )
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None


@dataclass
class CacheConfig:
    """
    Base class for cache configs
    """

    cache_implementation: None

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Constructs a CacheConfig instance from a dictionary of parameters.
        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration parameters.
            **kwargs: Additional keyword arguments to override dictionary values.

        Returns:
            CacheConfig: Instance of CacheConfig constructed from the dictionary.
        """
        config = cls(**config_dict)
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        return config

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.to_json_file
    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default
                `QuantizationConfig()` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.to_dict
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return copy.deepcopy(self.__dict__)

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.__iter__
    def __iter__(self):
        """allows `dict(obj)` for situations where obj may be a dict or QuantizationConfigMixin"""
        for attr, value in copy.deepcopy(self.__dict__).items():
            yield attr, value

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.__repr__
    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self):
        """
        Serializes this instance to a JSON formatted string.
        Returns:
            str: JSON formatted string representing the configuration instance.
        """
        return json.dumps(self.__dict__, indent=2) + "\n"

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.update
    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs


@dataclass
class QuantizedCacheConfig(CacheConfig):
    """
    Configuration class for quantized cache settings.

    Attributes:
        backend (`str`, *optional*, defaults to `"quanto"`):
            Backend to use when performing quantization, Can be one of [`quanto`, `HQQ`]
        nbits (`Optional[int]`, *optional*, defaults to 4):
            Number of bits, can be 2 or 4 for the `quanto` backend and one of [1, 2, 3, 4, 8] for the `HQQ` backend. Defaults to 2.
        axis_key (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the key tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        axis_value (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the value tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        q_group_size (`Optional[int]`, *optional*, defaults to 64):
            Size of the quantization group, should be a divisor of the model's hidden dimension.
            Defaults to 64.
        residual_length (`Optional[int]`, *optional*, defaults to 128):
            Length of the residual cache which will always be stored in original presicion.
            Defaults to 128.
        compute_dtype (`ms.dtype`, *optional*, defaults to `ms.float16`):
            The defualt dtype used for computations in the model. Keys and Values will be cast to this dtype after dequantization.
    """

    def __init__(
        self,
        backend: str = "quanto",
        nbits: Optional[int] = 4,
        axis_key: Optional[int] = 0,
        axis_value: Optional[int] = 0,
        q_group_size: Optional[int] = 64,
        residual_length: Optional[int] = 128,
        compute_dtype = ms.float16,
    ):
        self.backend = backend
        self.nbits = nbits
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size
        self.residual_length = residual_length
        self.compute_dtype = compute_dtype

    def validate(self):
        """Validates if the arguments passed are correct"""

        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        # Check that the values are reasonable in general (nbits, axis)
        # Later in QuantizedCache init we check if they are supported for that particular backend
        if self.nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="nbits",
                    correct_value="2 or 4 or 8",
                    found_value=self.nbits,
                ),
            )
        if self.q_group_size <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="q_group_size",
                    correct_value="a positive integer",
                    found_value=self.q_group_size,
                ),
            )
        if self.residual_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="residual_length",
                    correct_value="a positive integer",
                    found_value=self.residual_length,
                ),
            )

        if self.axis_key not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_key",
                    correct_value="`1` or `0`, `-1`",
                    found_value=self.axis_key,
                ),
            )

        if self.axis_value not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_value",
                    correct_value="`1` or `0` or `-1`",
                    found_value=self.axis_value,
                ),
            )


@dataclass
class StaticCacheConfig(CacheConfig):
    """
    Configuration class for static cache settings.
    """

    cache_implementation = "static"

    def __init__(self, batch_size: int, max_cache_len: int):
        self.batch_size = batch_size
        self.max_cache_len = max_cache_len

    def validate(self):
        """Validates if the arguments passed are correct"""

        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )

        if self.batch_size <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="batch_size",
                    correct_value="> 0",
                    found_value=self.batch_size,
                ),
            )

        if self.max_cache_len <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="max_cache_len",
                    correct_value="> 0",
                    found_value=self.max_cache_len,
                ),
            )


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
        >>> from mindspore import Tensor

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = DynamicCache()
        >>> outputs = model(Tensor(inputs.input_ids), past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        DynamicCache()
        ```
    """

    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[ms.Tensor] = []
        self.value_cache: List[ms.Tensor] = []

    def __getitem__(self, layer_idx: int) -> List[Tuple[ms.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`ms.Tensor`):
                The new key states to cache.
            value_states (`ms.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append([])
                self.value_cache.append([])
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif len(self.key_cache[layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = ops.cat([self.key_cache[layer_idx], key_states], axis=-2)
            self.value_cache[layer_idx] = ops.cat([self.value_cache[layer_idx], value_states], axis=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[ms.Tensor], Tuple[ms.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[ms.Tensor]]] = None, num_hidden_layers: int = None
    ) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            if self.key_cache[idx] != []:
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def batch_split(
        self, full_batch_size: int, split_size: int, num_hidden_layers: int = None
    ) -> List["DynamicCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = DynamicCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            out.append(current_split)
        return out

    @classmethod
    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def from_batch_splits(cls, splits: List["DynamicCache"], num_hidden_layers: int = None) -> "DynamicCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls()
        for idx in range(len(splits[0])):
            key_cache = [current.key_cache[idx] for current in splits if current.key_cache[idx] != []]
            value_cache = [current.key_cache[idx] for current in splits if current.key_cache[idx] != []]
            if key_cache != []:
                layer_keys = ops.cat(key_cache, axis=0)
                layer_values = ops.cat(value_cache, axis=0)
                cache.update(layer_keys, layer_values, idx)
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: ms.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]


class OffloadedCache(DynamicCache):
    """
    A drop-in replacement for DynamicCache that conserves GPU memory at the expense of more CPU memory.
    Useful for generating from models with very long context.

    In addition to the default CUDA stream, where all forward() computations happen,
    this class uses another stream, the prefetch stream, which it creates itself.
    Since scheduling of operations on separate streams happens independently, this class uses
    the prefetch stream to asynchronously prefetch the KV cache of layer k+1 when layer k is executing.
    The movement of the layer k-1 cache to the CPU is handled by the default stream as a simple way to
    ensure the eviction is scheduled after all computations on that cache are finished.
    """

    def __init__(self) -> None:
        raise RuntimeError("OffloadedCache can only be used with Ascend")


class QuantizedCache(DynamicCache):
    """
    A quantizer cache similar to what is described in the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://arxiv.org/abs/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for Key and Value cache by applying quantization.

    The cache has two types of storage, one for original precision and one for the quantized cache. A `residual length` is set as a maximum capacity for the
    original precision cache. When the length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache. The
    quantization is done per-channel with a set `q_group_size` for both Keys and Values, in contrast to what was described in the paper.

    It stores Keys and Values a list of quantized tensors (tuples in case we need to store metadata), one for each layer. Additionally, it stores the Key and
    Value in original precision states as a list of tensors, one for each layer. The size of each tensor
    is `[batch_size, num_heads, seq_len - residual_length, head_dim]`
    """

    def __init__(self, cache_config: QuantizedCacheConfig) -> None:
        super().__init__()
        self._quantized_key_cache: List[ms.Tensor] = []
        self._quantized_value_cache: List[ms.Tensor] = []

        self.nbits = cache_config.nbits
        self.residual_length = cache_config.residual_length
        self.q_group_size = cache_config.q_group_size
        self.axis_key = cache_config.axis_key
        self.axis_value = cache_config.axis_value
        self.compute_dtype = cache_config.compute_dtype

        super().__init__()

    def update(
        self,
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) < layer_idx:
            raise ValueError("QuantizedCache does not support model usage where layers are skipped. Use DynamicCache.")
        elif len(self.key_cache) == layer_idx:
            self._quantized_key_cache.append(self._quantize(key_states.contiguous(), axis=self.axis_key))
            self._quantized_value_cache.append(self._quantize(value_states.contiguous(), axis=self.axis_value))
            self.key_cache.append(ops.zeros(0, dtype=key_states.dtype))
            self.value_cache.append(ops.zeros(0, dtype=key_states.dtype))
            keys_to_return, values_to_return = key_states, value_states
        else:
            dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
            dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
            keys_to_return = [dequant_key, self.key_cache[layer_idx], key_states]
            values_to_return = [dequant_value, self.value_cache[layer_idx], value_states]

            keys_to_return = ops.cat(keys_to_return, axis=-2)
            values_to_return = ops.cat(values_to_return, axis=-2)
            if (
                self.key_cache[layer_idx].ndim == 4
                and self.key_cache[layer_idx].shape[-2] + 1 >= self.residual_length
            ):
                self._quantized_key_cache[layer_idx] = self._quantize(keys_to_return.contiguous(), axis=self.axis_key)
                self._quantized_value_cache[layer_idx] = self._quantize(
                    values_to_return.contiguous(), axis=self.axis_value
                )
                self.key_cache[layer_idx] = ops.zeros(0, dtype=key_states.dtype)
                self.value_cache[layer_idx] = ops.zeros(0, dtype=key_states.dtype)
            else:
                self.key_cache[layer_idx] = ops.cat([self.key_cache[layer_idx], key_states], axis=-2)
                self.value_cache[layer_idx] = ops.cat([self.value_cache[layer_idx], value_states], axis=-2)

        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and rely on `_seen_tokens` which is
        # updated every "layer_idx" == 0, this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def _quantize(self, tensor, axis):
        """Quantizes a key/value using a defined quantization method."""
        raise NotImplementedError("Make sure to implement `_quantize` in a subclass.")

    def _dequantize(self, q_tensor):
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""
        raise NotImplementedError("Make sure to implement `_dequantize` in a subclass.")


class QuantoQuantizedCache(QuantizedCache):
    """
    Quantized Cache class that uses `quanto` as a backend to perform quantization. Current implementation supports `int2` and `int4` dtypes only.

    Parameters:
        cache_config (`QuantizedCacheConfig`):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.

    Example:

        ```python
        >>> # Run pip install quanto first if you don't have it yet
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoQuantizedCache, QuantizedCacheConfig
        >>> from mindspore import Tensor

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> cache_config = QuantizedCacheConfig(nbits=4)
        >>> past_key_values = QuantoQuantizedCache(cache_config=cache_config)
        >>> outputs = model(Tensor(inputs.input_ids), past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        QuantoQuantizedCache()
        ```
    """

    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)

        # if is_optimum_quanto_available():
        #     from optimum.quanto import MaxOptimizer, qint2, qint4
        if is_quanto_available():
            logger.warning_once(
                "Importing from quanto will be deprecated in v4.47. Please install optimum-quanto instead `pip install optimum-quanto`"
            )
            quanto_version = version.parse(importlib.metadata.version("quanto"))
            if quanto_version < version.parse("0.2.0"):
                raise ImportError(
                    f"You need quanto package version to be greater or equal than 0.2.0 to use `QuantoQuantizedCache`. Detected version {quanto_version}. "
                    f"Since quanto will be deprecated, please install optimum-quanto instead with `pip install -U optimum-quanto`"
                )
            from quanto import MaxOptimizer, qint2, qint4

        if self.nbits not in [2, 4]:
            raise ValueError(f"`nbits` for `quanto` backend has to be one of [`2`, `4`] but got {self.nbits}")

        if self.axis_key not in [0, -1]:
            raise ValueError(f"`axis_key` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_key}")

        if self.axis_value not in [0, -1]:
            raise ValueError(
                f"`axis_value` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_value}"
            )

        self.qtype = qint4 if self.nbits == 4 else qint2
        self.optimizer = MaxOptimizer()  # hardcode as it's the only one for per-channel quantization

    def _quantize(self, tensor, axis):
        # We have two different API since in optimum-quanto, we don't use AffineQuantizer anymore
        # if is_optimum_quanto_available():
        #     from optimum.quanto import quantize_weight

        #     scale, zeropoint = self.optimizer(tensor, self.qtype, axis, self.q_group_size)
        #     qtensor = quantize_weight(tensor, self.qtype, axis, scale, zeropoint, self.q_group_size)
        #     return qtensor
        # el
        if is_quanto_available():
            logger.warning_once(
                "Importing from quanto will be deprecated in v4.47. Please install optimum-quanto instead `pip install optimum-quanto`"
            )
            from quanto import AffineQuantizer

            scale, zeropoint = self.optimizer(tensor, self.qtype.bits, axis, self.q_group_size)
            qtensor = AffineQuantizer.apply(tensor, self.qtype, axis, self.q_group_size, scale, zeropoint)

        return qtensor

    def _dequantize(self, qtensor):
        return qtensor.dequantize()


class HQQQuantizedCache(QuantizedCache):
    """
    Quantized Cache class that uses `HQQ` as a backend to perform quantization. Current implementation supports `int2`, `int4`, `int8` dtypes.

    Parameters:
        cache_config (`QuantizedCacheConfig`):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.
    """

    def __init__(self, cache_config: CacheConfig) -> None:
        raise RuntimeError("Do not support HQQQuantizer")


class SinkCache(Cache):
    """
    A cache that as described in the [Attention Sinks paper](https://arxiv.org/abs/2309.17453). It allows the model to
    generate beyond the length of its context window, without losing fluency in the conversation. As it discards past
    tokens, the model will lose the ability to generate tokens that depend on the context that was discarded.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Parameters:
        window_length (`int`):
            The length of the context window.
        num_sink_tokens (`int`):
            The number of sink tokens. See the original paper for more information.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, SinkCache
        >>> from mindspore import Tensor

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="np")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = SinkCache(window_length=256, num_sink_tokens=4)
        >>> outputs = model(Tensor(inputs.input_ids), past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        SinkCache()
        ```
    """

    is_sliding = True

    def __init__(self, window_length: int, num_sink_tokens: int) -> None:
        super().__init__()
        self.key_cache: List[ms.Tensor] = []
        self.value_cache: List[ms.Tensor] = []
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self.cos_sin_rerotation_cache = {}
        self._cos_cache = None
        self._sin_cache = None
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return ops.cat((-x2, x1), axis=-1)

    def _apply_key_rotary_pos_emb(
        self, key_states: ms.Tensor, cos: ms.Tensor, sin: ms.Tensor
    ) -> ms.Tensor:
        rotated_key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return rotated_key_states

    def _get_rerotation_cos_sin(
        self, key_states: ms.Tensor, cos: ms.Tensor, sin: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        if key_states.shape[-2] not in self.cos_sin_rerotation_cache:
            # Upcast to float32 temporarily for better accuracy
            cos = cos.to(ms.float32)
            sin = sin.to(ms.float32)

            # Compute the cos and sin required for back- and forward-rotating to one position earlier in the sequence
            original_cos = cos[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_cos = cos[self.num_sink_tokens : -key_states.shape[-2]]
            original_sin = sin[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_sin = sin[self.num_sink_tokens : -key_states.shape[-2]]
            rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
            rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin

            self.cos_sin_rerotation_cache[key_states.shape[-2]] = (
                rerotation_cos.to(key_states.dtype).unsqueeze(0),
                rerotation_sin.to(key_states.dtype).unsqueeze(0),
            )
        return self.cos_sin_rerotation_cache[key_states.shape[-2]]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object, in case of SinkCache it is the window length."""
        return self.window_length

    def update(
        self,
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`ms.Tensor`):
                The new key states to cache.
            value_states (`ms.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The following arguments can be used in `SinkCache`: `sin`,
                `cos` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
                rotation as the tokens are shifted.

        Return:
            A tuple containing the updated key and value states.
        """
        # Optional kwargs for `SinkCache` -- needed on models using RoPE. `partial_rotation_size` is used on models
        # with partially rotated position embeddings, like Phi or Persimmon.
        sin = cache_kwargs.get("sin")
        cos = cache_kwargs.get("cos")
        partial_rotation_size = cache_kwargs.get("partial_rotation_size")
        using_rope = cos is not None and sin is not None

        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the sin/cos cache, which holds sin/cos values for all possible positions
        if using_rope and layer_idx == 0:
            # BC: some models still pass `sin`/`cos` with 2 dims. In those models, they are the full sin/cos. Remove
            # after all RoPE models have a llama-like cache utilization.
            if cos.dim() == 2:
                self._cos_cache = cos
                self._sin_cache = sin
            else:
                if self._cos_cache is None:
                    self._cos_cache = cos[0, ...]
                    self._sin_cache = sin[0, ...]
                elif self._cos_cache.shape[0] < self.window_length:
                    self._cos_cache = ops.cat([self._cos_cache, cos[0, ...]], axis=0)
                    self._sin_cache = ops.cat([self._sin_cache, sin[0, ...]], axis=0)

        # [bsz, num_heads, seq_len, head_dim]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        elif key_states.shape[-2] + self.get_seq_length(layer_idx) < self.window_length:
            # Growing cache
            self.key_cache[layer_idx] = ops.cat([self.key_cache[layer_idx], key_states], axis=-2)
            self.value_cache[layer_idx] = ops.cat([self.value_cache[layer_idx], value_states], axis=-2)

        else:
            # Shifting cache
            keys_to_keep = self.key_cache[layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + key_states.shape[-2] :
            ]

            # On RoPE models, we need to recompute the Key rotation as the tokens are shifted
            if using_rope:
                rerotation_cos, rerotation_sin = self._get_rerotation_cos_sin(
                    key_states, self._cos_cache[: self.window_length], self._sin_cache[: self.window_length]
                )
                if partial_rotation_size is not None:
                    keys_to_keep, keys_pass = (
                        keys_to_keep[..., :partial_rotation_size],
                        keys_to_keep[..., partial_rotation_size:],
                    )
                keys_to_keep = self._apply_key_rotary_pos_emb(keys_to_keep, rerotation_cos, rerotation_sin)
                if partial_rotation_size is not None:
                    keys_to_keep = ops.cat((keys_to_keep, keys_pass), axis=-1)

            # Concatenate sink tokens, shifted & rotated tokens (if needed), and new tokens
            sink_keys = self.key_cache[layer_idx][:, :, : self.num_sink_tokens]
            self.key_cache[layer_idx] = ops.cat([sink_keys, keys_to_keep, key_states], axis=-2)

            sink_values = self.value_cache[layer_idx][:, :, : self.num_sink_tokens]
            values_to_keep = self.value_cache[layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + value_states.shape[-2] :
            ]
            self.value_cache[layer_idx] = ops.cat([sink_values, values_to_keep, value_states], axis=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)` and `torch.export()`.

    Parameters:
        config (`PretrainedConfig`):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used. If you are manually setting the batch size, make sure to take into account the number of beams if you are running beam search
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map(`Dict[int, Union[str, torch.device, int]]]`, `optional`):
            Mapping between the layers and its device. This is required when you are manually initializing the cache and the model is splitted between differents gpus.
            You can know which layers mapped to which device by checking the associated device_map: `model.hf_device_map`.
    """

    # TODO (joao): remove `=None` in non-optional arguments in v4.46. Remove from `OBJECTS_TO_IGNORE` as well.
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int = None,
        max_cache_len: int = None,
        device = None,
        dtype = None,
        max_batch_size: Optional[int] = None,
        layer_device_map = None,
    ) -> None:
        raise RuntimeError("Do not support Static Cache.")
   


class SlidingWindowCache(StaticCache):
    """
    Sliding Window Cache class to be used with `torch.compile` for models like Mistral that support sliding window attention.
    Every time when we try to update the cache, we compute the `indices` based on `cache_position >= self.config.sliding_window - 1`,
    if true(which means the cache can not hold all the old key value states and new states together because of the sliding window constraint),
    we need to do a cycle shift based on `indices` to replace the oldest states by the new key value states passed in.

    The `to_shift` is only true once we are above sliding_window. Thus with `sliding_window==64`:

    indices = (slicing + to_shift[-1].int()-1) % self.config.sliding_window
    tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63,  0])

    We overwrite the cache using these, then we always write at cache_position (clamped to `sliding_window`)

    Parameters:
        config (`PretrainedConfig`):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map(`Dict[int, Union[str, torch.device, int]]]`, `optional`):
            Mapping between the layers and its device. This is required when you are manually initializing the cache and the model is splitted between differents gpus.
            You can know which layers mapped to which device by checking the associated device_map: `model.hf_device_map`.
    """

    is_sliding = True

    # TODO (joao): remove `=None` in non-optional arguments in v4.46. Remove from `OBJECTS_TO_IGNORE` as well.
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int = None,
        max_cache_len: int = None,
        device = None,
        dtype = None,
        max_batch_size: Optional[int] = None,
        layer_device_map = None,
    ) -> None:
        raise RuntimeError("Do not support Sliding Window Cache.")


class EncoderDecoderCache(Cache):
    """
    Base, abstract class for all encoder-decoder caches. Can be used to hold combinations of self-attention and
    cross-attention caches.

    Example:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForCausalLM, DynamicCache, EncoderDecoderCache
        >>> from mindspore import Tensor

        >>> model = AutoModelForCausalLM.from_pretrained("openai/whisper-small")
        >>> processor = AutoProcessor.from_pretrained("openai/whisper-small")

        >>> inputs = processor(audio=YOUR-AUDIO, return_tensors="pt")

        >>> # Prepare cache classes for encoder and decoder and pass it to model's forward
        >>> self_attention_cache = DynamicCache()
        >>> cross_attention_cache = DynamicCache()
        >>> past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)
        >>> outputs = model(Tensor(inputs.input_ids), past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        EncoderDecoderCache()
        ```

    """

    def __init__(self, self_attention_cache: Cache, cross_attention_cache: Cache):
        super().__init__()
        self.self_attention_cache = self_attention_cache
        self.cross_attention_cache = cross_attention_cache

        self.is_updated = {}
        for layer_idx in range(len(cross_attention_cache.key_cache)):
            self.is_updated[layer_idx] = bool(cross_attention_cache.get_seq_length(layer_idx) > 0)

    def __getitem__(self, layer_idx: int) -> List[Tuple[ms.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (
                self.self_attention_cache.key_cache[layer_idx],
                self.self_attention_cache.value_cache[layer_idx],
                self.cross_attention_cache.key_cache[layer_idx],
                self.cross_attention_cache.value_cache[layer_idx],
            )
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.self_attention_cache)

    def to_legacy_cache(self) -> Tuple[Tuple[ms.Tensor], Tuple[ms.Tensor]]:
        """Converts the `EncoderDecoderCache` instance into  its equivalent in the legacy cache format."""
        legacy_cache = ()
        if len(self.cross_attention_cache) > 0:
            for self_attn, cross_attn in zip(
                self.self_attention_cache.to_legacy_cache(), self.cross_attention_cache.to_legacy_cache()
            ):
                legacy_cache += (self_attn + cross_attn,)
        else:
            legacy_cache = self.self_attention_cache.to_legacy_cache()
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[ms.Tensor]]] = None
    ) -> "EncoderDecoderCache":
        """Converts a cache in the legacy cache format into an equivalent `EncoderDecoderCache`."""
        cache = cls(
            self_attention_cache=DynamicCache(),
            cross_attention_cache=DynamicCache(),
        )
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx][:2]
                cache.self_attention_cache.update(key_states, value_states, layer_idx)
                if len(past_key_values[layer_idx]) > 2:
                    key_states, value_states = past_key_values[layer_idx][2:]
                    cache.cross_attention_cache.update(key_states, value_states, layer_idx)
                    cache.is_updated[layer_idx] = True
        return cache

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # check if empty list because in case of static cache it will be a tensors and we can't check `if not ms.Tensor`
        if self.self_attention_cache.key_cache == []:
            return 0
        if len(self.self_attention_cache.key_cache) > 1 and self.self_attention_cache.key_cache[layer_idx] == []:
            return 0
        return (self.self_attention_cache.key_cache[layer_idx][0, 0].any(axis=-1)).sum()

    def reset(self):
        if hasattr(self.self_attention_cache, "reset"):
            self.self_attention_cache.reset()
        if hasattr(self.cross_attention_cache, "reset"):
            self.cross_attention_cache.reset()
        elif not hasattr(self.self_attention_cache, "reset") and not hasattr(self.cross_attention_cache, "reset"):
            raise ValueError(
                "Neither self nor cross-attention cache have valid `.reset()` methods. `.reset()` should "
                "only be called on compatible cache classes, such as `StaticCache` or `SlidingWindowCache`. "
                f"Got {self.self_attention_cache.__str__()} for the self attention cache and "
                f"{self.cross_attention_cache.__str__()} for the cross attention cache."
            )
        for layer_idx in self.is_updated:
            self.is_updated[layer_idx] = False

    def reorder_cache(self, beam_idx: ms.Tensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        self.self_attention_cache.reorder_cache(beam_idx)
        self.cross_attention_cache.reorder_cache(beam_idx)

    def check_dynamic_cache(self, method: str):
        if not (
            isinstance(self.self_attention_cache, DynamicCache)
            and isinstance(self.cross_attention_cache, DynamicCache)
        ):
            raise ValueError(
                f"`{method}` is only defined for dynamic cache, got {self.self_attention_cache.__str__()} for the self "
                f"attention cache and {self.cross_attention_cache.__str__()} for the cross attention cache."
            )

    # TODO(gante, sanchit-gandhi): move following functionality into `.generate`
    def crop(self, maximum_length: int):
        """Crop the past key values up to a new `maximum_length` in terms of tokens. `maximum_length` can also be
        negative to remove `maximum_length` tokens. This is used in assisted decoding and contrastive search."""
        self.check_dynamic_cache(self.crop.__name__)
        self.self_attention_cache.crop(maximum_length)

    def batch_split(self, full_batch_size: int, split_size: int) -> "List[EncoderDecoderCache]":
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        self.check_dynamic_cache(self.batch_split.__name__)
        self_attention_cache = self.self_attention_cache.batch_split(full_batch_size, split_size)
        cross_attention_cache = self.cross_attention_cache.batch_split(full_batch_size, split_size)

        out = []
        for self_attn, cross_attn in zip(self_attention_cache, cross_attention_cache):
            out.append(EncoderDecoderCache(self_attn, cross_attn))
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["EncoderDecoderCache"]) -> "EncoderDecoderCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        self_attention_cache = DynamicCache()
        cross_attention_cache = DynamicCache()
        for idx in range(len(splits[0])):
            layer_keys = ops.cat([current.self_attention_cache.key_cache[idx] for current in splits], axis=0)
            layer_values = ops.cat([current.self_attention_cache.value_cache[idx] for current in splits], axis=0)
            self_attention_cache.update(layer_keys, layer_values, idx)

            layer_keys = ops.cat([current.cross_attention_cache.key_cache[idx] for current in splits], axis=0)
            layer_values = ops.cat([current.cross_attention_cache.value_cache[idx] for current in splits], axis=0)
            cross_attention_cache.update(layer_keys, layer_values, idx)
        return cls(self_attention_cache, cross_attention_cache)

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        self.check_dynamic_cache(self.batch_repeat_interleave.__name__)
        self.self_attention_cache.batch_repeat_interleave(repeats)
        self.cross_attention_cache.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: ms.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        self.check_dynamic_cache(self.batch_select_indices.__name__)
        self.self_attention_cache.batch_select_indices(indices)
        self.cross_attention_cache.batch_select_indices(indices)


class HybridCache(Cache):
    """
    Hybrid Cache class to be used with `torch.compile` for Gemma2 models that alternate between a local sliding window attention
    and global attention in every other layer. Under the hood, Hybrid Cache leverages ["SlidingWindowCache"] for sliding window attention
    and ["StaticCache"] for global attention. For more information, see the documentation of each subcomponeent cache class.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`, *optional*, defaults to `"cpu"`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (torch.dtype, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map(`Dict[int, Union[str, torch.device, int]]]`, `optional`):
            Mapping between the layers and its device. This is required when you are manually initializing the cache and the model is splitted between differents gpus.
            You can know which layers mapped to which device by checking the associated device_map: `model.hf_device_map`.

    """

    # TODO (joao): remove `=None` in non-optional arguments in v4.46. Remove from `OBJECTS_TO_IGNORE` as well.
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int = None,
        max_cache_len: int = None,
        device = None,
        dtype = None,
        max_batch_size = None,
        layer_device_map = None
    ) -> None:
        raise RuntimeError("Do not support Hybrid Cache.")


class MambaCache:
    """
    Cache for mamba model which does not have attention mechanism and key value states.

    Arguments:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The default `dtype` to use when initializing the layer.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. Should be the same as the layer.

    Attributes:
        dtype: (`torch.dtype`):
            The default `dtype` used to initializing the cache.
        intermediate_size: (`int`):
            Model's intermediate_size taken from config.
        ssm_state_size: (`int`):
            Model's state_size taken from config.
        conv_kernel_size: (`int`):
            Model's convolution kernel size taken from config
        conv_states: (`torch.Tensor`):
            A tensor of shape `[layer_idx, batch_size, intermediate_size, conv_kernel_size]` that holds convolutional states.
        ssm_states: (`torch.Tensor`):
            A tensor of shape `[layer_idx, batch_size, intermediate_size, ssm_state_size]` that holds ssm states

    """

    # TODO (joao): remove `=None` in non-optional arguments in v4.46. Remove from `OBJECTS_TO_IGNORE` as well.
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int = None,
        dtype = None,
        device = None,
        max_batch_size: Optional[int] = None,
    ):
       raise RuntimeError("Do not support Mamba Cache.")


class OffloadedStaticCache(StaticCache):
    """
    Static cache class to be used with `torch.compile(model)` that offloads to the CPU or
    another device.

    Args:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize
            the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`Union[str, torch.device]`):
            The device on which the cache should be initialized. Should be the same as the
            layer device.
        dtype (`torch.dtype`, *optional*):
            The default `dtype` to use when initializing the cache.
        offload_device (`Union[str, torch.device]`, *optional*, defaults to `cpu`):
            The device to offload to. Defaults to CPU.
    
    Attributes:
        key_cache (`List[torch.Tensor]`):
            Off-loaded key cache tensors. First one will be on device, where-as the others are
            off-loaded.
        value_cache (`List[torch.Tensor]`):
            Off-loaded value cache tensors. First one will be on device, where-as the others are
            off-loaded.
        max_batch_size (`int`):
            The maximum batch size with which this cache can be used.
        max_cache_len (`int`):
            The maximum sequence length with which this cache can be used.
        device (`torch.device`):
            The device on which the cache is used.
        offload_device (`torch.device`):
            The device used to offload to.
        dtype (`torch.dtype`):
            The `dtype` used to initializing the cache.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: Optional[int],
        device,
        dtype,
        offload_device,
    ) -> None:
        raise RuntimeError("Do not support Offloaded Static Cache.")