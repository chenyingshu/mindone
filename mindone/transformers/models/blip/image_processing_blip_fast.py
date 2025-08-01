# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
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
"""Fast Image processor class for BLIP."""

from transformers.utils import add_start_docstrings

from ...image_processing_utils_fast import BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BaseImageProcessorFast
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling


@add_start_docstrings(
    "Constructs a fast BLIP image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class BlipImageProcessorFast(BaseImageProcessorFast):
    # To be checked against the slow image processor
    # None values left after checking can be removed
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True


__all__ = ["BlipImageProcessorFast"]
