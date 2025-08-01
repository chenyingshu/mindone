"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/text_to_video_synthesis/pipeline_output.py."""

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL

import mindspore as ms

from ...utils import BaseOutput


@dataclass
class TextToVideoSDPipelineOutput(BaseOutput):
    """
     Output class for text-to-video pipelines.

    Args:
         frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
             List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
             denoised
     PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
    `(batch_size, num_frames, channels, height, width)`
    """

    frames: Union[ms.Tensor, np.ndarray, List[List[PIL.Image.Image]]]
