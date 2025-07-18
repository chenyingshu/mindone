# coding=utf-8
# Copyright 2023 HuggingFace Inc.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
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

import random
import unittest

import numpy as np
import pytest
import torch
from ddt import data, ddt, unpack
from PIL import Image
from transformers import CLIPTextConfig

import mindspore as ms

from mindone.diffusers import LEditsPPPipelineStableDiffusion
from mindone.diffusers.utils.testing_utils import load_downloaded_image_from_hf_hub, load_numpy_from_local_file, slow

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
    floats_tensor,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
]


@ddt
class LEditsPPPipelineStableDiffusionFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            dict(
                block_out_channels=(32, 64, 64),
                layers_per_block=2,
                sample_size=32,
                in_channels=4,
                out_channels=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
                cross_attention_dim=32,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler",
            "mindone.diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler",
            dict(
                algorithm_type="sde-dpmsolver++",
                solver_order=2,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(
                block_out_channels=[32, 64],
                in_channels=3,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
                latent_channels=4,
            ),
        ],
        [
            "text_encoder",
            "transformers.models.clip.modeling_clip.CLIPTextModel",
            "mindone.transformers.models.clip.modeling_clip.CLIPTextModel",
            dict(
                config=CLIPTextConfig(
                    bos_token_id=0,
                    eos_token_id=2,
                    hidden_size=32,
                    intermediate_size=37,
                    layer_norm_eps=1e-05,
                    num_attention_heads=4,
                    num_hidden_layers=5,
                    pad_token_id=1,
                    vocab_size=1000,
                ),
            ),
        ],
        [
            "tokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-clip",
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "unet",
                "scheduler",
                "vae",
                "text_encoder",
                "tokenizer",
                "safety_checker",
                "feature_extractor",
            ]
        }
        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self):
        inputs = {
            "editing_prompt": ["wearing glasses", "sunshine"],
            "reverse_editing_direction": [False, True],
            "edit_guidance_scale": [10.0, 5.0],
            "output_type": "np",
        }
        return inputs

    def get_dummy_inversion_inputs(self):
        images = floats_tensor((2, 3, 32, 32), rng=random.Random(0)).permute(0, 2, 3, 1)
        images = 255 * images
        image_1 = Image.fromarray(np.uint8(images[0])).convert("RGB")
        image_2 = Image.fromarray(np.uint8(images[1])).convert("RGB")

        inputs = {
            "image": [image_1, image_2],
            "source_prompt": "",
            "source_guidance_scale": 3.5,
            "num_inversion_steps": 20,
            "skip": 0.15,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_ledits_pp_sd(self, mode, dtype):
        if dtype == "float16":
            pytest.skip("Skipping this case since `reflection_pad2d` is not supported in torch")

        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module(
            "diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.LEditsPPPipelineStableDiffusion"
        )
        ms_pipe_cls = get_module(
            "mindone.diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.LEditsPPPipelineStableDiffusion"
        )

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        invert_inputs = self.get_dummy_inversion_inputs()
        invert_inputs["image"] = invert_inputs["image"][0]
        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        _ = pt_pipe.invert(**invert_inputs)
        torch.manual_seed(0)
        pt_image = pt_pipe(**inputs)

        torch.manual_seed(0)
        _ = ms_pipe.invert(**invert_inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)

        pt_image_slice = pt_image.images[0, -1, -3:, -3:]
        ms_image_slice = ms_image[0][0, -1, -3:, -3:]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class LEditsPPPipelineStableDiffusionSlowTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_ledits_pp_editing(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe = LEditsPPPipelineStableDiffusion.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None, mindspore_dtype=ms_dtype
        )
        pipe.set_progress_bar_config(disable=None)

        raw_image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "cat_6.png",
            subfolder="pix2pix",
        )
        raw_image = raw_image.convert("RGB").resize((512, 512))

        torch.manual_seed(0)
        _ = pipe.invert(image=raw_image)
        inputs = {
            "editing_prompt": ["cat", "dog"],
            "reverse_editing_direction": [True, False],
            "edit_guidance_scale": [5.0, 5.0],
            "edit_threshold": [0.8, 0.8],
        }
        torch.manual_seed(0)
        image = pipe(**inputs)[0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"sd_{dtype}.npy",
            subfolder="ledits_pp",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
