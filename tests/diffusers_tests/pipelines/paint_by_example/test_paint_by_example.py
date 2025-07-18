# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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
import torch
from ddt import data, ddt, unpack
from PIL import Image
from transformers import CLIPVisionConfig

import mindspore as ms

from mindone.diffusers import PaintByExamplePipeline
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
class PaintByExamplePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            dict(
                block_out_channels=(32, 64),
                layers_per_block=2,
                sample_size=32,
                in_channels=9,
                out_channels=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                cross_attention_dim=32,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_pndm.PNDMScheduler",
            "mindone.diffusers.schedulers.scheduling_pndm.PNDMScheduler",
            dict(
                skip_prk_steps=True,
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
            "image_encoder",
            "diffusers.pipelines.paint_by_example.PaintByExampleImageEncoder",
            "mindone.diffusers.pipelines.paint_by_example.PaintByExampleImageEncoder",
            dict(
                config=CLIPVisionConfig(
                    hidden_size=32,
                    projection_dim=32,
                    intermediate_size=37,
                    layer_norm_eps=1e-05,
                    num_attention_heads=4,
                    num_hidden_layers=5,
                    image_size=32,
                    patch_size=4,
                ),
                proj_size=32,
            ),
        ],
        [
            "feature_extractor",
            "transformers.models.clip.image_processing_clip.CLIPImageProcessor",
            "transformers.models.clip.image_processing_clip.CLIPImageProcessor",
            dict(
                crop_size=32,
                size=32,
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
                "image_encoder",
                "safety_checker",
                "feature_extractor",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((64, 64))
        example_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((32, 32))

        inputs = {
            "example_image": example_image,
            "image": init_image,
            "mask_image": mask_image,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_paint_by_example_inpaint(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.paint_by_example.PaintByExamplePipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.paint_by_example.PaintByExamplePipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class PaintByExamplePipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_paint_by_example(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe = PaintByExamplePipeline.from_pretrained(
            "Fantasy-Studio/Paint-by-Example", revision="refs/pr/3", mindspore_dtype=ms_dtype
        )
        pipe.set_progress_bar_config(disable=None)

        init_image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "dog_in_bucket.png",
            subfolder="paint_by_example",
        )
        mask_image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "mask.png",
            subfolder="paint_by_example",
        )
        example_image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "panda.jpg",
            subfolder="paint_by_example",
        )

        torch.manual_seed(0)
        image = pipe(
            image=init_image,
            mask_image=mask_image,
            example_image=example_image,
            guidance_scale=5.0,
            num_inference_steps=50,
        )[0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"paint_by_example_{dtype}.npy",
            subfolder="paint_by_example",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
