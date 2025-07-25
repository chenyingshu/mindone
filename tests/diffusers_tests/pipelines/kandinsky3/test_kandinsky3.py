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

import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers.utils.testing_utils import load_numpy_from_local_file, slow

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
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
class Kandinsky3PipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "unet",
            "diffusers.models.unets.unet_kandinsky3.Kandinsky3UNet",
            "mindone.diffusers.models.unets.unet_kandinsky3.Kandinsky3UNet",
            dict(
                in_channels=4,
                time_embedding_dim=4,
                groups=2,
                attention_head_dim=4,
                layers_per_block=3,
                block_out_channels=(32, 64),
                cross_attention_dim=4,
                encoder_hid_dim=32,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            dict(
                beta_start=0.00085,
                beta_end=0.012,
                steps_offset=1,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                thresholding=False,
            ),
        ],
        [
            "movq",
            "diffusers.models.autoencoders.vq_model.VQModel",
            "mindone.diffusers.models.autoencoders.vq_model.VQModel",
            {
                "block_out_channels": [32, 64],
                "down_block_types": ["DownEncoderBlock2D", "AttnDownEncoderBlock2D"],
                "in_channels": 3,
                "latent_channels": 4,
                "layers_per_block": 1,
                "norm_num_groups": 8,
                "norm_type": "spatial",
                "num_vq_embeddings": 12,
                "out_channels": 3,
                "up_block_types": [
                    "AttnUpDecoderBlock2D",
                    "UpDecoderBlock2D",
                ],
                "vq_embed_dim": 4,
            },
        ],
        [
            "text_encoder",
            "transformers.models.t5.modeling_t5.T5EncoderModel",
            "mindone.transformers.models.t5.modeling_t5.T5EncoderModel",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5",
                revision="refs/pr/1",
            ),
        ],
        [
            "tokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5",
            ),
        ],
    ]

    def get_dummy_components(self, time_cond_proj_dim=None):
        components = {
            key: None
            for key in [
                "unet",
                "scheduler",
                "movq",
                "text_encoder",
                "tokenizer",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
            "width": 16,
            "height": 16,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_kandinsky3(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.kandinsky3.Kandinsky3Pipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.kandinsky3.Kandinsky3Pipeline")

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
class Kandinsky3PipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_kandinskyV3(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.kandinsky3.Kandinsky3Pipeline")
        pipe = pipe_cls.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", mindspore_dtype=ms_dtype)
        pipe.set_progress_bar_config(disable=None)

        prompt = (
            "A photograph of the inside of a subway train. There are raccoons sitting on the seats. One of them "
            "is reading a newspaper. The window shows the city in the background."
        )

        torch.manual_seed(0)
        image = pipe(prompt, height=512, width=512, num_inference_steps=5)[0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"t2i_{dtype}.npy",
            subfolder="kandinsky3",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
