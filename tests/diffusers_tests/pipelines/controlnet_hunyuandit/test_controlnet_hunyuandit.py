# coding=utf-8
# Copyright 2025 HuggingFace Inc and Tencent Hunyuan Team.
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
from diffusers.utils.torch_utils import randn_tensor

import mindspore as ms

from mindone.diffusers import (
    HunyuanDiT2DControlNetModel,
    HunyuanDiT2DMultiControlNetModel,
    HunyuanDiTControlNetPipeline,
)
from mindone.diffusers.utils.testing_utils import load_downloaded_image_from_hf_hub, load_numpy_from_local_file, slow

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
class HunyuanDiTControlNetPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.hunyuan_transformer_2d.HunyuanDiT2DModel",
            "mindone.diffusers.models.transformers.hunyuan_transformer_2d.HunyuanDiT2DModel",
            dict(
                sample_size=16,
                num_layers=4,
                patch_size=2,
                attention_head_dim=8,
                num_attention_heads=3,
                in_channels=4,
                cross_attention_dim=32,
                cross_attention_dim_t5=32,
                pooled_projection_dim=16,
                hidden_size=24,
                activation_fn="gelu-approximate",
            ),
        ],
        [
            "controlnet",
            "diffusers.models.controlnets.controlnet_hunyuan.HunyuanDiT2DControlNetModel",
            "mindone.diffusers.models.controlnets.controlnet_hunyuan.HunyuanDiT2DControlNetModel",
            dict(
                sample_size=16,
                transformer_num_layers=4,
                patch_size=2,
                attention_head_dim=8,
                num_attention_heads=3,
                in_channels=4,
                cross_attention_dim=32,
                cross_attention_dim_t5=32,
                pooled_projection_dim=16,
                hidden_size=24,
                activation_fn="gelu-approximate",
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            dict(),
        ],
        [
            "text_encoder",
            "transformers.models.bert.modeling_bert.BertModel",
            "mindone.transformers.models.bert.modeling_bert.BertModel",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-BertModel",
                revision="refs/pr/1",
            ),
        ],
        [
            "tokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-BertModel",
            ),
        ],
        [
            "text_encoder_2",
            "transformers.models.t5.modeling_t5.T5EncoderModel",
            "mindone.transformers.models.t5.modeling_t5.T5EncoderModel",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5",
                revision="refs/pr/1",
            ),
        ],
        [
            "tokenizer_2",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5",
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "transformer",
                "vae",
                "scheduler",
                "text_encoder",
                "tokenizer",
                "text_encoder_2",
                "tokenizer_2",
                "safety_checker",
                "feature_extractor",
                "controlnet",
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)
        pt_components["transformer"] = pt_components["transformer"].eval()
        ms_components["transformer"] = ms_components["transformer"].set_train(False)
        pt_components["vae"] = pt_components["vae"].eval()
        ms_components["vae"] = ms_components["vae"].set_train(False)

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        generator = torch.manual_seed(seed)

        control_image = randn_tensor(
            (1, 3, 16, 16),
            generator=generator,
            device=torch.device("cpu"),
            dtype=torch.float16,
        )
        pt_control_image = control_image
        ms_control_image = ms.tensor(control_image.numpy())

        controlnet_conditioning_scale = 0.5

        pt_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "control_image": pt_control_image,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
        }

        ms_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "control_image": ms_control_image,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
        }

        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_controlnet_hunyuandit(self, mode, dtype):
        ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module(
            "diffusers.pipelines.controlnet_hunyuandit.pipeline_hunyuandit_controlnet.HunyuanDiTControlNetPipeline"
        )
        ms_pipe_cls = get_module(
            "mindone.diffusers.pipelines.controlnet_hunyuandit.pipeline_hunyuandit_controlnet.HunyuanDiTControlNetPipeline"
        )

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_inputs, ms_inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**pt_inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**ms_inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class HunyuanDiTControlNetPipelineSlowTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_depth(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        controlnet = HunyuanDiT2DControlNetModel.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Depth", mindspore_dtype=ms_dtype
        )
        pipe = HunyuanDiTControlNetPipeline.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers", controlnet=controlnet, mindspore_dtype=ms_dtype
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = (
            "In the dense forest, a black and white panda sits quietly in green trees and red flowers, "
            "surrounded by mountains, rivers, and the ocean. The background is the forest in a bright environment."
        )
        n_prompt = ""
        control_image = load_downloaded_image_from_hf_hub(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Depth",
            "depth.jpg",
            subfolder=None,
            repo_type="model",
        )

        torch.manual_seed(0)
        output = pipe(
            prompt,
            negative_prompt=n_prompt,
            control_image=control_image,
            controlnet_conditioning_scale=0.5,
            guidance_scale=5.0,
            num_inference_steps=2,
        )
        image = output[0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"controlnet_hunyuandit_depth_{dtype}.npy",
            subfolder="controlnet_hunyuandit",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL

    @data(*test_cases)
    @unpack
    def test_multi_controlnet(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        controlnet = HunyuanDiT2DControlNetModel.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny", mindspore_dtype=ms_dtype
        )
        controlnet = HunyuanDiT2DMultiControlNetModel([controlnet, controlnet])

        pipe = HunyuanDiTControlNetPipeline.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers", controlnet=controlnet, mindspore_dtype=ms_dtype
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = (
            "At night, an ancient Chinese-style lion statue stands in front of the hotel, its eyes gleaming "
            "as if guarding the building. The background is the hotel entrance at night, with a close-up, "
            "eye-level, and centered composition. This photo presents a realistic photographic style, embodies "
            "Chinese sculpture culture, and reveals a mysterious atmosphere."
        )
        n_prompt = ""
        control_image = load_downloaded_image_from_hf_hub(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny",
            "canny.jpg",
            subfolder=None,
            repo_type="model",
        )

        torch.manual_seed(0)
        output = pipe(
            prompt,
            negative_prompt=n_prompt,
            control_image=[control_image, control_image],
            controlnet_conditioning_scale=[0.25, 0.25],
            guidance_scale=5.0,
            num_inference_steps=2,
        )
        image = output[0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"controlnet_hunyuandit_multi_controlnet_{dtype}.npy",
            subfolder="controlnet_hunyuandit",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
