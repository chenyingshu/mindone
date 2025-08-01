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

# This model implementation is heavily based on:

import random
import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack
from PIL import Image
from transformers import CLIPTextConfig

import mindspore as ms

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
class ControlNetInpaintPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
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
            "controlnet",
            "diffusers.models.controlnets.controlnet.ControlNetModel",
            "mindone.diffusers.models.controlnets.controlnet.ControlNetModel",
            dict(
                block_out_channels=(32, 64),
                layers_per_block=2,
                in_channels=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                cross_attention_dim=32,
                conditioning_embedding_out_channels=(16, 32),
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            dict(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
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
                "controlnet",
                "scheduler",
                "vae",
                "text_encoder",
                "tokenizer",
                "safety_checker",
                "feature_extractor",
                "image_encoder",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        controlnet_embedder_scale_factor = 2
        pt_control_image = floats_tensor(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
            rng=random.Random(seed + 1),
        )
        ms_control_image = ms.Tensor(pt_control_image.numpy())
        init_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        init_image = init_image.cpu().permute(0, 2, 3, 1)[0]

        image = Image.fromarray(np.uint8(init_image)).convert("RGB").resize((64, 64))
        mask_image = Image.fromarray(np.uint8(init_image + 4)).convert("RGB").resize((64, 64))

        pt_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
            "image": image,
            "mask_image": mask_image,
            "control_image": pt_control_image,
        }

        ms_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
            "image": image,
            "mask_image": mask_image,
            "control_image": ms_control_image,
        }
        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.controlnet.StableDiffusionControlNetInpaintPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.controlnet.StableDiffusionControlNetInpaintPipeline")

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
class ControlNetInpaintPipelineSlowTests(PipelineTesterMixin, unittest.TestCase):
    def make_inpaint_condition(self, image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = ms.Tensor(image)
        return image

    @data(*test_cases)
    @unpack
    def test_canny(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        controlnet_cls = get_module("mindone.diffusers.models.controlnets.controlnet.ControlNetModel")
        controlnet = controlnet_cls.from_pretrained("lllyasviel/sd-controlnet-canny", mindspore_dtype=ms_dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.controlnet.StableDiffusionControlNetInpaintPipeline")
        pipe = pipe_cls.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
            variant="fp16",
            safety_checker=None,
            controlnet=controlnet,
            mindspore_dtype=ms_dtype,
        )
        pipe.set_progress_bar_config(disable=None)

        image = load_downloaded_image_from_hf_hub(
            "lllyasviel/sd-controlnet-canny",
            "bird.png",
            subfolder="images",
            repo_type="model",
        ).resize((512, 512))

        mask_image = load_downloaded_image_from_hf_hub(
            "diffusers/test-arrays",
            "input_bench_mask.png",
            subfolder="stable_diffusion_inpaint",
        ).resize((512, 512))

        prompt = "pitch black hole"

        control_image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "bird_canny.png",
            subfolder="sd_controlnet",
        ).resize((512, 512))

        torch.manual_seed(0)
        output = pipe(
            prompt,
            image=image,
            mask_image=mask_image,
            control_image=control_image,
            num_inference_steps=3,
        )

        image = output[0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"inpaint_canny_{dtype}.npy",
            subfolder="controlnet",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL

    @data(*test_cases)
    @unpack
    def test_inpaint(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        controlnet_cls = get_module("mindone.diffusers.models.controlnets.controlnet.ControlNetModel")
        controlnet = controlnet_cls.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", mindspore_dtype=ms_dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.controlnet.StableDiffusionControlNetInpaintPipeline")
        pipe = pipe_cls.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            safety_checker=None,
            controlnet=controlnet,
            mindspore_dtype=ms_dtype,
        )
        scheduler_cls = get_module("mindone.diffusers.schedulers.scheduling_ddim.DDIMScheduler")
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)

        init_image = load_downloaded_image_from_hf_hub(
            "diffusers/test-arrays",
            "boy.png",
            subfolder="stable_diffusion_inpaint",
        )
        init_image = init_image.resize((512, 512))

        mask_image = load_downloaded_image_from_hf_hub(
            "diffusers/test-arrays",
            "boy_mask.png",
            subfolder="stable_diffusion_inpaint",
        )
        mask_image = mask_image.resize((512, 512))

        prompt = "a handsome man with ray-ban sunglasses"

        control_image = self.make_inpaint_condition(init_image, mask_image)

        torch.manual_seed(33)
        output = pipe(
            prompt,
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
            guidance_scale=9.0,
            eta=1.0,
            num_inference_steps=20,
        )
        image = output[0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"inpaint_{dtype}.npy",
            subfolder="controlnet",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
