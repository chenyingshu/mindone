# Copyright 2025 The HuggingFace Team.
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
from PIL import Image

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
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "bfloat16"},
]


@ddt
class CogVideoXVideoToVideoPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.cogvideox_transformer_3d.CogVideoXTransformer3DModel",
            "mindone.diffusers.models.transformers.cogvideox_transformer_3d.CogVideoXTransformer3DModel",
            dict(
                # Product of num_attention_heads * attention_head_dim must be divisible by 16 for 3D positional embeddings
                # But, since we are using tiny-random-t5 here, we need the internal dim of CogVideoXTransformer3DModel
                # to be 32. The internal dim is product of num_attention_heads and attention_head_dim
                num_attention_heads=4,
                attention_head_dim=8,
                in_channels=4,
                out_channels=4,
                time_embed_dim=2,
                text_embed_dim=32,  # Must match with tiny-random-t5
                num_layers=1,
                sample_width=2,  # latent width: 2 -> final width: 16
                sample_height=2,  # latent height: 2 -> final height: 16
                sample_frames=9,  # latent frames: (9 - 1) / 4 + 1 = 3 -> final frames: 9
                patch_size=2,
                temporal_compression_ratio=4,
                max_text_seq_length=16,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl_cogvideox.AutoencoderKLCogVideoX",
            "mindone.diffusers.models.autoencoders.autoencoder_kl_cogvideox.AutoencoderKLCogVideoX",
            dict(
                in_channels=3,
                out_channels=3,
                down_block_types=(
                    "CogVideoXDownBlock3D",
                    "CogVideoXDownBlock3D",
                    "CogVideoXDownBlock3D",
                    "CogVideoXDownBlock3D",
                ),
                up_block_types=(
                    "CogVideoXUpBlock3D",
                    "CogVideoXUpBlock3D",
                    "CogVideoXUpBlock3D",
                    "CogVideoXUpBlock3D",
                ),
                block_out_channels=(8, 8, 8, 8),
                latent_channels=4,
                layers_per_block=1,
                norm_num_groups=2,
                temporal_compression_ratio=4,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            dict(),
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

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "transformer",
                "vae",
                "scheduler",
                "text_encoder",
                "tokenizer",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, num_frames: int = 8):
        video_height = 16
        video_width = 16
        video = [Image.new("RGB", (video_width, video_height))] * num_frames

        inputs = {
            "video": video,
            "prompt": "dance monkey",
            "negative_prompt": "",
            "num_inference_steps": 2,
            "strength": 0.5,
            "guidance_scale": 6.0,
            # Cannot reduce because convolution kernel becomes bigger than sample
            "height": video_height,
            "width": video_width,
            "max_sequence_length": 16,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module(
            "diffusers.pipelines.cogvideo.pipeline_cogvideox_video2video.CogVideoXVideoToVideoPipeline"
        )
        ms_pipe_cls = get_module(
            "mindone.diffusers.pipelines.cogvideo.pipeline_cogvideox_video2video.CogVideoXVideoToVideoPipeline"
        )

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_video = pt_pipe(**inputs).frames
        torch.manual_seed(0)
        ms_video = ms_pipe(**inputs)[0]

        pt_generated_video = pt_video[0]
        ms_generated_video = ms_video[0]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert (
            np.max(np.linalg.norm(pt_generated_video - ms_generated_video) / np.linalg.norm(pt_generated_video))
            < threshold
        )


@slow
@ddt
class CogVideoXVideoToVideoPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_cogvideox(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module(
            "mindone.diffusers.pipelines.cogvideo.pipeline_cogvideox_video2video.CogVideoXVideoToVideoPipeline"
        )
        pipe = pipe_cls.from_pretrained("THUDM/CogVideoX-5b", mindspore_dtype=ms_dtype)
        scheduler_cls = get_module("mindone.diffusers.schedulers.scheduling_dpm_cogvideox.CogVideoXDPMScheduler")
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)

        input_video_np = load_numpy_from_local_file(
            "mindone-testing-arrays",
            "hiker.npy",
            subfolder="cogvideo",
        )

        input_video = []
        for img_array in input_video_np:
            # 将 numpy 数组转换为 PIL 图像对象
            image = Image.fromarray(img_array)
            # 将 PIL 图像对象添加到列表中
            input_video.append(image)

        prompt = (
            "An astronaut stands triumphantly at the peak of a towering mountain. Panorama of rugged peaks and "
            "valleys. Very futuristic vibe and animated aesthetic. Highlights of purple and golden colors in "
            "the scene. The sky is looks like an animated/cartoonish dream of galaxies, nebulae, stars, planets, "
            "moons, but the remainder of the scene is mostly realistic."
        )

        torch.manual_seed(0)
        videos = pipe(video=input_video, prompt=prompt, strength=0.8, guidance_scale=6, num_inference_steps=50)[0]
        video = videos[0]

        expected_video = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"cogvideox_video2video_{dtype}.npy",
            subfolder="cogvideo",
        )
        assert np.mean(np.abs(np.array(video, dtype=np.float32) - expected_video)) < THRESHOLD_PIXEL
