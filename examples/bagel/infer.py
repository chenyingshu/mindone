import os
import time

from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
from PIL import Image

# from modeling.qwen2 import Qwen2Tokenizer
from transformers import Qwen2Tokenizer

import mindspore as ms

from examples.bagel.data.transforms import ImageTransform

ms.set_context(mode=1)
ms.runtime.launch_blocking()  # debug use

# Step 1: Model Initialization #

model_path = "/path/to/BAGEL-7B-MoT/weights"  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT

# LLM config preparing
llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing
vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# VAE loading
print("*" * 50)
print(f"Loading AE ckpt in {model_path}/ae.safetensors.")
start_time = time.time()
vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
print("Loaded VAE, time elapsed %.5fs" % (time.time() - start_time))
vae_model.set_train(False)

# Bagel config preparing
config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config,
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act="gelu_pytorch_tanh",
    latent_patch_size=2,
    max_latent_size=64,
)

# with init_empty_weights():
print("*" * 50)
print("Initializing LLM, ViT, and Bagel...")
start_time = time.time()
language_model = Qwen2ForCausalLM(llm_config)
vit_model = SiglipVisionModel(vit_config)
model = Bagel(language_model, vit_model, config)
model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
print("Initialized LLM, ViT, Bagel Model, time elapsed %.5fs" % (time.time() - start_time))

start_time = time.time()
model = model.to(ms.bfloat16)
print("Convert Bagel to bf16, time elapsed %.5fs" % (time.time() - start_time))

# Tokenizer Preparing
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)


# Step 2: Model Loading #
print("*" * 50)
print(f"Loading Bagel ckpt in {model_file}.")
# load pretrained checkpoint
model_file = os.path.join(model_path, "ema.safetensors")
start_time = time.time()
state_dict = ms.load_checkpoint(model_file, format="safetensors")
# Check loading keys:
model_state_dict = {k: v for k, v in model.parameters_and_names()}
# state_dict_tmp = {}
# for k, v in state_dict.items():
#     if ("norm" in k) and ("mlp" not in k):  # for LayerNorm but not ModLN's mlp
#         k = k.replace(".weight", ".gamma").replace(".bias", ".beta")
#     if "adam_" not in k:  # not to load optimizer
#         state_dict_tmp[k] = v
# state_dict = state_dict_tmp
loaded_keys = list(state_dict.keys())
expexted_keys = list(model_state_dict.keys())
original_loaded_keys = loaded_keys
missing_keys = list(set(expexted_keys) - set(loaded_keys))
unexpected_keys = list(set(loaded_keys) - set(expexted_keys))
mismatched_keys = []
for checkpoint_key in original_loaded_keys:
    if (
        checkpoint_key in model_state_dict
        and checkpoint_key in state_dict
        and state_dict[checkpoint_key].shape != model_state_dict[checkpoint_key].shape
    ):
        mismatched_keys.append(
            (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[checkpoint_key].shape)
        )

print(
    f"Loading BagelModel...\nmissing_keys: {missing_keys}, \nunexpected_keys: {unexpected_keys}, \nmismatched_keys: {mismatched_keys}"
)
print(f"Bagel state_dict.dtype {state_dict[loaded_keys[0]].dtype}")  # bfloat16
# Instantiate the model
param_not_load, ckpt_not_load = ms.load_param_into_net(model, state_dict, strict_load=False)
print(f"Loaded Bagel checkpoint: param_not_load {param_not_load}, ckpt_not_load {ckpt_not_load}")
print("Loaded Bagel, time elapsed %.5fs" % (time.time() - start_time))

model.set_train(False)

print("All models loaded.")


# Step 3: Inferencer Preparing #
from inferencer import InterleaveInferencer

inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids,
)

import random

import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
ms.set_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


# Inference 1: Image Generation #
print("# Inference 1: Image Generation #")
inference_hyper = dict(
    cfg_text_scale=4.0,
    cfg_img_scale=1.0,
    cfg_interval=[0.4, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=1.0,
    cfg_renorm_type="global",
)
prompt = (
    "A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, "
    "mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere."
)

print(prompt)
print("-" * 10)
output_dict = inferencer(text=prompt, **inference_hyper)
output_dict["image"].save("infer1_img_gen.jpg")

# Inference 2: Image Generation with Think #
print("# Inference 2: Image Generation with Think #")
inference_hyper = dict(
    max_think_token_n=1000,
    do_sample=False,
    # text_temperature=0.3,
    cfg_text_scale=4.0,
    cfg_img_scale=1.0,
    cfg_interval=[0.4, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=1.0,
    cfg_renorm_type="global",
)
prompt = "a car made of small cars"

print(prompt)
print("-" * 10)
output_dict = inferencer(text=prompt, think=True, **inference_hyper)
print(output_dict["text"])
output_dict["image"].save("infer2_img_gen_think.jpg")

# Inference 3: Editing #
print("# Inference 3: Editing #")
inference_hyper = dict(
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=1.0,
    cfg_renorm_type="text_channel",
)
image = Image.open("test_images/women.jpg")
prompt = "She boards a modern subway, quietly reading a folded newspaper, wearing the same clothes."

# display(image)
print(prompt)
print("-" * 10)
output_dict = inferencer(image=image, text=prompt, **inference_hyper)
output_dict["image"].save("infer3_img_editing.jpg")

# Inference 4: Edit with Think #
print("# Inference 4: Edit with Think #")
inference_hyper = dict(
    max_think_token_n=1000,
    do_sample=False,
    # text_temperature=0.3,
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
)
image = Image.open("test_images/octupusy.jpg")
prompt = "Could you display the sculpture that takes after this design?"

# display(image)
print("-" * 10)
output_dict = inferencer(image=image, text=prompt, think=True, **inference_hyper)
print(output_dict["text"])
output_dict["image"].save("infer4_img_editing_think.jpg")

# Inference 5: Understanding #
print("# Inference 5: Understanding #")
inference_hyper = dict(
    max_think_token_n=1000,
    do_sample=False,
    # text_temperature=0.3,
)
image = Image.open("test_images/meme.jpg")
prompt = "Can someone explain what’s funny about this meme??"

# display(image)
print(prompt)
print("-" * 10)
output_dict = inferencer(image=image, text=prompt, understanding_output=True, **inference_hyper)
print(output_dict["text"])
