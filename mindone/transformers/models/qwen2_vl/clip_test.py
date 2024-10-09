from mindspore import Tensor
from transformers import CLIPTokenizer
from mindone.transformers import CLIPTextModel

MODEL_NAME = "openai/clip-vit-large-patch14"
model = CLIPTextModel.from_pretrained(MODEL_NAME)
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

text_inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="np")
text_outputs = model(Tensor(text_inputs.input_ids))
print(text_outputs)
'''

import requests
from PIL import Image
from transformers import Blip2Processor
from mindone.transformers import Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

question = "how many dogs are in the picture?"
inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True).strip())
'''