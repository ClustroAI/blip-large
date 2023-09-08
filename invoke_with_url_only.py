import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import json

# Loading the model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

# Invoke function
def invoke(input_text):
    # Parsing the input_text
    image_url = input_text
    raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to("cuda")
    # Run inference
    out = model.generate(**inputs)
    # Return result
    return processor.decode(out[0], skip_special_tokens=True)
