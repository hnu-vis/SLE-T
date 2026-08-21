import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"


from transformers import pipeline
from transformers.image_utils import load_image

print(1111)




feature_extractor = pipeline(
    model="facebook/dinov3-convnext-base-pretrain-lvd1689m",
    task="image-feature-extraction", 
)
