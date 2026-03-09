from transformers import AutoModelForImageClassification
from PIL import Image
from timm.data.transforms_factory import create_transform
import requests
import torch
model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-T-1K", 
                                                        trust_remote_code=True)

# eval mode for inference
model.cuda().eval()

# prepare image for the model
url = 'http://images.cocodataset.org/val2017/000000020247.jpg'
image = Image.open(requests.get(url, stream=True).raw)
input_resolution = (3, 224, 224)  # MambaVision supports any input resolutions

transform = create_transform(input_size=input_resolution,
                             is_training=False,
                             mean=model.config.mean,
                             std=model.config.std,
                             crop_mode=model.config.crop_mode,
                             crop_pct=model.config.crop_pct)

inputs = transform(image).unsqueeze(0).cuda()
# model inference
# dummy_input = torch.randn(1, 3, 224, 224).cuda()

outputs = model(inputs)
logits = outputs['logits'] 
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])


### 
import numpy as np
logits = logits.detach().cpu().numpy()[0]
print(logits)
indices = np.argsort(logits)[::-1][:5]
print([(int(i), float(logits[i])) for i in indices])



