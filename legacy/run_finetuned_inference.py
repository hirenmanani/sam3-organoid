import torch
import sys
sys.path.insert(0, '/home/hmanani/sam3')
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

IMAGE_PATH = "images/ctl/231206_Rac1i_50uM_ctl_D2_(15).jpg"
SCORE_THRESHOLD = 0.25
PROMPTS = [
    "circular vesicle",
    "spherical vesicle",
    "round hollow structure",
    "organoid",
]

print("Loading base model sam3.pt...")
model = build_sam3_image_model(
    checkpoint_path="./checkpoints/sam3.pt",
    device="cpu",
    eval_mode=False,
    enable_segmentation=False
)

print("Overlaying fine-tuned weights...")
ft_ckpt = torch.load(
    "./organoid_training_logs_fix/checkpoints/checkpoint.pt",
    map_location="cpu"
)
missing, unexpected = model.load_state_dict(ft_ckpt["model"], strict=False)
print(f"Loaded fine-tuned weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

model.eval()
processor = Sam3Processor(model)
print("Model ready!")

image = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(image)
inference_state = processor.set_image(image)

total = 0
for prompt in PROMPTS:
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    filtered = [s for s in output["scores"] if s > SCORE_THRESHOLD]
    print(f"Prompt: '{prompt}' --> {len(filtered)} detections")
    total += len(filtered)

print(f"\nTotal detections: {total}")
