import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

IMAGE_PATH = "images/ctl/231206_Rac1i_50uM_ctl_D2_(15).jpg"
SCORE_THRESHOLD = 0.20  # lower threshold to catch anything

PROMPTS = [
    "circular vesicle",
    "vesicle",
    "circle",
    "circular",
    "round cell",
    "cell",
    "droplet",
    "bubble",
    "sphere",
    "organoid",
    "liposome",
    "lipid vesicle",
    "hollow circle",
    "round object",
    "circular structure",
    "ring",
    "round droplet",
    "circular droplet",
    "unilamellar vesicle",
    "multilamellar vesicle",
]

print("Loading model once...")
model = build_sam3_image_model(checkpoint_path="./organoid_training_logs/checkpoints/checkpoint_5.pt", device="cpu")
processor = Sam3Processor(model)
print("Model loaded! Testing prompts...\n")

image = Image.open(IMAGE_PATH).convert("RGB")

results = []
for prompt in PROMPTS:
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    scores = output["scores"]
    count = sum(1 for s in scores if s > SCORE_THRESHOLD)
    results.append((count, prompt))
    print(f"  {count:4d} detections  |  '{prompt}'")

print("\n--- TOP PROMPTS ---")
for count, prompt in sorted(results, reverse=True)[:5]:
    print(f"  {count:4d}  |  '{prompt}'")
