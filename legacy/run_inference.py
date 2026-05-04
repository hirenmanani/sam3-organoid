import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

IMAGE_PATH = "images/ctl/231206_Rac1i_50uM_ctl_D2_(15).jpg"
TEXT_PROMPT = "circular vesicle"
SCORE_THRESHOLD = 0.25

print("Loading model...")
model = build_sam3_image_model(checkpoint_path="./checkpoints/sam3.pt", device="cpu")
processor = Sam3Processor(model)
print("Model loaded! Running inference...")

image = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(image)

inference_state = processor.set_image(image)
output = processor.set_text_prompt(state=inference_state, prompt=TEXT_PROMPT)

masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]

filtered = [(m, b, s) for m, b, s in zip(masks, boxes, scores) if s > SCORE_THRESHOLD]
print(f"Detected {len(filtered)} objects above threshold {SCORE_THRESHOLD}")

fig, ax = plt.subplots(1, 1, figsize=(16, 11))
ax.imshow(image_np, cmap='gray')
colors = plt.cm.hsv(np.linspace(0, 1, max(len(filtered), 1)))

for i, (mask, box, score) in enumerate(filtered):
    color = colors[i]
    overlay = np.zeros((*image_np.shape[:2], 4))
    overlay[mask[0] > 0] = [*color[:3], 0.35]
    ax.imshow(overlay)
    x1, y1, x2, y2 = box
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
        linewidth=1.5, edgecolor=color[:3], facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1-4, f"{score:.2f}", color=color[:3], fontsize=6, fontweight='bold')

ax.set_title(f"SAM 3 — '{TEXT_PROMPT}' — {len(filtered)} detections", fontsize=13)
plt.axis('off')
plt.tight_layout()
plt.savefig("output_result.png", dpi=200)
print("Saved to output_result.png")
plt.show()
