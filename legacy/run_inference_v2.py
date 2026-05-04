import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

IMAGE_PATH = "/home/hmanani/data/new_dataset/train/231206_Rac1i_50uM_D0-D4_D2_(10)_jpg.rf.4wTdjTQruOzU42lmxXXj.jpg"
TEXT_PROMPT = "spherical lumen"
SCORE_THRESHOLD = 0.25

print("Loading model...")
model = build_sam3_image_model(checkpoint_path="./checkpoints/sam3.pt", device="cpu")
processor = Sam3Processor(model)
print("Model loaded! Running inference...")

image = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(image)

inference_state = processor.set_image(image)
output = processor.set_text_prompt(state=inference_state, prompt=TEXT_PROMPT)

masks  = output["masks"]
boxes  = output["boxes"]
scores = output["scores"]

filtered = [(m, b, s) for m, b, s in zip(masks, boxes, scores) if s > SCORE_THRESHOLD]
print(f"Detected {len(filtered)} objects above threshold {SCORE_THRESHOLD}")

fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.imshow(image_np, cmap='gray')
colors = plt.cm.hsv(np.linspace(0, 1, max(len(filtered), 1)))

for i, (mask, box, score) in enumerate(filtered):
    color = colors[i]
    m = mask[0].astype(np.uint8)  # binary mask

    # Draw filled mask overlay
    overlay = np.zeros((*image_np.shape[:2], 4))
    overlay[m > 0] = [*color[:3], 0.3]
    ax.imshow(overlay)

    # Draw contour outline (tight boundary, not bounding box)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        poly = plt.Polygon(pts, fill=False, edgecolor=color[:3], linewidth=1.5)
        ax.add_patch(poly)

    # Score label
    x1, y1, x2, y2 = box
    ax.text(x1, y1-4, f"{score:.2f}", color=color[:3], fontsize=5, fontweight='bold')

ax.set_title(f"SAM 3 — '{TEXT_PROMPT}' — {len(filtered)} detections", fontsize=13)
plt.axis('off')
plt.tight_layout()
plt.savefig("output_result_v2.png", dpi=200, bbox_inches='tight')
print("Saved to output_result_v2.png")
plt.show()
