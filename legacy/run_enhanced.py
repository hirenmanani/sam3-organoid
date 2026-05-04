import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import cv2
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

IMAGE_PATH = "images/ctl/231206_Rac1i_50uM_ctl_D2_(15).jpg"
SCORE_THRESHOLD = 0.30
MIN_AREA = 200
MAX_AREA = 80000
MIN_CIRCULARITY = 0.55
IOU_THRESHOLD = 0.4
PROMPTS = ["circular vesicle", "circular", "bubble", "round cell"]

print("Loading model...")
model = build_sam3_image_model(checkpoint_path="./checkpoints/sam3.pt", device="cpu")
processor = Sam3Processor(model)
print("Model loaded!")

# Load and enhance contrast
raw_image = Image.open(IMAGE_PATH).convert("RGB")
enhancer = ImageEnhance.Contrast(raw_image)
image = enhancer.enhance(2.5)   # boost contrast 2.5x
image_np = np.array(raw_image)  # keep original for display

def compute_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return inter / union if union > 0 else 0

all_masks, all_boxes, all_scores = [], [], []
for prompt in PROMPTS:
    print(f"Running prompt: '{prompt}'...")
    state = processor.set_image(image)  # feed enhanced image
    output = processor.set_text_prompt(state=state, prompt=prompt)
    for m, b, s in zip(output["masks"], output["boxes"], output["scores"]):
        if s > SCORE_THRESHOLD:
            all_masks.append(m)
            all_boxes.append(b)
            all_scores.append(float(s))

print(f"Total before filtering: {len(all_masks)}")

filtered = []
for mask, box, score in zip(all_masks, all_boxes, all_scores):
    m = mask[0].cpu().numpy().astype(np.uint8)
    area = int(m.sum())
    if area < MIN_AREA or area > MAX_AREA:
        continue
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
    cnt = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * area / (perimeter ** 2)
    if circularity < MIN_CIRCULARITY:
        continue
    filtered.append((m, box, score, circularity))

filtered.sort(key=lambda x: x[2], reverse=True)
kept = []
for candidate in filtered:
    overlap = any(compute_iou(candidate[0], k[0]) > IOU_THRESHOLD for k in kept)
    if not overlap:
        kept.append(candidate)

print(f"After deduplication: {len(kept)} unique vesicles")

fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.imshow(image_np, cmap='gray')
colors = plt.cm.hsv(np.linspace(0, 1, max(len(kept), 1)))

for i, (m, box, score, circ) in enumerate(kept):
    color = colors[i]
    overlay = np.zeros((*image_np.shape[:2], 4))
    overlay[m > 0] = [*color[:3], 0.3]
    ax.imshow(overlay)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        poly = plt.Polygon(pts, fill=False, edgecolor=color[:3], linewidth=1.8)
        ax.add_patch(poly)
    x1, y1, x2, y2 = box
    ax.text(x1, y1-4, f"{score:.2f}", color=color[:3], fontsize=5, fontweight='bold')

ax.set_title(f"SAM 3 Enhanced — {len(kept)} vesicles (contrast boosted)", fontsize=11)
plt.axis('off')
plt.tight_layout()
plt.savefig("output_enhanced.png", dpi=200, bbox_inches='tight')
print("Saved to output_enhanced.png")
plt.show()
