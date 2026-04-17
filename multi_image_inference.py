import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

IMAGES = [
    ("images/ctl/231206_Rac1i_50uM_ctl_D2_(15).jpg", "Control D2"),
    ("images/ctl/231206_Rac1i_50uM_ctl_D4_(17).jpg", "Control D4"),
    ("images/ctl/231206_Rac1i_50uM_ctl_D6_(5).jpg", "Control D6"),
    ("images/treatment_D4-D6/231206_Rac1i_50uM_D4-D6_D6_(1).jpg", "Treatment D4-D6"),
]
PROMPTS = ["circular vesicle", "circular", "round cell", "bubble"]
SCORE_THRESHOLD = 0.25
CIRCULARITY_MIN = 0.55
AREA_MIN, AREA_MAX = 200, 80000
IOU_THRESHOLD = 0.4

def to_numpy(x):
    if hasattr(x, 'cpu'): x = x.cpu()
    if hasattr(x, 'numpy'): x = x.numpy()
    return np.array(x)

def compute_circularity(mask):
    area = mask.sum()
    perimeter = (np.sum(np.abs(np.diff(mask.astype(int), axis=0))) +
                 np.sum(np.abs(np.diff(mask.astype(int), axis=1))))
    return (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

def compute_iou(b1, b2):
    x1,y1 = max(b1[0],b2[0]), max(b1[1],b2[1])
    x2,y2 = min(b1[2],b2[2]), min(b1[3],b2[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter/union if union > 0 else 0

print("Loading model...")
model = build_sam3_image_model(checkpoint_path="./checkpoints/sam3.pt", device="cuda")

fig, axes = plt.subplots(2, 2, figsize=(24, 16))
axes = axes.flatten()

for idx, (img_path, label) in enumerate(IMAGES):
    print(f"Processing {label}...")
    try:
        image = Image.open(img_path).convert("RGB")
    except:
        print(f"Skipping {img_path}")
        continue
    image_np = np.array(image)
    processor = Sam3Processor(model, device="cuda")
    all_dets = []
    for prompt in PROMPTS:
        state = processor.set_image(image)
        out = processor.set_text_prompt(state=state, prompt=prompt)
        for m, b, s in zip(out["masks"], out["boxes"], out["scores"]):
            if float(to_numpy(s)) < SCORE_THRESHOLD: continue
            mask = (to_numpy(m[0] if len(m.shape)>2 else m) > 0).astype(np.uint8)
            if mask.sum() < AREA_MIN or mask.sum() > AREA_MAX: continue
            if compute_circularity(mask) < CIRCULARITY_MIN: continue
            b_np = to_numpy(b).tolist()
            all_dets.append((mask, b_np, float(to_numpy(s))))
    final = []
    for det in all_dets:
        if not any(compute_iou(det[1], k[1]) > IOU_THRESHOLD for k in final):
            final.append(det)
    print(f"  {label}: {len(final)} detections")
    ax = axes[idx]
    ax.imshow(image_np, cmap='gray')
    colors = plt.cm.hsv(np.linspace(0, 1, max(len(final), 1)))
    for i, (mask, box, score) in enumerate(final):
        color = colors[i]
        overlay = np.zeros((*image_np.shape[:2], 4))
        overlay[mask > 0] = [*color[:3], 0.35]
        ax.imshow(overlay)
        x1,y1,x2,y2 = float(box[0]),float(box[1]),float(box[2]),float(box[3])
        rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1.5,edgecolor=color[:3],facecolor='none')
        ax.add_patch(rect)
    ax.set_title(f"SAM 3 — {label} — {len(final)} organoids detected", fontsize=14, fontweight='bold')
    ax.axis('off')

plt.suptitle("SAM 3 Organoid Detection Across Treatment Conditions\n(Multi-Prompt Pipeline: circular vesicle, circular, round cell, bubble)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("multi_condition_output.png", dpi=150, bbox_inches='tight')
print("Saved multi_condition_output.png")
