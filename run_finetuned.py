import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

IMAGE_PATH = "images/ctl/231206_Rac1i_50uM_ctl_D2_(15).jpg"
FINETUNED_CHECKPOINT = "./organoid_training_logs/checkpoints/checkpoint.pt"
BASE_CHECKPOINT = "./checkpoints/sam3.pt"
FINETUNED_WEIGHTS = "./finetuned_weights.pt"
PROMPTS = ["circular vesicle", "circular", "round cell", "bubble"]
SCORE_THRESHOLD = 0.05
CIRCULARITY_MIN = 0.55
AREA_MIN = 200
AREA_MAX = 80000
IOU_THRESHOLD = 0.4

def compute_circularity(mask):
    area = mask.sum()
    perimeter = (np.sum(np.abs(np.diff(mask.astype(int), axis=0))) +
                 np.sum(np.abs(np.diff(mask.astype(int), axis=1))))
    if perimeter == 0:
        return 0
    return (4 * np.pi * area) / (perimeter ** 2)

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def to_numpy(x):
    if hasattr(x, 'cpu'):
        x = x.cpu()
    if hasattr(x, 'numpy'):
        x = x.numpy()
    return np.array(x)

def run_pipeline(model, image, label):
    processor = Sam3Processor(model, device="cuda")
    image_np = np.array(image)
    all_detections = []
    for prompt in PROMPTS:
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        for m, b, s in zip(output["masks"], output["boxes"], output["scores"]):
            s_val = float(to_numpy(s))
            if s_val < SCORE_THRESHOLD:
                continue
            mask = to_numpy(m[0]) if len(m.shape) > 2 else to_numpy(m)
            mask = (mask > 0).astype(np.uint8)
            area = mask.sum()
            if area < AREA_MIN or area > AREA_MAX:
                continue
            circ = compute_circularity(mask)
            if circ < CIRCULARITY_MIN:
                continue
            b_np = to_numpy(b).tolist()
            all_detections.append((mask, b_np, s_val))
    final = []
    for det in all_detections:
        duplicate = False
        for kept in final:
            if compute_iou(det[1], kept[1]) > IOU_THRESHOLD:
                duplicate = True
                break
        if not duplicate:
            final.append(det)
    print(f"[{label}] {len(final)} unique detections")
    return final, image_np

# Load base model
print("Loading base model...")
base_model = build_sam3_image_model(checkpoint_path=BASE_CHECKPOINT, device="cuda")
image = Image.open(IMAGE_PATH).convert("RGB")
base_dets, image_np = run_pipeline(base_model, image, "BASE")
del base_model
torch.cuda.empty_cache()

# Extract fine-tuned weights
print("Loading fine-tuned model...")
import os
if not os.path.exists(FINETUNED_WEIGHTS):
    ckpt = torch.load(FINETUNED_CHECKPOINT, map_location="cpu")
    torch.save(ckpt["model"], FINETUNED_WEIGHTS)
    print("Extracted fine-tuned weights")
finetuned_model = build_sam3_image_model(checkpoint_path=FINETUNED_WEIGHTS, device="cuda")
ft_dets, _ = run_pipeline(finetuned_model, image, "FINETUNED")
del finetuned_model
torch.cuda.empty_cache()

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(24, 12))
for ax, dets, title in zip(axes,
    [base_dets, ft_dets],
    [f"Base SAM 3 — {len(base_dets)} detections",
     f"Fine-tuned SAM 3 — {len(ft_dets)} detections"]):
    ax.imshow(image_np, cmap='gray')
    colors = plt.cm.hsv(np.linspace(0, 1, max(len(dets), 1)))
    for i, (mask, box, score) in enumerate(dets):
        color = colors[i]
        overlay = np.zeros((*image_np.shape[:2], 4))
        overlay[mask > 0] = [*color[:3], 0.35]
        ax.imshow(overlay)
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
            linewidth=1.5, edgecolor=color[:3], facecolor='none')
        ax.add_patch(rect)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')

plt.suptitle("SAM 3 Organoid Detection: Before vs After Fine-Tuning", fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig("comparison_output.png", dpi=150, bbox_inches='tight')
print("Saved comparison_output.png")
