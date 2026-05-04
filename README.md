# SAM 3 for Organoid Brightfield Image Analysis

**From Detection to Segmentation: A Foundation Model Approach to Organoid Brightfield Image Analysis Using SAM 3**

Hiren Manani · M.S. Computer Science (AI/ML) · Syracuse University · May 2026  
Advisor: Prof. Michael R. Blatchley · Co-Advisor: Prof. Senem Velipasalar Gursoy

---

## What This Repository Does

This repository adapts [SAM 3](https://github.com/facebookresearch/sam3) — Meta's open-vocabulary segmentation model — to automatically detect and segment mouse small intestinal organoids in brightfield microscopy images, **without any manual clicking per image**.

You describe what you're looking for in plain English ("circular", "cell") and SAM 3 finds all matching organoids across your entire image set automatically.

**Key results:**
| Model | Detections/image | Precision | Recall | F1 |
|-------|-----------------|-----------|--------|----|
| TellU (lab baseline) | 112 (mean) | ~0.82 | ~0.80 | ~0.81 |
| SAM 3 Zero-shot | 122 (multi-prompt) | 0.843 | 0.692 | 0.760 |
| SAM 3 Fine-tuned | 139 (multi-prompt) | — | — | — |

Fine-tuning improved detection by **+7% to +21%** per prompt over the zero-shot baseline.

---

## Repository Structure

```
sam3-organoid/
│
├── README.md                      ← You are here
├── EVALUATION_README.md           ← Evaluation & analysis scripts guide
│
├── evaluation/                    ← Scripts to evaluate model performance
│   ├── sam3_eval_script.py        ← Main eval script (HTCondor GPU cluster)
│   └── ft_eval.py                 ← CLI eval script with arguments
│
├── tellu_analysis/                ← TellU vs SAM 3 comparison
│   └── tellu_analysis.py          ← Generates all comparison figures
│
├── find_prompts.py                ← Run multi-prompt detection on one image
├── multi_image_inference.py       ← Run inference on a folder of images
├── run_finetuned.py               ← Run the fine-tuned model
├── normalize_annotations.py       ← Fix COCO bbox coordinates (critical!)
├── augment_rle.py                 ← Data augmentation pipeline
├── debug_scores.py                ← Debug detection scores
│
├── sam3/                          ← SAM 3 source code (patched for organoids)
│   └── train/configs/organoid/    ← Fine-tuning configuration files
│
├── my_organoid_dataset/           ← Roboflow dataset (COCO JSON format)
├── organoid_training_logs/        ← Training logs and checkpoints (Job 1)
├── organoid_training_logs_combined/ ← Training logs (Job 2, expanded dataset)
│
├── submit_sam3.sub                ← HTCondor GPU job submit file
└── run_sam3.sh                    ← Wrapper shell script for cluster jobs
```

---

## Quick Start

### 1. Prerequisites

```bash
# Python 3.10, PyTorch with CUDA
conda create -n sam3 python=3.10
conda activate sam3

# Install PyTorch for CUDA 11.8 (OrangeGrid RTX 6000 compatible)
pip install torch==2.3.1+cu118 torchvision --index-url https://download.pytorch.org/whl/cu118

# Install SAM 3
cd ~/sam3
pip install -e .
pip install opencv-python einops psutil
```

### 2. Download the SAM 3 Checkpoint

```bash
# Authenticate with HuggingFace
hf auth login   # enter your HF token

# Download checkpoint (~3.45 GB)
hf download facebook/sam3 sam3.pt --local-dir ./checkpoints/
```

> ⚠️ **Important:** GitHub's 100MB limit means the checkpoint cannot be stored here.  
> Always download from HuggingFace directly.

### 3. Run Detection on Your Images

```bash
# Edit find_prompts.py to point to your image
# Change IMAGE_PATH to your image file
python find_prompts.py
```

This tests 20+ prompts and shows which ones detect the most organoids.  
**Key finding:** Shape-based prompts work best — `"cell"`, `"circular"`, `"round cell"`, `"bubble"`.  
Biological terms like `"organoid"` return **0 detections** (not in SAM 3's vocabulary).

### 4. Run on Multiple Images

```bash
python multi_image_inference.py
```

---

## Fine-Tuning on Your Own Organoid Images

### Step 1: Annotate Your Images

1. Upload images to [Roboflow](https://roboflow.com)
2. Create an **Instance Segmentation** project
3. Label organoids using Smart Polygon tool
4. Export in **COCO JSON** format
5. Download to `my_organoid_dataset/`

### Step 2: Fix COCO Annotation Coordinates (Critical!)

> ⚠️ **This is the most important step.** SAM 3's dataset loader expects **normalized [0,1] coordinates**, but Roboflow exports **absolute pixel coordinates**. If you skip this step, bbox loss will be 0.0 throughout training and the model will not learn.

```bash
python normalize_annotations.py \
    --input  my_organoid_dataset/organoid-segmentation/train/_annotations.coco.json \
    --output my_organoid_dataset/organoid-segmentation/train/_annotations_normalized.coco.json
```

Verify the fix:
```python
import json
with open("_annotations_normalized.coco.json") as f:
    d = json.load(f)
bbox = d["annotations"][0]["bbox"]
print(bbox)  # Should be like [0.74, 0.82, 0.04, 0.07] — all values between 0 and 1
```

### Step 3: Augment Your Dataset

```bash
python augment_rle.py \
    --input  my_organoid_dataset/organoid-segmentation/train/_annotations_normalized.coco.json \
    --output my_organoid_dataset/augmented/ \
    --factor 5
```

This creates horizontal flip, vertical flip, 90° rotation, and brightness/contrast variants (5× expansion).

### Step 4: Configure Fine-Tuning

Edit `sam3/train/configs/organoid/organoid_finetune_gpu.yaml`:

```yaml
# Key parameters to check:
dataset_key: roboflow100          # Must match your dataset key
img_folder: /path/to/your/images/
ann_file: /path/to/_annotations_normalized.coco.json

# Loss routing — MUST include your dataset key
loss_fns_find:
  roboflow100: ${roboflow_train.loss}   # ← This line is critical

# Matcher cost weights (tuned for single-category datasets)
cost_class: 0.1     # Low — spatial overlap drives matching, not class
cost_bbox: 5.0
cost_giou: 2.0

# Learning rate scale
lr_scale: 0.01      # 100× increase from default 0.0001
```

### Step 5: Submit to GPU Cluster (OrangeGrid)

```bash
# Edit run_sam3.sh to point to your config
condor_submit submit_sam3.sub

# Monitor
condor_q
tail -f ~/sam3_train.out
```

**Expected training output:**
```
Epoch 0, batch 0: loss=0.434
Epoch 0, batch 50: loss=0.371
Epoch 0, batch 170: loss=0.296
```

If you see `loss_bbox: 0.0` — check Step 2 (normalization) and the loss routing config.

### Step 6: Run Inference with Fine-Tuned Model

```bash
python run_finetuned.py
```

---

## Common Errors & Fixes

These 10 issues were encountered and resolved during this project:

| # | Error | Fix |
|---|-------|-----|
| 1 | HTCondor job sits idle for days | Use `+request_gpus = 1` (lowercase, with `+`) not `request_GPUs` |
| 2 | `RuntimeError: CUDA not available` | Reinstall PyTorch for correct CUDA version (`cu118` for RTX 6000) |
| 3 | `Error: config not found` | Add empty `__init__.py` to `configs/organoid/` folder |
| 4 | `EOFError` loading checkpoint | Re-download from HuggingFace — GitHub truncates files >100MB to 0 bytes |
| 5 | `loss_bbox = 0.0` throughout training | Add `roboflow100: ${roboflow_train.loss}` to loss config section |
| 6 | **`loss_bbox = 0.0` — annotations issue** | **Run `normalize_annotations.py` — Roboflow exports absolute pixels, SAM 3 needs [0,1]** |
| 7 | `RuntimeError: CPU/GPU tensor mismatch` | Patch `decoder.py`: add `.to(boxes_xyxy.device)` to coordinate tensors |
| 8 | `GradScaler` crash with bfloat16 | Set `enabled: false` in AMP config |
| 9 | Half of batches produce zero matches | Remove empty category 0 ("objects") from COCO JSON |
| 10 | `DDP unused parameter` crash | Freeze `vision_backbone.convs` and `text_projection` before DDP init |

> **Bug #6 is the most critical.** SAM 3's loader multiplies your coordinates by image dimensions internally. If you pass `x=2297` (absolute pixels), it computes `2297 × 3088 = 7,093,936` — completely off-screen. The Hungarian matcher finds zero valid matches and no gradients flow to the detection head.

---

## Dataset Information

- **Organism:** Mouse small intestinal organoids (Lgr5+ stem cells)
- **Culture:** Matrigel (Corning 356231), ENRCV media
- **Microscope:** Olympus CKX53, brightfield, no fluorescence
- **Acquired:** Anseth Lab, University of Colorado Anschutz Medical Campus
- **Conditions:** Control, D0–D2 Rac1i, D0–D4, D0–D6, D2–D6, D4–D6 (50µM)
- **Time points:** Day 2, Day 4, Day 6
- **Scale bar:** 200 µm (consistent across all images)
- **Annotations:** ~220 images, 140,065+ COCO bounding boxes via Roboflow

---

## Comparison with TellU

TellU is the organoid analysis tool currently used in the Blatchley laboratory.  
SAM 3 was evaluated against TellU on the same 20 test images:

```
python tellu_analysis/tellu_analysis.py \
    --tellu_dir /path/to/TellU/runs/detect/Test \
    --out_dir   comparison_figures/
```

| Metric | TellU | SAM 3 Baseline | SAM 3 Fine-tuned |
|--------|-------|----------------|-----------------|
| Mean detections | 112 | 122 | 139 |
| Count agreement | — | 90.9% | 76.0%* |
| Morphology classes | 4 | 1 | 1 |

\* Fine-tuned over-detects relative to TellU — reflects improved sensitivity.

**Key difference:** TellU classifies organoids into Spheroid/Cyst/EarlyOrganoid/LateOrganoid.  
SAM 3 currently outputs binary (organoid/not). Morphology classification is future work.

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{manani2026sam3organoid,
  author  = {Hiren Manani},
  title   = {From Detection to Segmentation: A Foundation Model Approach
             to Organoid Brightfield Image Analysis Using SAM 3},
  school  = {Syracuse University},
  year    = {2026},
  advisor = {Michael R. Blatchley}
}
```

Also cite the original SAM 3 paper:

```bibtex
@misc{carion2025sam3segmentconcepts,
  title   = {SAM 3: Segment Anything with Concepts},
  author  = {Nicolas Carion and Laura Gustafson and Yuan-Ting Hu et al.},
  year    = {2025},
  eprint  = {2511.16719},
  url     = {https://arxiv.org/abs/2511.16719}
}
```

---

## Contact

**Hiren Manani**  
hmanani@syr.edu · [github.com/hirenmanani](https://github.com/hirenmanani) · [hirenmanani.github.io/portfolio](https://hirenmanani.github.io/portfolio)

For biological questions about the dataset, contact:  
**Prof. Michael R. Blatchley** · mrblatch@syr.edu · Syracuse University
