# Fine-Tuning SAM 3 on Organoid Images

## Step 1: Annotate
- Upload to Roboflow → Instance Segmentation project
- Label with Smart Polygon, single class "organoid"
- Export as COCO JSON

## Step 2: Normalize Annotations (CRITICAL — do not skip)
    python normalize_annotations.py --input _annotations.coco.json
    python normalize_annotations.py --input _annotations_normalized.coco.json --verify

If skipped: loss_bbox = 0.0 throughout training. Model learns nothing.

## Step 3: Augment
    python augment_rle.py

## Step 4: Configure
Edit sam3/train/configs/organoid/organoid_finetune_gpu.yaml:
- img_folder and ann_file paths
- Loss routing must include: roboflow100: ${roboflow_train.loss}
- Matcher: cost_class=0.1, cost_bbox=5.0, cost_giou=2.0
- lr_scale: 0.01

## Step 5: Submit to OrangeGrid
    condor_submit cluster_scripts/submit_sam3.sub
    tail -f ~/sam3_train.out

Expected: loss decreases from ~0.43 to ~0.30 in epoch 0.
If loss_bbox = 0.0: check normalization and loss routing config.

## Step 6: Inference
    python run_finetuned.py
