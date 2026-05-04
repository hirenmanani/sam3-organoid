# Engineering Fixes — SAM 3 Organoid Fine-Tuning

10 issues resolved. All required for valid training.

1. HTCondor idle for days     → use +request_gpus = 1 (lowercase, with +)
2. CUDA not available          → reinstall PyTorch cu118 for RTX 6000
3. Hydra config not found      → add __init__.py to configs/organoid/
4. EOFError loading checkpoint → re-download from HuggingFace (GitHub truncates >100MB)
5. loss_bbox=0 (config)        → add roboflow100: ${roboflow_train.loss} to loss section
6. loss_bbox=0 (data) ★        → run normalize_annotations.py (see below)
7. CPU/GPU tensor mismatch     → add .to(boxes_xyxy.device) in decoder.py
8. GradScaler crash            → set AMP enabled: false in config
9. Half batches zero matches   → remove empty category 0 from COCO JSON
10. DDP unused param crash     → freeze vision_backbone.convs + text_projection

★ Bug 6 — Most Critical:
  SAM 3 loader multiplies coords by image dims. Roboflow exports absolute pixels.
  2297 (pixels) × 3088 (width) = 7,093,936 — off-screen.
  Hungarian matcher: zero matches → loss_bbox=0 → no gradients to detection head.
  Fix: python normalize_annotations.py --input annotations.coco.json
