# Evaluation Scripts

## sam3_eval_script.py
Evaluates SAM 3 baseline and fine-tuned on held-out images.
Computes Precision, Recall, F1 at IoU threshold 0.50.

Results:
  SAM 3 Baseline:   P=0.843  R=0.692  F1=0.760 (30 images, IoU@0.5)
  SAM 3 Fine-tuned: +13.6% detection count vs baseline

## ft_eval.py
CLI version:
  python ft_eval.py \
    --gt_json /path/to/annotations.json \
    --img_dir /path/to/images \
    --baseline_ckpt  checkpoints/sam3.pt \
    --finetuned_ckpt organoid_training_logs/checkpoints/checkpoint_5.pt \
    --max_images 30
