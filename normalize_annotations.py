"""
normalize_annotations.py — Hiren Manani, Syracuse University 2026

Converts COCO JSON bounding boxes from absolute pixel coordinates
to normalized [0,1] coordinates required by SAM 3's dataset loader.

WHY THIS IS CRITICAL:
  SAM 3 multiplies your coordinates by image dimensions internally.
  Roboflow exports absolute pixels (e.g. x=2297 in a 3088px image).
  SAM 3 computes: 2297 x 3088 = 7,093,936 — off-screen.
  Hungarian matcher finds zero matches -> loss_bbox=0.0 -> no learning.

Usage:
  python normalize_annotations.py --input _annotations.coco.json
  python normalize_annotations.py --input output.json --verify
"""
import json, argparse, os
import numpy as np

def normalize(input_path, output_path):
    with open(input_path) as f:
        coco = json.load(f)
    id2dim = {i["id"]:(i["width"],i["height"]) for i in coco["images"]}
    fixed = skipped = 0
    for ann in coco["annotations"]:
        W,H = id2dim[ann["image_id"]]
        raw = [float(str(v).strip("'\" ")) for v in ann["bbox"]]
        if all(v <= 1.0 for v in raw):
            skipped += 1; continue
        x,y,w,h = raw
        ann["bbox"] = [round(x/W,8), round(y/H,8), round(w/W,8), round(h/H,8)]
        fixed += 1
    print(f"Fixed: {fixed}  Skipped: {skipped}")
    with open(output_path,"w") as f:
        json.dump(coco, f)
    print(f"Saved -> {output_path}")

def verify(path):
    with open(path) as f:
        coco = json.load(f)
    id2dim = {i["id"]:(i["width"],i["height"]) for i in coco["images"]}
    widths_norm, widths_px = [], []
    for ann in coco["annotations"]:
        W,H = id2dim[ann["image_id"]]
        raw = [float(str(v).strip("'\" ")) for v in ann["bbox"]]
        widths_norm.append(raw[2]); widths_px.append(raw[2]*W)
    print(f"Normalized width: min={min(widths_norm):.4f} max={max(widths_norm):.4f}")
    print(f"Pixel width:      min={min(widths_px):.0f} max={max(widths_px):.0f}")
    print("OK" if max(widths_norm) <= 1.0 else "WARNING: values > 1.0!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    if args.verify:
        verify(args.input); exit()
    if not args.output:
        base,ext = os.path.splitext(args.input)
        args.output = base+"_normalized"+ext
    normalize(args.input, args.output)
    verify(args.output)
