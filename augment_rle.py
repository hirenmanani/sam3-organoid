import os, json, cv2
import numpy as np
import albumentations as A
from pycocotools import mask as maskUtils

DATASETS = {
    "D2-D6": "/home/hmanani/data/Organoid/D2-D6/train",
    "D4-D6": "/home/hmanani/data/Organoid/D4-D6/train",
    "main":  "/home/hmanani/data/Organoid/main/train",
}
OUTPUT_DIR = "/home/hmanani/data/augmented/train"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_coco(json_path):
    with open(json_path) as f:
        return json.load(f)

def get_anns_for_image(coco, image_id):
    return [a for a in coco["annotations"] if a["image_id"] == image_id]

def seg_to_mask(seg, ih, iw):
    if isinstance(seg, dict):
        rle = {"counts": seg["counts"].encode("utf-8"), "size": seg["size"]}
        return maskUtils.decode(rle).astype(np.uint8)
    else:
        mask = np.zeros((ih, iw), dtype=np.uint8)
        for poly in seg:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask

def mask_to_rle(mask):
    rle = maskUtils.encode(np.asfortranarray(mask))
    return {"counts": rle["counts"].decode("utf-8"), "size": rle["size"]}

def bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0, 0, 1, 1]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax-cmin+1), int(rmax-rmin+1)]

def transform_mask(mask, aug_type):
    if aug_type == "hflip":
        return cv2.flip(mask, 1)
    elif aug_type == "vflip":
        return cv2.flip(mask, 0)
    elif aug_type == "rot90":
        return cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    else:
        return mask

merged = {"images": [], "annotations": [], "categories": []}
img_id = 1
ann_id = 1
cats_set = False

for ds, train_dir in DATASETS.items():
    coco = load_coco(os.path.join(train_dir, "_annotations.coco.json"))
    if not cats_set:
        merged["categories"] = coco["categories"]
        cats_set = True

    for img_info in coco["images"]:
        img = cv2.imread(os.path.join(train_dir, img_info["file_name"]))
        if img is None:
            print(f"Skipping: {img_info['file_name']}")
            continue
        ih, iw = img.shape[:2]
        anns = get_anns_for_image(coco, img_info["id"])

        aug_list = [
            ("orig",       img),
            ("hflip",      cv2.flip(img, 1)),
            ("vflip",      cv2.flip(img, 0)),
            ("rot90",      cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
            ("brightness", A.RandomBrightnessContrast(p=1.0)(image=img)["image"]),
        ]

        for aug_name, aug_img in aug_list:
            fname = f"{ds}_{aug_name}_{img_info['file_name']}"
            cv2.imwrite(os.path.join(OUTPUT_DIR, fname), aug_img)
            ah, aw = aug_img.shape[:2]
            merged["images"].append({"id": img_id, "file_name": fname, "width": aw, "height": ah})

            for ann in anns:
                mask = seg_to_mask(ann["segmentation"], ih, iw)
                t_mask = transform_mask(mask, aug_name)
                new_seg = mask_to_rle(t_mask)
                new_bbox = bbox_from_mask(t_mask)
                merged["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": ann["category_id"],
                    "segmentation": new_seg,
                    "bbox": new_bbox,
                    "area": int(t_mask.sum()),
                    "iscrowd": ann.get("iscrowd", 0)
                })
                ann_id += 1
            img_id += 1

        print(f"Processed {ds} - {img_info['file_name']}")

out_json = "/home/hmanani/data/augmented/train/_annotations.coco.json"
with open(out_json, "w") as f:
    json.dump(merged, f)
print(f"\nDone! Total images: {len(merged['images'])}")
print(f"Total annotations: {len(merged['annotations'])}")
