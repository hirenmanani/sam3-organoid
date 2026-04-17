import json
import copy

INPUT = '/home/hmanani/data/augmented/train/_annotations.coco.json'
OUTPUT = '/home/hmanani/data/augmented/train/_annotations_normalized.coco.json'

with open(INPUT) as f:
    data = json.load(f)

img_sizes = {img['id']: (img['width'], img['height']) for img in data['images']}

fixed = 0
for ann in data['annotations']:
    w, h = img_sizes[ann['image_id']]
    bbox = ann['bbox']  # [x, y, w, h] in pixels
    # Check if already normalized (all values <= 1)
    if all(v <= 1.0 for v in bbox):
        continue
    # Normalize to 0-1
    ann['bbox'] = [
        bbox[0] / w,  # x
        bbox[1] / h,  # y
        bbox[2] / w,  # width
        bbox[3] / h   # height
    ]
    fixed += 1

with open(OUTPUT, 'w') as f:
    json.dump(data, f)

print(f"Fixed {fixed} annotations")
print(f"Sample normalized bbox: {data['annotations'][0]['bbox']}")
print(f"Saved to {OUTPUT}")
