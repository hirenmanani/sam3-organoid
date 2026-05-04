import json, os

paths = [
    '/home/hmanani/data/Organoid/D2-D6/train/_annotations.coco.json',
    '/home/hmanani/data/Organoid/D4-D6/train/_annotations.coco.json',
]

for path in paths:
    if not os.path.exists(path):
        print(f"NOT FOUND: {path}")
        continue
    with open(path) as f:
        d = json.load(f)
    img = d['images'][0]
    anns = [a for a in d['annotations'] if a['image_id'] == img['id']][:3]
    print(f"PATH: {path}")
    print(f"Images: {len(d['images'])}, Annotations: {len(d['annotations'])}")
    print(f"Image size: {img['width']}x{img['height']}")
    print(f"Sample boxes: {[a['bbox'] for a in anns]}")
    print()
