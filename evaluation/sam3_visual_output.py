import sys, os, json
sys.path.insert(0, '/home/hmanani/sam3')
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image, ImageEnhance, ImageDraw
import numpy as np

OUT = os.path.expanduser('~/sam3_visual_results')
os.makedirs(OUT, exist_ok=True)

# The 20 TellU test images — we need to find them
TELLU_NAMES = [
    'ctl_Day2_29','ctl Day 2_28','ctl_D4_25','ctl_Day4_16',
    'D0-D2_5','D0-D2_13','D0-D4_1','D0-D4_D2_1',
    'D0-D4_D2_15','D0-D4_D2_7','D0-D4_D4_7','D0-D6_D2_15',
    'D0-D6_D4_(12)','D0-D6_D4_(13)','D2-D6_D2_(13)','D2-D6_D2_(9)',
    'D2-D6_D6_(14)','D4-D6_D2_(13)','D4-D6_D4_(3)','D4-D6_D6_(9)'
]

# Find images on cluster
import glob
all_jpgs = glob.glob('/home/hmanani/data/**/*.jpg', recursive=True)
name_to_path = {}
for p in all_jpgs:
    stem = os.path.splitext(os.path.basename(p))[0]
    for t in TELLU_NAMES:
        if t.replace(' ','_') in stem or stem in t.replace(' ','_'):
            name_to_path[t] = p
            break

print(f'Found {len(name_to_path)} of 20 TellU images')

PROMPTS = ['cell','circular','round cell','bubble']
COLORS = [(0,255,200),(255,200,0),(255,100,0),(100,200,255)]

for ckpt_name, ckpt_path, label in [
    ('baseline', '/home/hmanani/sam3/checkpoints/sam3.pt', 'Baseline'),
]:
    print(f'\n=== SAM 3 {label} ===')
    model = build_sam3_image_model(checkpoint_path=ckpt_path, device='cuda')
    processor = Sam3Processor(model, device='cuda')
    
    for img_name, img_path in list(name_to_path.items())[:6]:  # 6 representative images
        print(f'  {img_name}...')
        img = Image.open(img_path).convert('RGB')
        img_contrast = ImageEnhance.Contrast(img).enhance(2.5)
        draw = ImageDraw.Draw(img_contrast)
        
        total = 0
        for pi, prompt in enumerate(PROMPTS):
            state = processor.set_image(img_contrast)
            output = processor.set_text_prompt(state=state, prompt=prompt)
            scores = output['scores']
            boxes = output.get('boxes', None)
            s = scores.cpu().numpy() if hasattr(scores,'cpu') else np.array(scores)
            
            if boxes is not None and len(s) > 0:
                boxes_np = boxes.cpu().numpy() if hasattr(boxes,'cpu') else np.array(boxes)
                for score, box in zip(s, boxes_np):
                    if score > 0.25:
                        x,y,w,h = box[0],box[1],box[2],box[3]
                        draw.rectangle([x,y,x+w,y+h], outline=COLORS[pi], width=3)
                        total += 1
        
        out_name = f'{img_name.replace(" ","_")}_sam3_{label.lower()}.jpg'
        img_contrast.save(os.path.join(OUT, out_name))
        print(f'    {total} detections saved')
    
    del model, processor

print(f'\nImages saved to {OUT}')
