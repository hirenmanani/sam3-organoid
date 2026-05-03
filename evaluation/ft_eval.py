import sys, os, json, math
sys.path.insert(0, '/home/hmanani/sam3')
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image, ImageEnhance
import numpy as np

GT_JSON = '/home/hmanani/data/new_dataset/train/_annotations.coco.json'
IMG_DIR = '/home/hmanani/data/new_dataset/train'
PROMPTS = ['cell','circular','round cell','bubble']
SCORE_THRESH = 0.25
IOU_MATCH = 0.50
MAX_IMAGES = 5

with open(GT_JSON) as f: coco = json.load(f)
id2file = {i['id']:i['file_name'] for i in coco['images']}
id2dim  = {i['id']:(i['width'],i['height']) for i in coco['images']}
gt_by_file = {}
for ann in coco['annotations']:
    fname = id2file[ann['image_id']]
    W,H = id2dim[ann['image_id']]
    raw = [float(str(v).strip("'\" ")) for v in ann['bbox']]
    box = [raw[0]*W,raw[1]*H,raw[2]*W,raw[3]*H] if all(v<=1.0 for v in raw) else raw
    if box[2] > 150:
        gt_by_file.setdefault(fname,[]).append(box)

eval_imgs = [f for f in os.listdir(IMG_DIR)
             if f.endswith('.jpg') and f in gt_by_file
             and len(gt_by_file[f]) >= 5][:MAX_IMAGES]
print(f'Eval images: {len(eval_imgs)}')
for f in eval_imgs:
    print(f'  {f}: {len(gt_by_file[f])} GT boxes')

def iou(b1,b2):
    x1,y1,w1,h1=b1; x2,y2,w2,h2=b2
    ix=max(0,min(x1+w1,x2+w2)-max(x1,x2))
    iy=max(0,min(y1+h1,y2+h2)-max(y1,y2))
    inter=ix*iy; union=w1*h1+w2*h2-inter
    return inter/union if union>0 else 0.0

def prf(preds, gts):
    matched=set(); tp=0
    for p in preds:
        best=0; bestj=-1
        for j,g in enumerate(gts):
            if j in matched: continue
            v=iou(p,g)
            if v>best: best=v; bestj=j
        if best>=IOU_MATCH: tp+=1; matched.add(bestj)
    fp=len(preds)-tp; fn=len(gts)-tp
    pr=tp/(tp+fp) if tp+fp>0 else 0.0
    rc=tp/(tp+fn) if tp+fn>0 else 0.0
    f1=2*pr*rc/(pr+rc) if pr+rc>0 else 0.0
    return pr,rc,f1,tp,fp,fn

for ckpt,label in [
    ('./checkpoints/sam3.pt','BASELINE'),
    ('./organoid_training_logs/checkpoints/checkpoint_5.pt','FINETUNED'),
]:
    print(f'\n=== {label} ===')
    model = build_sam3_image_model(checkpoint_path=ckpt, device='cuda')
    processor = Sam3Processor(model, device="cuda")
    tp_all=fp_all=fn_all=0

    for fname in eval_imgs:
        img = Image.open(os.path.join(IMG_DIR,fname)).convert('RGB')
        img = ImageEnhance.Contrast(img).enhance(2.5)
        W,H = img.size
        gts = gt_by_file[fname]
        preds = []

        for prompt in PROMPTS:
            inference_state = processor.set_image(img)
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)
            scores = output['scores']
            scores_np = scores.cpu().numpy() if hasattr(scores,'cpu') else np.array(scores)

            # Print all output keys first time
            if fname == eval_imgs[0] and prompt == 'cell' and label == 'BASELINE':
                print(f'  Output keys: {list(output.keys())}')
                for k,v in output.items():
                    if hasattr(v,'shape'): print(f'    {k}: {v.shape}')
                    elif hasattr(v,'__len__'): print(f'    {k}: len={len(v)}')

            # Try every possible key for boxes/masks
            boxes_found = None
            for key in ['boxes','pred_boxes','bboxes','masks','segmentation','pred_masks']:
                if key in output and output[key] is not None:
                    val = output[key]
                    if hasattr(val,'cpu'): val = val.cpu().numpy()
                    elif not isinstance(val, np.ndarray): val = np.array(val)
                    if len(val) > 0:
                        boxes_found = (key, val)
                        break

            for idx, score in enumerate(scores_np):
                if score < SCORE_THRESH: continue
                if boxes_found is not None:
                    key, val = boxes_found
                    if idx < len(val):
                        box = val[idx]
                        if key in ['masks','segmentation','pred_masks']:
                            # Convert mask to bbox
                            m = np.array(box).squeeze()
                            if m.ndim == 2:
                                rows = np.any(m, axis=1)
                                cols = np.any(m, axis=0)
                                if rows.any():
                                    r0,r1 = np.where(rows)[0][[0,-1]]
                                    c0,c1 = np.where(cols)[0][[0,-1]]
                                    preds.append([float(c0),float(r0),float(c1-c0),float(r1-r0)])
                        else:
                            b = box.flatten()
                            if len(b) >= 4:
                                # Handle both cx,cy,w,h and x,y,w,h formats
                                if b[0] <= 1.0 and b[1] <= 1.0:
                                    preds.append([b[0]*W,b[1]*H,b[2]*W,b[3]*H])
                                else:
                                    preds.append([float(b[0]),float(b[1]),float(b[2]),float(b[3])])
                else:
                    # No boxes — use score count only
                    preds.append([0,0,1,1])

        pr,rc,f1,tp,fp,fn = prf(preds,gts)
        tp_all+=tp; fp_all+=fp; fn_all+=fn
        print(f'  {fname[:35]} pred={len(preds)} gt={len(gts)} P={pr:.2f} R={rc:.2f} F1={f1:.2f}')

    pr=tp_all/(tp_all+fp_all) if tp_all+fp_all>0 else 0.0
    rc=tp_all/(tp_all+fn_all) if tp_all+fn_all>0 else 0.0
    f1=2*pr*rc/(pr+rc) if pr+rc>0 else 0.0
    print(f'\n  AGGREGATE: P={pr:.3f} R={rc:.3f} F1={f1:.3f}')
    print(f'  TP={tp_all} FP={fp_all} FN={fn_all}')
    del model, processor
