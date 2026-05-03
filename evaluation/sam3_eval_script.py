import os, sys, json, math
import numpy as np
from PIL import Image, ImageEnhance
import torch
sys.path.insert(0, os.path.expanduser("~/sam3"))

BASELINE_CKPT  = os.path.expanduser("~/sam3/checkpoints/sam3.pt")
FINETUNED_CKPT = os.path.expanduser("~/sam3/organoid_training_logs/checkpoints/checkpoint_5.pt")
GT_JSON        = "/home/hmanani/data/new_dataset/train/_annotations.coco.json"
IMG_DIR        = "/home/hmanani/data/new_dataset/train"
OUT_DIR        = os.path.expanduser("~/sam3_eval_output")
PROMPTS        = ["cell", "circular", "round cell", "bubble"]
SCORE_THRESH   = 0.25
CIRC_MIN       = 0.55
AREA_MIN, AREA_MAX = 200, 80000
IOU_DEDUP      = 0.40
IOU_MATCH      = 0.50
MAX_IMAGES     = 30
os.makedirs(OUT_DIR, exist_ok=True)

def circularity(m):
    import cv2
    c, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not c: return 0.0
    c = max(c, key=cv2.contourArea)
    a = cv2.contourArea(c); p = cv2.arcLength(c, True)
    return (4*math.pi*a/(p**2)) if p>0 else 0.0

def mask_to_box(m):
    rows=np.any(m,axis=1); cols=np.any(m,axis=0)
    if not rows.any(): return None
    r0,r1=np.where(rows)[0][[0,-1]]; c0,c1=np.where(cols)[0][[0,-1]]
    return [int(c0),int(r0),int(c1-c0+1),int(r1-r0+1)]

def iou(b1,b2):
    x1,y1,w1,h1=b1; x2,y2,w2,h2=b2
    ix=max(0,min(x1+w1,x2+w2)-max(x1,x2))
    iy=max(0,min(y1+h1,y2+h2)-max(y1,y2))
    inter=ix*iy; union=w1*h1+w2*h2-inter
    return inter/union if union>0 else 0.0

def dedup(boxes):
    boxes=sorted(boxes,key=lambda x:x["score"],reverse=True)
    kept=[]
    for b in boxes:
        if all(iou(b["bbox"],k["bbox"])<IOU_DEDUP for k in kept):
            kept.append(b)
    return kept

def prf(preds, gts):
    matched=set(); tp=0
    for p in preds:
        best_iou=0; best_j=-1
        for j,g in enumerate(gts):
            if j in matched: continue
            v=iou(p,g)
            if v>best_iou: best_iou=v; best_j=j
        if best_iou>=IOU_MATCH: tp+=1; matched.add(best_j)
    fp=len(preds)-tp; fn=len(gts)-tp
    pr=tp/(tp+fp) if tp+fp>0 else 0.0
    rc=tp/(tp+fn) if tp+fn>0 else 0.0
    f1=2*pr*rc/(pr+rc) if pr+rc>0 else 0.0
    return {"tp":tp,"fp":fp,"fn":fn,"precision":pr,"recall":rc,"f1":f1,
            "n_pred":len(preds),"n_gt":len(gts)}

# Load GT
print("Loading ground truth...")
with open(GT_JSON) as f: coco=json.load(f)
id2file={i["id"]:i["file_name"] for i in coco["images"]}
id2dim ={i["id"]:(i["width"],i["height"]) for i in coco["images"]}
gt_by_file={}
for ann in coco["annotations"]:
    fname=id2file[ann["image_id"]]
    W,H=id2dim[ann["image_id"]]
    raw=[float(str(v).strip("'\" ")) for v in ann["bbox"]]
    box=[raw[0]*W,raw[1]*H,raw[2]*W,raw[3]*H] if all(v<=1.0 for v in raw) else raw
    gt_by_file.setdefault(fname,[]).append(box)
print(f"GT images: {len(gt_by_file)}, annotations: {len(coco['annotations'])}")

# Select eval images
eval_imgs=sorted([f for f in os.listdir(IMG_DIR)
                  if f.lower().endswith((".jpg",".jpeg",".png"))
                  and f in gt_by_file])[:MAX_IMAGES]
print(f"Evaluating on {len(eval_imgs)} images")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

device="cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

def infer(img_path, processor):
    img=Image.open(img_path).convert("RGB")
    img=ImageEnhance.Contrast(img).enhance(2.5)
    W,H=img.size
    boxes=[]
    for prompt in PROMPTS:
        try:
            state=processor.set_image(img)
            output=processor.set_text_prompt(state=state, prompt=prompt)
            scores=output["scores"]
            masks=output.get("masks", output.get("segmentation", None))
            for idx, score in enumerate(scores):
                if float(score) < SCORE_THRESH: continue
                if masks is not None:
                    try:
                        m=np.array(masks[idx]).squeeze()
                        if m.ndim!=2: raise ValueError
                        area=m.sum()
                        if area<AREA_MIN or area>AREA_MAX: continue
                        if circularity(m)<CIRC_MIN: continue
                        b=mask_to_box(m)
                        if b: boxes.append({"bbox":b,"score":float(score),"prompt":prompt})
                    except:
                        # fallback: use bounding box from output if available
                        if "boxes" in output:
                            raw_box=output["boxes"][idx]
                            b=[float(v) for v in raw_box]
                            boxes.append({"bbox":b,"score":float(score),"prompt":prompt})
                else:
                    if "boxes" in output:
                        raw_box=output["boxes"][idx]
                        b=[float(v) for v in raw_box]
                        boxes.append({"bbox":b,"score":float(score),"prompt":prompt})
        except Exception as e:
            print(f"    prompt '{prompt}' error: {e}")
            continue
    return dedup(boxes)

results={}
for ckpt,label,is_ft in [
    (BASELINE_CKPT,"sam3_baseline",False),
    (FINETUNED_CKPT,"sam3_finetuned",True)]:
    print(f"\n=== {label} ===")
    if not os.path.exists(ckpt):
        print(f"SKIP: {ckpt} not found"); continue
    print(f"Loading checkpoint directly: {ckpt}")
    model=build_sam3_image_model(checkpoint_path=ckpt, device=device)
    model.to(device); model.eval()
    processor=Sam3Processor(model, device=device)
    per_img={}; tp_all=fp_all=fn_all=0
    for i,fname in enumerate(eval_imgs):
        gts=gt_by_file.get(fname,[])
        preds=[b["bbox"] for b in infer(os.path.join(IMG_DIR,fname),processor)]
        m=prf(preds,gts)
        per_img[fname]=m
        tp_all+=m["tp"]; fp_all+=m["fp"]; fn_all+=m["fn"]
        print(f"  [{i+1:2d}/{len(eval_imgs)}] {fname[:35]:35s} "
              f"pred={m['n_pred']:3d} gt={m['n_gt']:3d} "
              f"P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")
    pr=tp_all/(tp_all+fp_all) if tp_all+fp_all>0 else 0.0
    rc=tp_all/(tp_all+fn_all) if tp_all+fn_all>0 else 0.0
    f1=2*pr*rc/(pr+rc) if pr+rc>0 else 0.0
    ps=[v["precision"] for v in per_img.values()]
    rs=[v["recall"] for v in per_img.values()]
    fs=[v["f1"] for v in per_img.values()]
    agg={"micro_precision":round(pr,4),"micro_recall":round(rc,4),"micro_f1":round(f1,4),
         "macro_precision":round(sum(ps)/len(ps),4),"macro_recall":round(sum(rs)/len(rs),4),
         "macro_f1":round(sum(fs)/len(fs),4),
         "total_tp":tp_all,"total_fp":fp_all,"total_fn":fn_all,"n_images":len(eval_imgs)}
    print(f"\n  AGGREGATE: P={pr:.3f} R={rc:.3f} F1={f1:.3f}")
    print(f"  TP={tp_all} FP={fp_all} FN={fn_all}")
    results[label]={"per_image":per_img,"aggregate":agg}
    del model, processor
    if torch.cuda.is_available(): torch.cuda.empty_cache()

out=os.path.join(OUT_DIR,"eval_results.json")
with open(out,"w") as f: json.dump(results,f,indent=2)
print(f"\nSaved -> {out}")
print("Upload eval_results.json here for final figures.")
