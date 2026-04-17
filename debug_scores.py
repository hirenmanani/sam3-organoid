import torch
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model(checkpoint_path="./finetuned_weights.pt", device="cuda")
processor = Sam3Processor(model, device="cuda")
image = Image.open("images/ctl/231206_Rac1i_50uM_ctl_D2_(15).jpg").convert("RGB")

for prompt in ["organoid", "circular vesicle", "circular", "round cell", "bubble", "cell"]:
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    scores = output["scores"]
    scores_np = scores.cpu().numpy() if hasattr(scores, 'cpu') else np.array(scores)
    above = (scores_np > 0.05).sum()
    max_s = scores_np.max() if len(scores_np) > 0 else 0
    print(f"Prompt '{prompt}': {len(scores_np)} predictions, max={max_s:.4f}, above 0.05={above}")
