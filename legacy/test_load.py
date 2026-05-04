import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

print("Loading model...")
model = build_sam3_image_model(
    checkpoint_path="./checkpoints/sam3.pt",
    device="cpu"
)
processor = Sam3Processor(model)
print("Model loaded!")
