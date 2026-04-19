"""Freeze parameters that don't receive gradients in the organoid fine-tuning setup."""
FREEZE_PATTERNS = [
    'backbone.vision_backbone.convs.0.',
    'backbone.vision_backbone.convs.1.',
    'backbone.vision_backbone.convs.3.',
    'backbone.language_backbone.encoder.text_projection',
]

def freeze_unused_params(model):
    frozen = []
    for name, param in model.named_parameters():
        if any(name.startswith(p) or name == p for p in FREEZE_PATTERNS):
            param.requires_grad = False
            frozen.append(name)
    print(f"Frozen {len(frozen)} parameters to avoid DDP unused param error")
    return model
