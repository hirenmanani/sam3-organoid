# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Modified for Mac compatibility - replaced triton kernel with scipy fallback

import torch
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt as scipy_edt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def edt_triton(data: torch.Tensor):
    """
    Computes the Euclidean Distance Transform (EDT) of a batch of binary images.
    Mac-compatible fallback using scipy instead of triton.

    Args:
        data: A tensor of shape (B, H, W) representing a batch of binary images.

    Returns:
        A tensor of the same shape as data containing the EDT.
    """
    assert data.dim() == 3
    device = data.device
    data_np = data.cpu().numpy().astype(bool)
    B = data_np.shape[0]
    results = []
    for i in range(B):
        # scipy edt: distance to nearest zero pixel
        # data is 1 where object is, 0 elsewhere
        # we want distance to nearest 0, so invert
        dist = scipy_edt(data_np[i])
        results.append(dist)
    result = np.stack(results, axis=0)
    return torch.tensor(result, dtype=torch.float32, device=device)
