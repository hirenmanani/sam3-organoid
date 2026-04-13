# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Modified for Mac compatibility - replaced triton kernel with pure PyTorch

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="none"):
    """
    Pure PyTorch implementation of sigmoid focal loss.
    Equivalent to the triton kernel version.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * ce_loss * ((1 - p_t) ** gamma)

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class SigmoidFocalLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, alpha=0.25, gamma=2.0):
        loss = sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction="sum")
        ctx.save_for_backward(inputs, targets)
        ctx.alpha = alpha
        ctx.gamma = gamma
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        inputs, targets = ctx.saved_tensors
        alpha, gamma = ctx.alpha, ctx.gamma
        prob = inputs.sigmoid()
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        focal_weight = alpha_t * gamma * ((1 - p_t) ** (gamma - 1))
        grad = grad_output * (focal_weight * ce_loss * (2 * prob - 1) + alpha_t * ((1 - p_t) ** gamma) * (prob - targets))
        return grad, None, None, None


triton_sigmoid_focal_loss = SigmoidFocalLoss.apply


class SigmoidFocalLossReduced(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, alpha=0.25, gamma=2.0):
        loss = sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction="sum")
        ctx.save_for_backward(inputs, targets)
        ctx.alpha = alpha
        ctx.gamma = gamma
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        inputs, targets = ctx.saved_tensors
        alpha, gamma = ctx.alpha, ctx.gamma
        prob = inputs.sigmoid()
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        focal_weight = alpha_t * gamma * ((1 - p_t) ** (gamma - 1))
        grad = grad_output * (focal_weight * ce_loss * (2 * prob - 1) + alpha_t * ((1 - p_t) ** gamma) * (prob - targets))
        n = inputs.numel()
        return grad / n, None, None, None


triton_sigmoid_focal_loss_reduce = SigmoidFocalLossReduced.apply
