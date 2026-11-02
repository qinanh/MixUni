"""Ways of combining the per-property losses (``train.loss_weighting``) and PCGrad."""

from __future__ import annotations

import random

import torch
from torch import nn


class UncertaintyWeighting(nn.Module):
    """Kendall et al. (2018): sum_i exp(-2 s_i) L_i + s_i with learned s_i = log sigma_i."""

    def __init__(self, num_tasks):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses, active):
        total = None
        for i, (loss, on) in enumerate(zip(losses, active)):
            if on:
                term = torch.exp(-2 * self.log_sigma[i]) * loss + self.log_sigma[i]
                total = term if total is None else total + term
        return total


class GradNormWeighting(nn.Module):
    """Learned task weights w = K * softmax(a), the parameterisation used by GradNorm
    (Chen et al., 2018). Only the weighted sum is minimised; there is no separate
    gradient-norm balancing step."""

    def __init__(self, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_weights = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses, active):
        w = torch.softmax(self.log_weights, dim=0) * self.num_tasks
        terms = [w[i] * loss for i, (loss, on) in enumerate(zip(losses, active)) if on]
        return torch.stack(terms).sum() if terms else None


def build_weighting(name, num_tasks):
    """None for "plain" (one MSE over all samples of the batch)."""
    if name == "plain":
        return None
    return {"uncertainty": UncertaintyWeighting, "gradnorm": GradNormWeighting}[name](num_tasks)


class PCGrad:
    """Yu et al. (2020): project each task gradient off the gradients it conflicts with."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, task_loss, num_tasks, params, *, grad_clip=None, extra_loss=None):
        """``task_loss(i)`` returns the loss of task i (or None if the batch has no
        sample of it); ``extra_loss()`` is added after projection (auxiliary losses)."""
        params = [p for p in params if p.requires_grad]
        grads = {}
        for i in range(num_tasks):
            self.optimizer.zero_grad(set_to_none=True)
            loss = task_loss(i)
            if loss is None:
                continue
            loss.backward()
            grads[i] = torch.cat([
                p.grad.detach().cpu().reshape(-1) if p.grad is not None
                else torch.zeros(p.numel(), dtype=p.dtype) for p in params
            ])

        order = list(grads)
        random.shuffle(order)
        projected = []
        for i in order:
            g = grads[i].clone()
            for j in order:
                if j != i:
                    dot = torch.dot(g, grads[j])
                    if dot < 0:
                        g = g - dot / torch.dot(grads[j], grads[j]).clamp_min(1e-12) * grads[j]
            projected.append(g)

        self.optimizer.zero_grad(set_to_none=True)
        if not projected:
            return
        flat = torch.stack(projected).mean(dim=0)
        offset = 0
        for p in params:
            p.grad = flat[offset: offset + p.numel()].view_as(p).to(p.device)
            offset += p.numel()
        if extra_loss is not None:
            extra_loss().backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        self.optimizer.step()
