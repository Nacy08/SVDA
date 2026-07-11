"""Fast Gradient Method (FGM) adversarial perturbation on word embeddings.

Reference: AT-BERT (Zhu et al., SDU@AAAI-21 winning solution).
"""
from __future__ import annotations

from typing import Dict

import torch


class FGM:
    def __init__(
        self,
        model: torch.nn.Module,
        emb_name: str = "word_embeddings",
        epsilon: float = 1.0,
    ) -> None:
        self.model = model
        self.emb_name = emb_name
        self.epsilon = float(epsilon)
        self.backup: Dict[str, torch.Tensor] = {}

    def attack(self) -> None:
        for name, param in self.model.named_parameters():
            if not param.requires_grad or self.emb_name not in name:
                continue
            if param.grad is None:
                continue
            self.backup[name] = param.data.clone()
            grad_norm = torch.norm(param.grad)
            if grad_norm.item() == 0 or torch.isnan(grad_norm):
                continue
            r_at = self.epsilon * param.grad / grad_norm
            param.data.add_(r_at)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if not param.requires_grad or self.emb_name not in name:
                continue
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
