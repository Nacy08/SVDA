"""Model definitions for ExpI.

Reuses the loss components from ExpF_new (HybridRankingLoss /
GlobalFallbackRankingLoss / classifier head) without modifying the canonical
evaluation metrics. Adds an R-Drop-aware forward pass mode that returns
detached components so the training loop can compute the symmetric KL on
two stochastic dropout passes.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

# Import loss + head primitives from ExpF_new without polluting sys.path,
# so the training script's `from utils import ...` keeps resolving to expI/utils.py.
_EXPF_NEW_MODELS = Path(__file__).resolve().parents[1] / "shared_ranking" / "models.py"
_spec = importlib.util.spec_from_file_location("_expf_new_models", _EXPF_NEW_MODELS)
_expf_new_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_expf_new_models)

SVDAClassifierHead = _expf_new_models.SVDAClassifierHead
HybridRankingLoss = _expf_new_models.HybridRankingLoss
GlobalFallbackRankingLoss = _expf_new_models.GlobalFallbackRankingLoss


def _zero_like(t: torch.Tensor) -> torch.Tensor:
    return t.sum() * 0.0


class RobertaForSVDAClassificationI(nn.Module):
    """Classifier that returns logits + losses; supports two-pass R-Drop."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_labels: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        candidate_map: Optional[Dict[int, Sequence[int]]] = None,
        ranking_margin: float = 1.0,
        top_k_hybrid: int = 1,
        top_k_global: int = 1,
    ) -> None:
        super().__init__()
        self.encoder_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, local_files_only=True
        )
        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            config=self.encoder_config,
            local_files_only=True,
        )
        self.classifier = SVDAClassifierHead(
            input_dim=self.encoder_config.hidden_size,
            hidden_dim=hidden_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.hybrid_loss_fn = HybridRankingLoss(
            candidate_map=candidate_map or {},
            margin=ranking_margin,
            top_k=top_k_hybrid,
        )
        self.global_loss_fn = GlobalFallbackRankingLoss(
            margin=ranking_margin,
            top_k=top_k_global,
        )

    def load_encoder_state_dict_from_checkpoint(
        self, checkpoint: Dict[str, object]
    ) -> None:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                encoder_state[key[len("encoder.") :]] = value
        if not encoder_state:
            raise ValueError("No encoder weights found in warm-up checkpoint.")
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        if unexpected:
            raise ValueError(
                f"Unexpected encoder keys in warm-up checkpoint: {unexpected}"
            )
        if missing:
            print(f"[warmup] Encoder missing keys after load: {missing}")

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pooled = self.encode(input_ids, attention_mask)
        logits = self.classifier(pooled)
        result = {"logits": logits, "pooled": pooled}
        if labels is not None:
            ce_loss = self.ce_loss_fn(logits, labels)
            hybrid_loss = self.hybrid_loss_fn(logits, labels)
            global_loss = self.global_loss_fn(logits, labels)
            result.update(
                {
                    "ce_loss": ce_loss,
                    "hybrid_ranking_loss": hybrid_loss,
                    "global_fallback_ranking_loss": global_loss,
                }
            )
        return result
