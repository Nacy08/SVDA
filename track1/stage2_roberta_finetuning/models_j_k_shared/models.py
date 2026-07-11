"""Model + losses for ExpJ.

Adds:
- `FocalLoss` (Lin et al., ICCV 2017): down-weights well-classified
  examples by (1 - p_t)^gamma. Replaces CE.
- `SiblingAwareHybridRanking`: extends HybridRankingLoss so that when the
  current hard-negative happens to be a "sibling" of the gold class
  (e.g., Self-direction–action vs Self-direction–thought), the margin is
  raised from `default_margin` to `sibling_margin` — pushing the model
  to discriminate within-family pairs more aggressively.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel

_EXPF_NEW_MODELS = Path(__file__).resolve().parents[1] / "shared_ranking" / "models.py"
_spec = importlib.util.spec_from_file_location("_expf_new_models", _EXPF_NEW_MODELS)
_expf_new_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_expf_new_models)

SVDAClassifierHead = _expf_new_models.SVDAClassifierHead
GlobalFallbackRankingLoss = _expf_new_models.GlobalFallbackRankingLoss


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        gold_log_p = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        gold_p = gold_log_p.exp()
        focal_weight = (1.0 - gold_p) ** self.gamma
        return (-focal_weight * gold_log_p).mean()


class SiblingAwareHybridRanking(nn.Module):
    """Like HybridRankingLoss but the hinge margin is per-pair.

    For each anchor (row), we look at the candidate negatives in
    `candidate_map[gold]`, pick the top-K hardest by logit, and apply a
    hinge:  max(0, m - gold_logit + neg_logit). If (gold, neg) is a sibling
    pair, m = sibling_margin; otherwise m = default_margin.
    """

    def __init__(
        self,
        candidate_map: Dict[int, Sequence[int]],
        sibling_pairs: Optional[Iterable[Tuple[int, int]]] = None,
        default_margin: float = 1.0,
        sibling_margin: float = 1.5,
        top_k: int = 1,
    ) -> None:
        super().__init__()
        self.default_margin = float(default_margin)
        self.sibling_margin = float(sibling_margin)
        self.top_k = int(top_k)
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        normalized: Dict[int, Tuple[int, ...]] = {}
        for label_id, ids in candidate_map.items():
            gold = int(label_id)
            deduped: List[int] = []
            for cand in ids:
                cand = int(cand)
                if cand == gold or cand in deduped:
                    continue
                deduped.append(cand)
            normalized[gold] = tuple(deduped)
        self.candidate_map = normalized
        self.sibling_set: Set[Tuple[int, int]] = set()
        if sibling_pairs:
            for a, b in sibling_pairs:
                if int(a) == int(b):
                    continue
                self.sibling_set.add((int(a), int(b)))
                self.sibling_set.add((int(b), int(a)))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        for row_index, gold_label in enumerate(labels.detach().cpu().tolist()):
            gold = int(gold_label)
            cands = self.candidate_map.get(gold, ())
            if not cands:
                continue
            cand_list = list(cands)
            gold_logit = logits[row_index, gold]
            cand_logits = logits[row_index, cand_list]
            top_k = min(self.top_k, cand_logits.numel())
            top_vals, top_idx = torch.topk(cand_logits, k=top_k)
            row_losses = []
            for k in range(top_k):
                local_idx = int(top_idx[k].item())
                cand_label = cand_list[local_idx]
                margin = self.sibling_margin if (gold, cand_label) in self.sibling_set else self.default_margin
                hinge = torch.relu(margin - gold_logit + top_vals[k])
                row_losses.append(hinge)
            losses.append(torch.stack(row_losses).mean())
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


class RobertaForSVDAClassificationJ(nn.Module):
    """Identical encoder + head as expI's model, but exposes focal + sibling
    margin ranking losses inside forward."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_labels: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        focal_gamma: float = 2.0,
        candidate_map: Optional[Dict[int, Sequence[int]]] = None,
        sibling_pairs: Optional[Iterable[Tuple[int, int]]] = None,
        default_margin: float = 1.0,
        sibling_margin: float = 1.5,
        top_k_hybrid: int = 1,
        top_k_global: int = 1,
    ) -> None:
        super().__init__()
        self.encoder_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path, config=self.encoder_config, local_files_only=True
        )
        self.classifier = SVDAClassifierHead(
            input_dim=self.encoder_config.hidden_size,
            hidden_dim=hidden_dim, num_labels=num_labels, dropout=dropout,
        )
        self.focal_loss_fn = FocalLoss(gamma=focal_gamma)
        self.hybrid_loss_fn = SiblingAwareHybridRanking(
            candidate_map=candidate_map or {},
            sibling_pairs=sibling_pairs,
            default_margin=default_margin,
            sibling_margin=sibling_margin,
            top_k=top_k_hybrid,
        )
        self.global_loss_fn = GlobalFallbackRankingLoss(margin=default_margin, top_k=top_k_global)

    def load_encoder_state_dict_from_checkpoint(self, checkpoint: Dict[str, object]) -> None:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                encoder_state[key[len("encoder.") :]] = value
        if not encoder_state:
            raise ValueError("No encoder weights found in warm-up checkpoint.")
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected encoder keys: {unexpected}")
        if missing:
            print(f"[warmup] Encoder missing keys after load: {missing}")

    def encode(self, input_ids, attention_mask):
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
            result["focal_loss"] = self.focal_loss_fn(logits, labels)
            result["ce_loss"] = result["focal_loss"]  # alias so existing logging code works
            result["hybrid_ranking_loss"] = self.hybrid_loss_fn(logits, labels)
            result["global_fallback_ranking_loss"] = self.global_loss_fn(logits, labels)
        return result
