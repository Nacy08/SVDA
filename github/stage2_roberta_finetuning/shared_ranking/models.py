from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


def _zero_like_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits.sum() * 0.0


class ConfusableRankingLoss(nn.Module):
    def __init__(self, confusable_pairs: Sequence[Tuple[int, int]], margin: float = 1.0) -> None:
        super().__init__()
        self.margin = float(margin)
        confusables: Dict[int, List[int]] = {}
        for left, right in confusable_pairs:
            left_id = int(left)
            right_id = int(right)
            if left_id == right_id:
                continue
            confusables.setdefault(left_id, [])
            confusables.setdefault(right_id, [])
            if right_id not in confusables[left_id]:
                confusables[left_id].append(right_id)
            if left_id not in confusables[right_id]:
                confusables[right_id].append(left_id)
        self.confusables = {label_id: tuple(ids) for label_id, ids in confusables.items()}

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        losses = []
        for row_index, gold_label in enumerate(labels.detach().cpu().tolist()):
            confusable_ids = self.confusables.get(int(gold_label), ())
            if not confusable_ids:
                continue
            gold_logit = logits[row_index, int(gold_label)]
            confusable_logits = logits[row_index, list(confusable_ids)]
            pair_losses = torch.relu(self.margin - gold_logit + confusable_logits)
            losses.append(pair_losses.mean())
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


class HybridRankingLoss(nn.Module):
    def __init__(self, candidate_map: Dict[int, Sequence[int]], margin: float = 1.0, top_k: int = 1) -> None:
        super().__init__()
        self.margin = float(margin)
        self.top_k = int(top_k)
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        normalized: Dict[int, Tuple[int, ...]] = {}
        for label_id, candidate_ids in candidate_map.items():
            gold_id = int(label_id)
            deduped: List[int] = []
            for candidate_id in candidate_ids:
                candidate_id = int(candidate_id)
                if candidate_id == gold_id or candidate_id in deduped:
                    continue
                deduped.append(candidate_id)
            normalized[gold_id] = tuple(deduped)
        self.candidate_map = normalized

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        losses = []
        for row_index, gold_label in enumerate(labels.detach().cpu().tolist()):
            gold_id = int(gold_label)
            candidate_ids = self.candidate_map.get(gold_id, ())
            if not candidate_ids:
                continue
            gold_logit = logits[row_index, gold_id]
            candidate_logits = logits[row_index, list(candidate_ids)]
            top_k = min(self.top_k, candidate_logits.numel())
            hard_logits = torch.topk(candidate_logits, k=top_k).values
            losses.append(torch.relu(self.margin - gold_logit + hard_logits).mean())
        if not losses:
            return _zero_like_logits(logits)
        return torch.stack(losses).mean()


class GlobalFallbackRankingLoss(nn.Module):
    def __init__(self, margin: float = 1.0, top_k: int = 1) -> None:
        super().__init__()
        self.margin = float(margin)
        self.top_k = int(top_k)
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        losses = []
        num_labels = logits.size(-1)
        for row_index, gold_label in enumerate(labels.detach().cpu().tolist()):
            gold_id = int(gold_label)
            candidate_ids = [label_id for label_id in range(num_labels) if label_id != gold_id]
            if not candidate_ids:
                continue
            gold_logit = logits[row_index, gold_id]
            candidate_logits = logits[row_index, candidate_ids]
            top_k = min(self.top_k, candidate_logits.numel())
            hard_logits = torch.topk(candidate_logits, k=top_k).values
            losses.append(torch.relu(self.margin - gold_logit + hard_logits).mean())
        if not losses:
            return _zero_like_logits(logits)
        return torch.stack(losses).mean()


def compute_svda_total_loss(
    ce_loss: torch.Tensor,
    hybrid_ranking_loss: torch.Tensor,
    global_fallback_ranking_loss: torch.Tensor,
    lambda_hybrid: float,
    lambda_global: float,
    epoch: Optional[int],
    start_epoch: int,
) -> torch.Tensor:
    current_epoch = int(epoch) if epoch is not None else int(start_epoch)
    if current_epoch < int(start_epoch):
        return ce_loss
    return ce_loss + float(lambda_hybrid) * hybrid_ranking_loss + float(lambda_global) * global_fallback_ranking_loss


class SVDAClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_labels: int = 19,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout_in = nn.Dropout(dropout)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout_out = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout_in(pooled_output)
        hidden = self.linear1(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout_out(hidden)
        return self.linear2(hidden)


class RobertaForSVDAClassification(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_labels: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        confusable_pairs: Optional[Sequence[Tuple[int, int]]] = None,
        lambda_conf: float = 0.0,
        confusable_margin: float = 1.0,
        candidate_map: Optional[Dict[int, Sequence[int]]] = None,
        lambda_hybrid: float = 0.0,
        lambda_global: float = 0.0,
        ranking_margin: float = 1.0,
        top_k_hybrid: int = 1,
        top_k_global: int = 1,
        start_epoch: int = 1,
    ) -> None:
        super().__init__()
        self.encoder_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=True,
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
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_conf = float(lambda_conf)
        self.confusable_loss_fn = ConfusableRankingLoss(
            confusable_pairs=confusable_pairs or [],
            margin=confusable_margin,
        )
        self.lambda_hybrid = float(lambda_hybrid)
        self.lambda_global = float(lambda_global)
        self.start_epoch = int(start_epoch)
        self.hybrid_loss_fn = HybridRankingLoss(
            candidate_map=candidate_map or {},
            margin=ranking_margin,
            top_k=top_k_hybrid,
        )
        self.global_fallback_loss_fn = GlobalFallbackRankingLoss(
            margin=ranking_margin,
            top_k=top_k_global,
        )

    def load_encoder_state_dict_from_checkpoint(self, checkpoint: Dict[str, object]) -> None:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                encoder_state_dict[key[len("encoder.") :]] = value
        if not encoder_state_dict:
            raise ValueError("No encoder weights found in warm-up checkpoint.")
        missing, unexpected = self.encoder.load_state_dict(encoder_state_dict, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected encoder keys in warm-up checkpoint: {unexpected}")
        if missing:
            print(f"[warmup] Encoder missing keys after load: {missing}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        current_epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        result = {"logits": logits}
        if labels is not None:
            ce_loss = self.loss_fn(logits, labels)
            if self.lambda_hybrid or self.lambda_global:
                hybrid_ranking_loss = self.hybrid_loss_fn(logits, labels)
                global_fallback_ranking_loss = self.global_fallback_loss_fn(logits, labels)
                confusable_ranking_loss = _zero_like_logits(logits)
                total_loss = compute_svda_total_loss(
                    ce_loss=ce_loss,
                    hybrid_ranking_loss=hybrid_ranking_loss,
                    global_fallback_ranking_loss=global_fallback_ranking_loss,
                    lambda_hybrid=self.lambda_hybrid,
                    lambda_global=self.lambda_global,
                    epoch=current_epoch,
                    start_epoch=self.start_epoch,
                )
            else:
                confusable_ranking_loss = self.confusable_loss_fn(logits, labels)
                hybrid_ranking_loss = _zero_like_logits(logits)
                global_fallback_ranking_loss = _zero_like_logits(logits)
                total_loss = ce_loss + self.lambda_conf * confusable_ranking_loss
            result["ce_loss"] = ce_loss
            result["confusable_ranking_loss"] = confusable_ranking_loss
            result["hybrid_ranking_loss"] = hybrid_ranking_loss
            result["global_fallback_ranking_loss"] = global_fallback_ranking_loss
            result["loss"] = total_loss
        return result


class FULCRAValueHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        num_value_dims: int = 11,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout_in = nn.Dropout(dropout)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout_out = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, num_value_dims)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout_in(pooled_output)
        hidden = self.linear1(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout_out(hidden)
        return self.linear2(hidden)


class RobertaForFULCRARegression(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_value_dims: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=True,
        )
        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            config=self.encoder_config,
            local_files_only=True,
        )
        self.value_head = FULCRAValueHead(
            input_dim=self.encoder_config.hidden_size,
            hidden_dim=hidden_dim,
            num_value_dims=num_value_dims,
            dropout=dropout,
        )
        self.loss_fn = nn.MSELoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        value_scores = self.value_head(pooled_output)
        result = {"value_scores": value_scores}
        if labels is not None:
            result["loss"] = self.loss_fn(value_scores, labels)
        return result
