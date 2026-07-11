from typing import Dict, Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


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
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels)
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
