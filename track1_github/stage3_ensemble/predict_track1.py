"""Generate Track 1 submission predictions with the final I+J+K ensemble.

The output format is one JSON object per input line:
{"Value": "<predicted label>"}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CODE_DIR = _PROJECT_ROOT / "stage2_roberta_finetuning" / "model_i"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from data import build_label_mappings, extract_examples, load_jsonl_records
from metrics import verify_canonical_metrics
from models import RobertaForSVDAClassificationI
from utils import load_yaml, resolve_device, set_seed


DEFAULT_INPUT = str(_PROJECT_ROOT / "data" / "track1.jsonl")
DEFAULT_OUTPUT = str(Path(__file__).resolve().parent / "track1.pred.jsonl")
DEFAULT_CONFIG = str(_CODE_DIR / "config.yaml")
DEFAULT_CHECKPOINTS = str(Path(__file__).resolve().parent / "final_3model_IJK.json")


class Track1Dataset(Dataset):
    def __init__(self, records: Sequence[Mapping[str, object]], text_field: str) -> None:
        self.texts: List[str] = []
        for index, record in enumerate(records):
            if text_field not in record:
                raise KeyError(f"Missing text field '{text_field}' in input line {index + 1}")
            text = str(record[text_field]).strip()
            if not text:
                raise ValueError(f"Empty text field '{text_field}' in input line {index + 1}")
            self.texts.append(text)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> str:
        return self.texts[index]


class Track1Collator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, batch: Sequence[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


def load_model(checkpoint_path: str, config: Mapping[str, object], device: torch.device) -> torch.nn.Module:
    model_config = config["model"]
    model = RobertaForSVDAClassificationI(
        pretrained_model_name_or_path=str(model_config["pretrained_model_name_or_path"]),
        num_labels=int(model_config["num_labels"]),
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
        candidate_map={},
        ranking_margin=1.0,
        top_k_hybrid=1,
        top_k_global=1,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    kept = {
        key: value
        for key, value in state_dict.items()
        if key.startswith(("encoder.", "classifier."))
    }
    missing, _ = model.load_state_dict(kept, strict=False)
    required_missing = [key for key in missing if key.startswith(("encoder.", "classifier."))]
    if required_missing:
        raise RuntimeError(
            f"Encoder/classifier weights missing from {checkpoint_path}: {required_missing[:5]}"
        )
    model.to(device)
    model.eval()
    return model


def collect_logits(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )["logits"]
            chunks.append(logits.cpu())
    return torch.cat(chunks, dim=0)


def soft_vote(
    logits_per_model: Sequence[torch.Tensor],
    temperature: float,
    weights: Sequence[float],
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    if len(logits_per_model) != len(weights):
        raise ValueError("Weights must match logits.")
    if any(weight <= 0 for weight in weights):
        raise ValueError("All weights must be positive.")

    total_weight = float(sum(weights))
    probs_sum = None
    for logits, weight in zip(logits_per_model, weights):
        probs = F.softmax(logits / float(temperature), dim=-1) * (float(weight) / total_weight)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    if probs_sum is None:
        raise ValueError("No checkpoint logits were collected.")
    return probs_sum


def write_submission(path: str, predicted_ids: Sequence[int], id2label: Mapping[int, str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        for predicted_id in predicted_ids:
            record = {"Value": id2label[int(predicted_id)]}
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Track 1 labels with final I+J+K ensemble.")
    parser.add_argument("--input_jsonl", default=DEFAULT_INPUT)
    parser.add_argument("--output_jsonl", default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoints_json", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--text_field", default="Consistent Value Response")
    parser.add_argument("--label_field", default="Value")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=None, help="Override config device, for example cuda or cpu.")
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_canonical_metrics()
    config = load_yaml(args.config)
    set_seed(args.seed)
    device = resolve_device(args.device or str(config["runtime"]["device"]))

    records = load_jsonl_records(args.input_jsonl)
    train_records = load_jsonl_records(str(config["data"]["train_path"]))
    train_examples = extract_examples(
        train_records,
        text_field=args.text_field,
        label_field=args.label_field,
        use_question=False,
    )
    id2label = build_label_mappings(train_examples)["id2label"]

    tokenizer = AutoTokenizer.from_pretrained(
        str(config["model"]["pretrained_model_name_or_path"]),
        local_files_only=True,
    )
    dataloader = DataLoader(
        Track1Dataset(records, args.text_field),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=Track1Collator(tokenizer, int(config["model"]["max_length"])),
    )

    specs = json.loads(Path(args.checkpoints_json).read_text(encoding="utf-8"))
    logits_per_model: List[torch.Tensor] = []
    weights: List[float] = []
    for spec in specs:
        label = str(spec.get("label", spec["path"]))
        print(f"[load] {label}")
        model = load_model(str(spec["path"]), config, device)
        logits_per_model.append(collect_logits(model, dataloader, device))
        weights.append(float(spec.get("weight", 1.0)))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    probs = soft_vote(logits_per_model, args.temperature, weights)
    predicted_ids = probs.argmax(dim=-1).tolist()
    if len(predicted_ids) != len(records):
        raise RuntimeError(f"Prediction count mismatch: {len(predicted_ids)} vs {len(records)}")

    write_submission(args.output_jsonl, predicted_ids, id2label)
    print(f"[output] wrote {len(predicted_ids)} predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()
