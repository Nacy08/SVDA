"""Data loading + WeightedRandomSampler for class-balanced training.

Each example's sampling weight is `1 / sqrt(class_count)`, which boosts
small-support classes (Universalism–tolerance 68, Security–societal 70)
to be sampled ~2-2.5x as often as large classes (Stimulation 400). This
also raises the joint probability of sibling co-occurrence within a batch
without requiring an explicit sibling-pair sampler.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


@dataclass
class SVDAExample:
    text: str
    label: str


class SVDADataset(Dataset):
    def __init__(self, examples: Sequence[SVDAExample], label2id: Dict[str, int]) -> None:
        self.examples = list(examples)
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        ex = self.examples[idx]
        return {"text": ex.text, "label": ex.label, "labels": self.label2id[ex.label]}


def load_jsonl_records(path: str) -> List[Dict[str, object]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    records: List[Dict[str, object]] = []
    with file_path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"No records found in {file_path}")
    return records


def extract_examples(records: Sequence[Dict[str, object]], text_field: str, label_field: str) -> List[SVDAExample]:
    out: List[SVDAExample] = []
    for i, r in enumerate(records):
        if text_field not in r or label_field not in r:
            raise KeyError(f"Record {i} missing required fields")
        t = str(r[text_field]).strip()
        l = str(r[label_field]).strip()
        if not t or not l:
            raise ValueError(f"Record {i} empty text/label")
        out.append(SVDAExample(text=t, label=l))
    return out


def build_label_mappings(examples: Sequence[SVDAExample]) -> Dict[str, Dict[object, object]]:
    labels = sorted({ex.label for ex in examples})
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return {"label2id": label2id, "id2label": id2label}


def build_svda_datasets(config: Dict[str, object]) -> Dict[str, object]:
    data_config = config["data"]
    train_examples = extract_examples(
        load_jsonl_records(str(data_config["train_path"])),
        text_field=str(data_config["text_field"]),
        label_field=str(data_config["label_field"]),
    )
    dev_examples = extract_examples(
        load_jsonl_records(str(data_config["dev_path"])),
        text_field=str(data_config["text_field"]),
        label_field=str(data_config["label_field"]),
    )
    m = build_label_mappings(train_examples)
    label2id = m["label2id"]
    id2label = m["id2label"]
    unknown = sorted({ex.label for ex in dev_examples if ex.label not in label2id})
    if unknown:
        raise ValueError(f"Dev has labels missing from train: {unknown}")
    return {
        "train_dataset": SVDADataset(train_examples, label2id),
        "dev_dataset": SVDADataset(dev_examples, label2id),
        "label2id": label2id,
        "id2label": id2label,
    }


class SVDACollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        texts = [item["text"] for item in batch]
        labels = [item["labels"] for item in batch]
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        enc["labels"] = enc["input_ids"].new_tensor(labels)
        enc["texts"] = texts
        return enc


def compute_class_balanced_weights(dataset: SVDADataset, power: float = 0.5) -> List[float]:
    """Per-example weight = 1 / class_count^power.

    power=0.5 (default) gives roughly the gold "balanced" sampling commonly used in
    long-tail recognition (see "Class-Balanced Loss Based on Effective Number of
    Samples", CVPR 2019). power=1.0 fully equalizes per-class expected uses;
    power=0 keeps the natural distribution.
    """
    counts = Counter([ex.label for ex in dataset.examples])
    if power <= 0:
        return [1.0 for _ in dataset.examples]
    return [1.0 / float(counts[ex.label]) ** float(power) for ex in dataset.examples]


def build_train_dataloader(
    dataset: SVDADataset,
    tokenizer,
    batch_size: int,
    max_length: int,
    num_workers: int,
    sampling: str = "weighted",
    class_balance_power: float = 0.5,
    epochs_target_size: int = -1,
    generator: torch.Generator = None,
) -> DataLoader:
    if sampling == "weighted":
        weights = compute_class_balanced_weights(dataset, power=class_balance_power)
        num_samples = int(epochs_target_size) if epochs_target_size > 0 else len(dataset)
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=num_samples, replacement=True, generator=generator
        )
        return DataLoader(
            dataset, batch_size=int(batch_size), sampler=sampler,
            num_workers=int(num_workers),
            collate_fn=SVDACollator(tokenizer=tokenizer, max_length=max_length),
        )
    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=True,
        num_workers=int(num_workers),
        collate_fn=SVDACollator(tokenizer=tokenizer, max_length=max_length),
    )


def build_eval_dataloader(
    dataset: SVDADataset, tokenizer, batch_size: int, max_length: int, num_workers: int
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False,
        num_workers=int(num_workers),
        collate_fn=SVDACollator(tokenizer=tokenizer, max_length=max_length),
    )
