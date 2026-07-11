"""Data loading for ExpI.

Compared with ExpF_new, the SVDA input is enriched with the `Question` field
(concatenated to the consistent response with a single space). The hidden
prior is that the model needs the question to anchor what "consistent"
means in context.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


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

    def __getitem__(self, index: int) -> Dict[str, object]:
        example = self.examples[index]
        return {
            "text": example.text,
            "label": example.label,
            "labels": self.label2id[example.label],
        }


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
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {file_path}: {exc}"
                ) from exc
    if not records:
        raise ValueError(f"No records found in {file_path}")
    return records


def extract_examples(
    records: Sequence[Dict[str, object]],
    text_field: str,
    label_field: str,
    question_field: str = "",
    use_question: bool = False,
) -> List[SVDAExample]:
    examples: List[SVDAExample] = []
    for index, record in enumerate(records):
        if text_field not in record:
            raise KeyError(f"Missing text field '{text_field}' in record {index}")
        if label_field not in record:
            raise KeyError(f"Missing label field '{label_field}' in record {index}")
        text = str(record[text_field]).strip()
        label = str(record[label_field]).strip()
        if not text:
            raise ValueError(f"Empty text in record {index}")
        if not label:
            raise ValueError(f"Empty label in record {index}")
        if use_question:
            if question_field not in record:
                raise KeyError(
                    f"Missing question field '{question_field}' in record {index}"
                )
            question = str(record[question_field]).strip()
            text = f"{question} {text}" if question else text
        examples.append(SVDAExample(text=text, label=label))
    return examples


def build_label_mappings(examples: Sequence[SVDAExample]) -> Dict[str, Dict[object, object]]:
    labels = sorted({example.label for example in examples})
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return {"label2id": label2id, "id2label": id2label}


def validate_dev_labels(dev_examples: Sequence[SVDAExample], label2id: Dict[str, int]) -> None:
    unknown_labels = sorted({example.label for example in dev_examples if example.label not in label2id})
    if unknown_labels:
        raise ValueError(
            f"Found labels in dev split that are absent from train split: {', '.join(unknown_labels)}"
        )


def build_svda_datasets(config: Dict[str, object]) -> Dict[str, object]:
    data_config = config["data"]
    use_question = bool(data_config.get("use_question", False))
    question_field = str(data_config.get("question_field", "Question"))
    train_examples = extract_examples(
        load_jsonl_records(str(data_config["train_path"])),
        text_field=str(data_config["text_field"]),
        label_field=str(data_config["label_field"]),
        question_field=question_field,
        use_question=use_question,
    )
    dev_examples = extract_examples(
        load_jsonl_records(str(data_config["dev_path"])),
        text_field=str(data_config["text_field"]),
        label_field=str(data_config["label_field"]),
        question_field=question_field,
        use_question=use_question,
    )
    mappings = build_label_mappings(train_examples)
    label2id = mappings["label2id"]
    id2label = mappings["id2label"]
    validate_dev_labels(dev_examples, label2id)
    train_dataset = SVDADataset(train_examples, label2id)
    dev_dataset = SVDADataset(dev_examples, label2id)
    return {
        "train_dataset": train_dataset,
        "dev_dataset": dev_dataset,
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
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = encoded["input_ids"].new_tensor(labels)
        encoded["texts"] = texts
        return encoded


def build_dataloader(
    dataset: Dataset,
    tokenizer,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=SVDACollator(tokenizer=tokenizer, max_length=max_length),
    )
