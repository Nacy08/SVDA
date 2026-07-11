import json
import random
import re
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
    def __init__(self, examples: Sequence[SVDAExample], label2id: Dict[str, int]):
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


@dataclass
class FULCRAExample:
    text: str
    values: List[float]


class FULCRADataset(Dataset):
    def __init__(self, examples: Sequence[FULCRAExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        example = self.examples[index]
        return {
            "text": example.text,
            "labels": example.values,
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
        examples.append(SVDAExample(text=text, label=label))
    return examples


def build_label_mappings(examples: Sequence[SVDAExample]) -> Dict[str, Dict[object, object]]:
    labels = sorted({example.label for example in examples})
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return {"label2id": label2id, "id2label": id2label}


def parse_value_type_item(item: object, record_index: int) -> tuple[str, int]:
    text = str(item).strip()
    match = re.match(r"^\s*(.+?)\s*:\s*(.*?)\s*$", text)
    if not match:
        raise ValueError(f"Invalid value_types item in record {record_index}: {text}")
    raw_sign = match.group(2).strip()
    if raw_sign in {"+1", "1"}:
        sign = 1
    elif raw_sign == "-1":
        sign = -1
    else:
        sign = 0
    return match.group(1).strip(), sign


def build_value_type_mappings(records: Sequence[Dict[str, object]], value_types_field: str) -> Dict[str, Dict[object, object]]:
    value_types = set()
    for index, record in enumerate(records):
        if value_types_field not in record:
            raise KeyError(f"Missing value types field '{value_types_field}' in record {index}")
        raw_items = record[value_types_field]
        if not isinstance(raw_items, list):
            raise TypeError(f"Expected list field '{value_types_field}' in record {index}")
        for item in raw_items:
            value_type, _ = parse_value_type_item(item, index)
            value_types.add(value_type)
    ordered = sorted(value_types)
    value_type2id = {value_type: idx for idx, value_type in enumerate(ordered)}
    id2value_type = {idx: value_type for value_type, idx in value_type2id.items()}
    return {"value_type2id": value_type2id, "id2value_type": id2value_type}


def extract_fulcra_examples(
    records: Sequence[Dict[str, object]],
    text_field: str,
    value_types_field: str,
    value_type2id: Dict[str, int],
) -> List[FULCRAExample]:
    examples: List[FULCRAExample] = []
    for index, record in enumerate(records):
        if text_field not in record:
            raise KeyError(f"Missing text field '{text_field}' in record {index}")
        if value_types_field not in record:
            raise KeyError(f"Missing value types field '{value_types_field}' in record {index}")

        text = str(record[text_field]).strip()
        if not text:
            raise ValueError(f"Empty text in record {index}")
        raw_items = record[value_types_field]
        if not isinstance(raw_items, list):
            raise TypeError(f"Expected list field '{value_types_field}' in record {index}")

        sparse_values: Dict[str, int] = {}
        for item in raw_items:
            value_type, sign = parse_value_type_item(item, index)
            if value_type in sparse_values and sparse_values[value_type] != sign:
                sparse_values[value_type] = 0
            elif value_type not in sparse_values:
                sparse_values[value_type] = sign

        vector = [0.0] * len(value_type2id)
        for value_type, sign in sparse_values.items():
            vector[value_type2id[value_type]] = float(sign)
        examples.append(FULCRAExample(text=text, values=vector))
    return examples


def validate_dev_labels(dev_examples: Sequence[SVDAExample], label2id: Dict[str, int]) -> None:
    unknown_labels = sorted({example.label for example in dev_examples if example.label not in label2id})
    if unknown_labels:
        joined = ", ".join(unknown_labels)
        raise ValueError(f"Found labels in dev split that are absent from train split: {joined}")


def build_svda_datasets(config: Dict[str, object]) -> Dict[str, object]:
    data_config = config["data"]
    train_examples = extract_examples(
        load_jsonl_records(data_config["train_path"]),
        text_field=data_config["text_field"],
        label_field=data_config["label_field"],
    )
    dev_examples = extract_examples(
        load_jsonl_records(data_config["dev_path"]),
        text_field=data_config["text_field"],
        label_field=data_config["label_field"],
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


def build_fulcra_datasets(config: Dict[str, object], seed: int) -> Dict[str, object]:
    data_config = config["fulcra_data"]
    records = load_jsonl_records(data_config["path"])
    mappings = build_value_type_mappings(records, data_config["value_types_field"])
    value_type2id = mappings["value_type2id"]
    id2value_type = mappings["id2value_type"]
    examples = extract_fulcra_examples(
        records,
        text_field=data_config["text_field"],
        value_types_field=data_config["value_types_field"],
        value_type2id=value_type2id,
    )

    validation_split = float(data_config.get("validation_split", 0.1))
    if not 0.0 < validation_split < 1.0:
        raise ValueError("fulcra_data.validation_split must be between 0 and 1.")
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    num_val = max(1, round(len(indices) * validation_split))
    num_val = min(num_val, len(indices) - 1)
    val_indices = set(indices[:num_val])
    train_examples = [example for idx, example in enumerate(examples) if idx not in val_indices]
    val_examples = [example for idx, example in enumerate(examples) if idx in val_indices]

    return {
        "train_dataset": FULCRADataset(train_examples),
        "val_dataset": FULCRADataset(val_examples),
        "value_type2id": value_type2id,
        "id2value_type": id2value_type,
        "num_value_dims": len(value_type2id),
    }


class SVDACollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

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


class FULCRACollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        encoded["labels"] = torch.tensor(labels, dtype=torch.float)
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
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=SVDACollator(tokenizer=tokenizer, max_length=max_length),
    )


def build_fulcra_dataloader(
    dataset: Dataset,
    tokenizer,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=FULCRACollator(tokenizer=tokenizer, max_length=max_length),
    )
