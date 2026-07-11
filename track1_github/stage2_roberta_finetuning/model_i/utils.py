"""Run-path and IO helpers for ExpI."""
from __future__ import annotations

import csv
import json
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


EXPI_SUMMARY_COLUMNS = [
    "run_dir",
    "seed",
    "candidate_mode",
    "lambda_hybrid",
    "lambda_global",
    "lambda_rdrop",
    "fgm_epsilon",
    "ema_decay",
    "use_question",
    "best_epoch",
    "best_dev_accuracy",
    "best_dev_macro_precision",
    "best_dev_macro_recall",
    "best_dev_macro_f1",
    "ema_dev_accuracy",
    "ema_dev_macro_precision",
    "ema_dev_macro_recall",
    "ema_dev_macro_f1",
    "warmup_checkpoint",
    "best_checkpoint",
]


def load_yaml(path: str) -> Dict[str, object]:
    import yaml
    with Path(path).open("r", encoding="utf-8") as infile:
        return yaml.safe_load(infile)


def set_seed(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str):
    import torch
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def ensure_local_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Local pretrained model directory missing: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {path}")
    return path


def create_expI_run_paths(output_root: str, seed: int, tag: str) -> Dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_dir = Path(output_root) / tag
    svda_run_dir = tag_dir / f"{timestamp}_seed{seed}"
    paths = {
        "output_root": Path(output_root),
        "tag_dir": tag_dir,
        "svda_run_dir": svda_run_dir,
        "svda_finetune": svda_run_dir,
        "svda_checkpoints": svda_run_dir / "checkpoints",
        "logs": svda_run_dir / "logs",
        "metrics": svda_run_dir / "metrics",
        "plots": svda_run_dir / "plots",
        "artifacts": svda_run_dir / "artifacts",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def append_jsonl(path: Path, record: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(data: Dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=2)


def save_confusion_matrix_csv(matrix: List[List[int]], labels: List[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["label"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + row)


def save_confusion_matrix_png(matrix: List[List[int]], labels: List[str], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="ExpI Dev Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = np.max(matrix) / 2.0 if matrix else 0.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            axis.text(
                j, i, format(matrix[i][j], "d"),
                ha="center", va="center",
                color="white" if matrix[i][j] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def copy_config_file(config_path: str, destination: Path) -> None:
    shutil.copy2(config_path, destination)


def save_label_mappings(label2id: Dict[str, int], id2label: Dict[int, str], artifacts_dir: Path) -> None:
    save_json(label2id, artifacts_dir / "label2id.json")
    save_json({str(idx): label for idx, label in id2label.items()}, artifacts_dir / "id2label.json")


def _iter_valid_run_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and re.match(r"^\d{8}_\d{6}_seed\d+$", path.name)]
    )


def refresh_expI_summary_csv(output_root: str) -> Path:
    root = Path(output_root)
    rows: List[Dict[str, object]] = []
    if root.exists():
        for tag_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for run_dir in _iter_valid_run_dirs(tag_dir):
                metrics_file = run_dir / "metrics" / "metrics.json"
                if not metrics_file.exists():
                    continue
                with metrics_file.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                row = {key: payload.get(key, "") for key in EXPI_SUMMARY_COLUMNS}
                rows.append(row)
    summary_path = root / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=EXPI_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return summary_path
