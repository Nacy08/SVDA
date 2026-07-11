import csv
import json
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

SUMMARY_COLUMNS = [
    "run_dir",
    "seed",
    "best_epoch",
    "best_dev_accuracy",
    "best_dev_macro_precision",
    "best_dev_macro_recall",
    "best_dev_macro_f1",
]

EXPB_SUMMARY_COLUMNS = [
    "run_dir",
    "parent_run_dir",
    "seed",
    "best_epoch",
    "best_dev_accuracy",
    "best_dev_macro_precision",
    "best_dev_macro_recall",
    "best_dev_macro_f1",
    "num_value_dims",
    "warmup_checkpoint",
]

EXPF_SUMMARY_COLUMNS = [
    "run_dir",
    "seed",
    "lambda_conf",
    "confusable_margin",
    "confusable_pairs_count",
    "best_epoch",
    "best_dev_accuracy",
    "best_dev_macro_precision",
    "best_dev_macro_recall",
    "best_dev_macro_f1",
    "num_value_dims",
    "warmup_checkpoint",
    "best_checkpoint",
]

EXPF_NEW_SUMMARY_COLUMNS = [
    "run_dir",
    "seed",
    "candidate_mode",
    "lambda_hybrid",
    "lambda_global",
    "ranking_margin",
    "top_k_hybrid",
    "top_k_global",
    "start_epoch",
    "candidate_labels_total",
    "best_epoch",
    "best_dev_accuracy",
    "best_dev_macro_precision",
    "best_dev_macro_recall",
    "best_dev_macro_f1",
    "num_value_dims",
    "warmup_checkpoint",
    "best_checkpoint",
]


def load_yaml(path: str) -> Dict[str, object]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as infile:
        return yaml.safe_load(infile)


def save_yaml(data: Dict[str, object], path: Path) -> None:
    import yaml

    with path.open("w", encoding="utf-8") as outfile:
        yaml.safe_dump(data, outfile, sort_keys=False, allow_unicode=True)


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
        raise FileNotFoundError(
            f"Local pretrained model directory does not exist: {path}. "
            "Please place roberta-large files there or update model.pretrained_model_name_or_path."
        )
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory for the local pretrained model, got: {path}")
    return path


def create_run_directories(output_root: str, seed: int) -> Dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{timestamp}_seed{seed}"
    subdirs = {
        "run_dir": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "logs": run_dir / "logs",
        "metrics": run_dir / "metrics",
        "plots": run_dir / "plots",
        "artifacts": run_dir / "artifacts",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=False)
    return subdirs


def create_expB_run_directories(output_root: str, seed: int) -> Dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{timestamp}_seed{seed}"
    subdirs = {
        "run_dir": run_dir,
        "fulcra_warmup": run_dir / "fulcra_warmup",
        "fulcra_checkpoints": run_dir / "fulcra_warmup" / "checkpoints",
        "svda_finetune": run_dir / "svda_finetune",
        "svda_checkpoints": run_dir / "svda_finetune" / "checkpoints",
        "logs": run_dir / "logs",
        "metrics": run_dir / "metrics",
        "plots": run_dir / "plots",
        "artifacts": run_dir / "artifacts",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=False)
    return subdirs


def get_expB_run_paths_from_warmup_checkpoint(warmup_ckpt: str) -> Dict[str, Path]:
    checkpoint_path = Path(warmup_ckpt)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Warm-up checkpoint not found: {checkpoint_path}")
    run_dir = checkpoint_path.parents[2]
    if run_dir.name == "fulcra_warmup":
        run_dir = checkpoint_path.parents[1]
    paths = {
        "run_dir": run_dir,
        "fulcra_warmup": run_dir / "fulcra_warmup",
        "fulcra_checkpoints": run_dir / "fulcra_warmup" / "checkpoints",
        "svda_finetune": run_dir / "svda_finetune",
        "svda_checkpoints": run_dir / "svda_finetune" / "checkpoints",
        "logs": run_dir / "logs",
        "metrics": run_dir / "metrics",
        "plots": run_dir / "plots",
        "artifacts": run_dir / "artifacts",
    }
    for key, path in paths.items():
        if key == "run_dir":
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    return paths


def create_expB_svda_finetune_run_paths(warmup_ckpt: str, seed: int) -> Dict[str, Path]:
    checkpoint_path = Path(warmup_ckpt)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Warm-up checkpoint not found: {checkpoint_path}")
    parent_run_dir = checkpoint_path.parents[2]
    if parent_run_dir.name == "fulcra_warmup":
        parent_run_dir = checkpoint_path.parents[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    svda_run_dir = parent_run_dir / "svda_finetune_runs" / f"{timestamp}_seed{seed}"
    paths = {
        "run_dir": parent_run_dir,
        "fulcra_warmup": parent_run_dir / "fulcra_warmup",
        "fulcra_checkpoints": parent_run_dir / "fulcra_warmup" / "checkpoints",
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


def create_expF_svda_finetune_run_paths(output_root: str, seed: int, lambda_conf: float) -> Dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lambda_dir = Path(output_root) / f"lambda_{float(lambda_conf):.3f}"
    svda_run_dir = lambda_dir / f"{timestamp}_seed{seed}"
    paths = {
        "output_root": Path(output_root),
        "lambda_dir": lambda_dir,
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


def create_expF_new_svda_finetune_run_paths(
    output_root: str,
    seed: int,
    candidate_mode: str,
    lambda_hybrid: float,
    lambda_global: float,
) -> Dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_dir = Path(output_root) / str(candidate_mode)
    lambda_dir = mode_dir / f"lambda_hybrid_{float(lambda_hybrid):.3f}_lambda_global_{float(lambda_global):.3f}"
    svda_run_dir = lambda_dir / f"{timestamp}_seed{seed}"
    paths = {
        "output_root": Path(output_root),
        "candidate_mode_dir": mode_dir,
        "lambda_dir": lambda_dir,
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
        title="Dev Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = np.max(matrix) / 2.0 if matrix else 0.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            axis.text(
                j,
                i,
                format(matrix[i][j], "d"),
                ha="center",
                va="center",
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


def save_value_type_mappings(
    value_type2id: Dict[str, int],
    id2value_type: Dict[int, str],
    artifacts_dir: Path,
) -> None:
    save_json(value_type2id, artifacts_dir / "value_type2id.json")
    save_json({str(idx): value_type for idx, value_type in id2value_type.items()}, artifacts_dir / "id2value_type.json")


def build_summary_row(metrics_payload: Dict[str, object]) -> Dict[str, object]:
    return {
        "run_dir": metrics_payload["run_dir"],
        "seed": metrics_payload["seed"],
        "best_epoch": metrics_payload["best_epoch"],
        "best_dev_accuracy": metrics_payload["best_dev_accuracy"],
        "best_dev_macro_precision": metrics_payload["best_dev_macro_precision"],
        "best_dev_macro_recall": metrics_payload["best_dev_macro_recall"],
        "best_dev_macro_f1": metrics_payload["best_dev_macro_f1"],
    }


def build_expB_summary_row(metrics_payload: Dict[str, object]) -> Dict[str, object]:
    return {
        "run_dir": metrics_payload["run_dir"],
        "parent_run_dir": metrics_payload.get("parent_run_dir", ""),
        "seed": metrics_payload["seed"],
        "best_epoch": metrics_payload.get("best_epoch", ""),
        "best_dev_accuracy": metrics_payload["best_dev_accuracy"],
        "best_dev_macro_precision": metrics_payload["best_dev_macro_precision"],
        "best_dev_macro_recall": metrics_payload["best_dev_macro_recall"],
        "best_dev_macro_f1": metrics_payload["best_dev_macro_f1"],
        "num_value_dims": metrics_payload["num_value_dims"],
        "warmup_checkpoint": metrics_payload.get("warmup_checkpoint", ""),
    }


def build_expF_summary_row(metrics_payload: Dict[str, object]) -> Dict[str, object]:
    return {
        "run_dir": metrics_payload["run_dir"],
        "seed": metrics_payload["seed"],
        "lambda_conf": metrics_payload["lambda_conf"],
        "confusable_margin": metrics_payload["confusable_margin"],
        "confusable_pairs_count": metrics_payload["confusable_pairs_count"],
        "best_epoch": metrics_payload.get("best_epoch", ""),
        "best_dev_accuracy": metrics_payload["best_dev_accuracy"],
        "best_dev_macro_precision": metrics_payload["best_dev_macro_precision"],
        "best_dev_macro_recall": metrics_payload["best_dev_macro_recall"],
        "best_dev_macro_f1": metrics_payload["best_dev_macro_f1"],
        "num_value_dims": metrics_payload["num_value_dims"],
        "warmup_checkpoint": metrics_payload.get("warmup_checkpoint", ""),
        "best_checkpoint": metrics_payload.get("best_checkpoint", ""),
    }


def build_expF_new_summary_row(metrics_payload: Dict[str, object]) -> Dict[str, object]:
    return {
        "run_dir": metrics_payload["run_dir"],
        "seed": metrics_payload["seed"],
        "candidate_mode": metrics_payload["candidate_mode"],
        "lambda_hybrid": metrics_payload["lambda_hybrid"],
        "lambda_global": metrics_payload["lambda_global"],
        "ranking_margin": metrics_payload["ranking_margin"],
        "top_k_hybrid": metrics_payload["top_k_hybrid"],
        "top_k_global": metrics_payload["top_k_global"],
        "start_epoch": metrics_payload["start_epoch"],
        "candidate_labels_total": metrics_payload["candidate_labels_total"],
        "best_epoch": metrics_payload.get("best_epoch", ""),
        "best_dev_accuracy": metrics_payload["best_dev_accuracy"],
        "best_dev_macro_precision": metrics_payload["best_dev_macro_precision"],
        "best_dev_macro_recall": metrics_payload["best_dev_macro_recall"],
        "best_dev_macro_f1": metrics_payload["best_dev_macro_f1"],
        "num_value_dims": metrics_payload["num_value_dims"],
        "warmup_checkpoint": metrics_payload.get("warmup_checkpoint", ""),
        "best_checkpoint": metrics_payload.get("best_checkpoint", ""),
    }


def _iter_valid_run_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and re.match(r"^\d{8}_\d{6}_seed\d+$", path.name)]
    )


def _iter_expB_svda_metrics(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    metrics_paths: List[Path] = []
    for run_dir in _iter_valid_run_dirs(root):
        legacy_metrics = run_dir / "svda_finetune" / "metrics.json"
        if legacy_metrics.exists():
            metrics_paths.append(legacy_metrics)
        nested_root = run_dir / "svda_finetune_runs"
        if nested_root.exists():
            metrics_paths.extend(sorted(nested_root.glob("*/metrics.json")))
            metrics_paths.extend(sorted(nested_root.glob("*/metrics/metrics.json")))
    return metrics_paths


def _iter_expF_svda_metrics(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    metrics_paths: List[Path] = []
    for lambda_dir in sorted(root.glob("lambda_*")):
        if not lambda_dir.is_dir():
            continue
        for run_dir in _iter_valid_run_dirs(lambda_dir):
            direct_metrics = run_dir / "metrics.json"
            nested_metrics = run_dir / "metrics" / "metrics.json"
            if nested_metrics.exists():
                metrics_paths.append(nested_metrics)
            elif direct_metrics.exists():
                metrics_paths.append(direct_metrics)
    return metrics_paths


def _iter_expF_new_svda_metrics(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    metrics_paths: List[Path] = []
    for mode_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for lambda_dir in sorted(mode_dir.glob("lambda_hybrid_*_lambda_global_*")):
            if not lambda_dir.is_dir():
                continue
            for run_dir in _iter_valid_run_dirs(lambda_dir):
                nested_metrics = run_dir / "metrics" / "metrics.json"
                direct_metrics = run_dir / "metrics.json"
                if nested_metrics.exists():
                    metrics_paths.append(nested_metrics)
                elif direct_metrics.exists():
                    metrics_paths.append(direct_metrics)
    return metrics_paths


def _sort_summary_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    def sort_key(row: Dict[str, object]):
        run_name = Path(str(row["run_dir"])).name
        match = re.match(r"^(?P<timestamp>\d{8}_\d{6})_seed(?P<seed>\d+)$", run_name)
        if match:
            return match.group("timestamp"), int(match.group("seed"))
        return run_name, int(row["seed"])

    return sorted(rows, key=sort_key)


def _sort_expF_summary_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    def sort_key(row: Dict[str, object]):
        run_name = Path(str(row["run_dir"])).name
        match = re.match(r"^(?P<timestamp>\d{8}_\d{6})_seed(?P<seed>\d+)$", run_name)
        timestamp = match.group("timestamp") if match else run_name
        return int(row["seed"]), float(row["lambda_conf"]), timestamp

    return sorted(rows, key=sort_key)


def _sort_expF_new_summary_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    def sort_key(row: Dict[str, object]):
        run_name = Path(str(row["run_dir"])).name
        match = re.match(r"^(?P<timestamp>\d{8}_\d{6})_seed(?P<seed>\d+)$", run_name)
        timestamp = match.group("timestamp") if match else run_name
        return (
            int(row["seed"]),
            str(row["candidate_mode"]),
            float(row["lambda_hybrid"]),
            float(row["lambda_global"]),
            timestamp,
        )

    return sorted(rows, key=sort_key)


def refresh_summary_csv(output_root: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.csv"

    rows: List[Dict[str, object]] = []
    for run_dir in _iter_valid_run_dirs(root):
        metrics_path = run_dir / "metrics" / "metrics.json"
        run_info_path = run_dir / "artifacts" / "run_info.json"
        if not metrics_path.exists():
            continue
        if not run_info_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as infile:
            payload = json.load(infile)
        with run_info_path.open("r", encoding="utf-8") as infile:
            run_info = json.load(infile)
        required_fields = {
            "run_dir",
            "seed",
            "best_epoch",
            "best_dev_accuracy",
            "best_dev_macro_precision",
            "best_dev_macro_recall",
            "best_dev_macro_f1",
        }
        if not required_fields.issubset(payload):
            continue
        if str(run_info.get("run_dir", "")).replace("\\", "/") != str(payload["run_dir"]).replace("\\", "/"):
            continue
        rows.append(build_summary_row(payload))

    rows = _sort_summary_rows(rows)
    with summary_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return summary_path


def refresh_expB_summary_csv(output_root: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.csv"

    rows: List[Dict[str, object]] = []
    for metrics_path in _iter_expB_svda_metrics(root):
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as infile:
            payload = json.load(infile)
        required_fields = {
            "run_dir",
            "seed",
            "best_epoch",
            "best_dev_accuracy",
            "best_dev_macro_precision",
            "best_dev_macro_recall",
            "best_dev_macro_f1",
            "num_value_dims",
        }
        if not required_fields.issubset(payload):
            continue
        rows.append(build_expB_summary_row(payload))

    rows = _sort_summary_rows(rows)
    with summary_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=EXPB_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return summary_path


def refresh_expF_summary_csv(output_root: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.csv"

    rows: List[Dict[str, object]] = []
    for metrics_path in _iter_expF_svda_metrics(root):
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as infile:
            payload = json.load(infile)
        required_fields = {
            "run_dir",
            "seed",
            "lambda_conf",
            "confusable_margin",
            "confusable_pairs_count",
            "best_epoch",
            "best_dev_accuracy",
            "best_dev_macro_precision",
            "best_dev_macro_recall",
            "best_dev_macro_f1",
            "num_value_dims",
        }
        if not required_fields.issubset(payload):
            continue
        rows.append(build_expF_summary_row(payload))

    rows = _sort_expF_summary_rows(rows)
    with summary_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=EXPF_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return summary_path


def refresh_expF_new_summary_csv(output_root: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.csv"

    rows: List[Dict[str, object]] = []
    for metrics_path in _iter_expF_new_svda_metrics(root):
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as infile:
            payload = json.load(infile)
        required_fields = {
            "run_dir",
            "seed",
            "candidate_mode",
            "lambda_hybrid",
            "lambda_global",
            "ranking_margin",
            "top_k_hybrid",
            "top_k_global",
            "start_epoch",
            "candidate_labels_total",
            "best_epoch",
            "best_dev_accuracy",
            "best_dev_macro_precision",
            "best_dev_macro_recall",
            "best_dev_macro_f1",
            "num_value_dims",
        }
        if not required_fields.issubset(payload):
            continue
        rows.append(build_expF_new_summary_row(payload))

    rows = _sort_expF_new_summary_rows(rows)
    with summary_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=EXPF_NEW_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return summary_path
