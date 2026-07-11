import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import build_dataloader, build_svda_datasets
from metrics import compute_classification_metrics
from models import RobertaForSVDAClassification
from utils import (
    append_jsonl,
    copy_config_file,
    create_expF_new_svda_finetune_run_paths,
    ensure_local_model_path,
    load_yaml,
    refresh_expF_new_summary_csv,
    resolve_device,
    save_confusion_matrix_csv,
    save_confusion_matrix_png,
    save_json,
    save_label_mappings,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ExpF_new SVDA classifier from a FULCRA warm-up checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the Experiment F_new YAML config file.")
    parser.add_argument("--seed", required=True, type=int, help="Random seed for a single run.")
    parser.add_argument("--warmup_ckpt", required=True, help="Path to fulcra_warmup/checkpoints/best.pt.")
    parser.add_argument("--lambda_hybrid", required=True, type=float, help="Weight for HybridRankingLoss.")
    parser.add_argument("--lambda_global", required=True, type=float, help="Weight for GlobalFallbackRankingLoss.")
    parser.add_argument("--candidate_mode", required=True, help="Candidate mode for hybrid hard negatives.")
    return parser.parse_args()


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
    }


def evaluate(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    id2label: Dict[int, str],
) -> Tuple[Dict[str, object], List[int], List[int]]:
    model.eval()
    predictions: List[int] = []
    labels: List[int] = []
    with torch.no_grad():
        for batch in dataloader:
            tensors = move_batch_to_device(batch, device)
            outputs = model(**tensors)
            preds = torch.argmax(outputs["logits"], dim=-1)
            predictions.extend(preds.cpu().tolist())
            labels.extend(tensors["labels"].cpu().tolist())
    metrics = compute_classification_metrics(labels=labels, predictions=predictions, id2label=id2label)
    return metrics, predictions, labels


def validate_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"--{name} must be non-negative.")


def validate_warmup_checkpoint_path(warmup_ckpt: str) -> Path:
    checkpoint_path = Path(warmup_ckpt)
    if str(checkpoint_path).startswith("/path/to/"):
        raise FileNotFoundError(
            "The --warmup_ckpt value is still the README placeholder path. "
            "Use the real ExpB FULCRA warm-up checkpoint, for example: "
            "/root/Task1_baseline1/expB/outputs/20260428_215642_seed42/fulcra_warmup/checkpoints/best.pt"
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Warm-up checkpoint not found: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Warm-up checkpoint is not a file: {checkpoint_path}")
    return checkpoint_path


def _dedupe_without_gold(candidate_ids: Sequence[int], gold_id: int) -> Tuple[int, ...]:
    deduped: List[int] = []
    for candidate_id in candidate_ids:
        candidate_id = int(candidate_id)
        if candidate_id == gold_id or candidate_id in deduped:
            continue
        deduped.append(candidate_id)
    return tuple(deduped)


def _validate_label_list(labels: Sequence[str], label2id: Dict[str, int], field_name: str) -> None:
    missing = sorted({str(label) for label in labels if str(label) not in label2id})
    if missing:
        raise ValueError(f"{field_name} contains labels absent from label2id: {', '.join(missing)}")


def build_candidate_map(
    label2id: Dict[str, int],
    circular_order: Sequence[str],
    candidate_mode: str,
    core_groups: Sequence[Sequence[str]],
) -> Dict[int, Tuple[int, ...]]:
    if candidate_mode not in {"adjacent_1", "adjacent_2", "core_plus_adjacent_1"}:
        raise ValueError(
            "candidate_mode must be one of adjacent_1, adjacent_2, core_plus_adjacent_1; "
            f"got {candidate_mode}"
        )
    _validate_label_list(circular_order, label2id, "circular_order")
    if len(set(circular_order)) != len(circular_order):
        raise ValueError("circular_order must not contain duplicate labels.")
    if set(circular_order) != set(label2id):
        missing_from_order = sorted(set(label2id) - set(circular_order))
        extra_in_order = sorted(set(circular_order) - set(label2id))
        raise ValueError(
            "circular_order must contain exactly the training labels. "
            f"missing={missing_from_order}, extra={extra_in_order}"
        )
    for group_index, group in enumerate(core_groups):
        _validate_label_list(group, label2id, f"core_groups[{group_index}]")

    order_ids = [label2id[str(label)] for label in circular_order]
    radius = 2 if candidate_mode == "adjacent_2" else 1
    candidate_map: Dict[int, Tuple[int, ...]] = {}
    total_labels = len(order_ids)
    for index, gold_id in enumerate(order_ids):
        candidate_ids: List[int] = []
        for offset in range(radius, 0, -1):
            candidate_ids.append(order_ids[(index - offset) % total_labels])
        for offset in range(1, radius + 1):
            candidate_ids.append(order_ids[(index + offset) % total_labels])
        candidate_map[gold_id] = _dedupe_without_gold(candidate_ids, gold_id)

    if candidate_mode == "core_plus_adjacent_1":
        for group in core_groups:
            group_ids = [label2id[str(label)] for label in group]
            for gold_id in group_ids:
                candidate_map[gold_id] = _dedupe_without_gold(
                    list(candidate_map.get(gold_id, ())) + group_ids,
                    gold_id,
                )
    return candidate_map


def build_confusable_pairs(
    config_pairs: Sequence[Sequence[str]],
    label2id: Dict[str, int],
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    missing_labels = set()
    for pair in config_pairs:
        if len(pair) != 2:
            raise ValueError(f"Each confusable hard pair must contain exactly two labels, got: {pair}")
        left_label, right_label = str(pair[0]), str(pair[1])
        if left_label not in label2id:
            missing_labels.add(left_label)
        if right_label not in label2id:
            missing_labels.add(right_label)
        if left_label in label2id and right_label in label2id:
            pairs.append((label2id[left_label], label2id[right_label]))
    if missing_labels:
        joined = ", ".join(sorted(missing_labels))
        raise ValueError(f"Confusable hard pair labels are absent from label2id: {joined}")
    return pairs


def train_one_epoch(model, dataloader, optimizer, scheduler, device: torch.device, epoch: int) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_hybrid_ranking_loss = 0.0
    total_global_fallback_ranking_loss = 0.0
    for batch in dataloader:
        tensors = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        outputs = model(**tensors, current_epoch=epoch)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        total_ce_loss += outputs["ce_loss"].item()
        total_hybrid_ranking_loss += outputs["hybrid_ranking_loss"].item()
        total_global_fallback_ranking_loss += outputs["global_fallback_ranking_loss"].item()
    denominator = max(len(dataloader), 1)
    return {
        "train_loss": total_loss / denominator,
        "train_ce_loss": total_ce_loss / denominator,
        "train_hybrid_ranking_loss": total_hybrid_ranking_loss / denominator,
        "train_global_fallback_ranking_loss": total_global_fallback_ranking_loss / denominator,
    }


def save_best_checkpoint(checkpoint_path: Path, model, epoch: int, best_metrics: Dict[str, object]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_metrics": best_metrics,
        },
        checkpoint_path,
    )


def main() -> None:
    args = parse_args()
    validate_non_negative("lambda_hybrid", args.lambda_hybrid)
    validate_non_negative("lambda_global", args.lambda_global)
    warmup_checkpoint_path = validate_warmup_checkpoint_path(args.warmup_ckpt)
    config = load_yaml(args.config)
    config["train"]["seed"] = args.seed

    set_seed(args.seed)
    ensure_local_model_path(config["model"]["pretrained_model_name_or_path"])
    run_paths = create_expF_new_svda_finetune_run_paths(
        config["experiment"]["output_root"],
        args.seed,
        args.candidate_mode,
        args.lambda_hybrid,
        args.lambda_global,
    )
    device = resolve_device(config["runtime"]["device"])

    copy_config_file(args.config, run_paths["artifacts"] / "config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        local_files_only=True,
    )
    dataset_bundle = build_svda_datasets(config)
    label2id = dataset_bundle["label2id"]
    id2label = dataset_bundle["id2label"]
    ranking_config = config.get("hybrid_ranking", {})
    ranking_enabled = bool(ranking_config.get("enabled", True))
    effective_lambda_hybrid = args.lambda_hybrid if ranking_enabled else 0.0
    effective_lambda_global = args.lambda_global if ranking_enabled else 0.0
    ranking_margin = float(ranking_config.get("margin", 1.0))
    top_k_hybrid = int(ranking_config.get("top_k_hybrid", 1))
    top_k_global = int(ranking_config.get("top_k_global", 1))
    start_epoch = int(ranking_config.get("start_epoch", 2))
    candidate_map = build_candidate_map(
        label2id=label2id,
        circular_order=ranking_config.get("circular_order", []),
        candidate_mode=args.candidate_mode,
        core_groups=ranking_config.get("core_groups", []),
    )

    num_labels = len(label2id)
    expected_labels = int(config["model"]["num_labels"])
    if num_labels != expected_labels:
        raise ValueError(f"Config expects {expected_labels} labels, but training data produced {num_labels} labels.")
    save_label_mappings(label2id, id2label, run_paths["artifacts"])

    train_loader = build_dataloader(
        dataset_bundle["train_dataset"],
        tokenizer=tokenizer,
        batch_size=int(config["train"]["batch_size"]),
        max_length=int(config["model"]["max_length"]),
        shuffle=True,
        num_workers=int(config["train"]["num_workers"]),
    )
    dev_loader = build_dataloader(
        dataset_bundle["dev_dataset"],
        tokenizer=tokenizer,
        batch_size=int(config["train"]["batch_size"]),
        max_length=int(config["model"]["max_length"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
    )

    model = RobertaForSVDAClassification(
        pretrained_model_name_or_path=config["model"]["pretrained_model_name_or_path"],
        num_labels=num_labels,
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
        candidate_map=candidate_map,
        lambda_hybrid=effective_lambda_hybrid,
        lambda_global=effective_lambda_global,
        ranking_margin=ranking_margin,
        top_k_hybrid=top_k_hybrid,
        top_k_global=top_k_global,
        start_epoch=start_epoch,
    )
    warmup_checkpoint = torch.load(warmup_checkpoint_path, map_location=device)
    model.load_encoder_state_dict_from_checkpoint(warmup_checkpoint)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    total_training_steps = len(train_loader) * int(config["train"]["num_epochs"])
    warmup_steps = math.ceil(total_training_steps * float(config["train"]["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    training_log_path = run_paths["logs"] / "svda_finetune_log.jsonl"
    best_metric_name = config["eval"]["primary_metric"]
    best_score = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0
    best_checkpoint = run_paths["svda_checkpoints"] / "best.pt"

    for epoch in range(1, int(config["train"]["num_epochs"]) + 1):
        train_losses = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        dev_metrics, _, _ = evaluate(model, dev_loader, device, id2label)
        current_score = float(dev_metrics[best_metric_name])
        is_best = current_score > best_score
        if is_best:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0
            save_best_checkpoint(best_checkpoint, model, epoch, dev_metrics)
        else:
            epochs_without_improvement += 1

        append_jsonl(
            training_log_path,
            {
                "epoch": epoch,
                "train_loss": train_losses["train_loss"],
                "train_ce_loss": train_losses["train_ce_loss"],
                "train_hybrid_ranking_loss": train_losses["train_hybrid_ranking_loss"],
                "train_global_fallback_ranking_loss": train_losses["train_global_fallback_ranking_loss"],
                "lambda_hybrid": effective_lambda_hybrid,
                "lambda_global": effective_lambda_global,
                "candidate_mode": args.candidate_mode,
                "ranking_margin": ranking_margin,
                "top_k_hybrid": top_k_hybrid,
                "top_k_global": top_k_global,
                "start_epoch": start_epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "dev_accuracy": dev_metrics["accuracy"],
                "dev_macro_precision": dev_metrics["macro_precision"],
                "dev_macro_recall": dev_metrics["macro_recall"],
                "dev_macro_f1": dev_metrics["macro_f1"],
                "is_best": is_best,
            },
        )
        print(
            "Epoch {epoch}: val_f1={f1:.6f} val_acc={accuracy:.6f}".format(
                epoch=epoch,
                accuracy=dev_metrics["accuracy"],
                f1=dev_metrics["macro_f1"],
            )
        )

        if epochs_without_improvement >= int(config["train"]["early_stopping_patience"]):
            break

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_dev_metrics, _, _ = evaluate(model, dev_loader, device, id2label)
    num_value_dims = int(warmup_checkpoint.get("num_value_dims", 0))

    metrics_payload = {
        "run_dir": str(run_paths["svda_run_dir"]).replace("\\", "/"),
        "seed": args.seed,
        "best_epoch": int(checkpoint["epoch"]),
        "best_dev_accuracy": final_dev_metrics["accuracy"],
        "best_dev_macro_precision": final_dev_metrics["macro_precision"],
        "best_dev_macro_recall": final_dev_metrics["macro_recall"],
        "best_dev_macro_f1": final_dev_metrics["macro_f1"],
        "num_train_examples": len(dataset_bundle["train_dataset"]),
        "num_dev_examples": len(dataset_bundle["dev_dataset"]),
        "num_value_dims": num_value_dims,
        "device": str(device),
        "primary_metric": best_metric_name,
        "warmup_checkpoint": str(warmup_checkpoint_path).replace("\\", "/"),
        "candidate_mode": args.candidate_mode,
        "lambda_hybrid": effective_lambda_hybrid,
        "lambda_global": effective_lambda_global,
        "ranking_margin": ranking_margin,
        "top_k_hybrid": top_k_hybrid,
        "top_k_global": top_k_global,
        "start_epoch": start_epoch,
        "candidate_labels_total": sum(len(candidate_ids) for candidate_ids in candidate_map.values()),
        "best_checkpoint": str(best_checkpoint).replace("\\", "/"),
    }
    save_json(metrics_payload, run_paths["metrics"] / "metrics.json")
    save_json(
        {"per_class_metrics": final_dev_metrics["per_class_metrics"]},
        run_paths["svda_finetune"] / "per_class_metrics.json",
    )
    save_confusion_matrix_csv(
        final_dev_metrics["confusion_matrix"],
        final_dev_metrics["labels"],
        run_paths["svda_finetune"] / "confusion_matrix.csv",
    )
    if bool(config["runtime"]["save_confusion_matrix_png"]):
        save_confusion_matrix_png(
            final_dev_metrics["confusion_matrix"],
            final_dev_metrics["labels"],
            run_paths["svda_finetune"] / "confusion_matrix.png",
        )

    save_json(
        {
            "seed": args.seed,
            "run_dir": str(run_paths["svda_run_dir"]).replace("\\", "/"),
            "stage": "svda_finetune",
            "best_epoch": metrics_payload["best_epoch"],
            "device": str(device),
            "num_value_dims": num_value_dims,
            "warmup_checkpoint": str(warmup_checkpoint_path).replace("\\", "/"),
            "hybrid_ranking_config": {
                "enabled": ranking_enabled,
                "candidate_mode": args.candidate_mode,
                "lambda_hybrid": effective_lambda_hybrid,
                "lambda_global": effective_lambda_global,
                "margin": ranking_margin,
                "top_k_hybrid": top_k_hybrid,
                "top_k_global": top_k_global,
                "start_epoch": start_epoch,
                "candidate_map": {str(label_id): list(candidate_ids) for label_id, candidate_ids in candidate_map.items()},
            },
            "train_config": config["train"],
            "model_config": config["model"],
            "data_config": config["data"],
        },
        run_paths["artifacts"] / "run_info.json",
    )

    summary_path = refresh_expF_new_summary_csv(config["experiment"]["output_root"])
    print(
        "SVDA fine-tune finished. Best validation F1={f1:.6f} acc={acc:.6f}".format(
            f1=metrics_payload["best_dev_macro_f1"],
            acc=metrics_payload["best_dev_accuracy"],
        )
    )
    print(f"Summary written to {summary_path}")
    print(f"Run directory: {run_paths['svda_run_dir']}")


if __name__ == "__main__":
    main()
