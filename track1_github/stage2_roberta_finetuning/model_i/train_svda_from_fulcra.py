"""ExpI fine-tune: FULCRA-warmup encoder + Hybrid/Global ranking + R-Drop + FGM + EMA.

Evaluation goes through the canonical `metrics.compute_classification_metrics`
re-exported from `expI/metrics.py`. The wrapper aborts training if the
canonical evaluation file has drifted from its baseline SHA-256.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Make sibling modules importable when invoking as a script.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from data import build_dataloader, build_svda_datasets
from ema import EMA
from fgm import FGM
from metrics import compute_classification_metrics, verify_canonical_metrics
from models import RobertaForSVDAClassificationI
from utils import (
    append_jsonl,
    copy_config_file,
    create_expI_run_paths,
    ensure_local_model_path,
    load_yaml,
    refresh_expI_summary_csv,
    resolve_device,
    save_confusion_matrix_csv,
    save_confusion_matrix_png,
    save_json,
    save_label_mappings,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExpI: R-Drop + FGM + EMA fine-tune on SVDA.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--warmup_ckpt", required=True)
    parser.add_argument("--candidate_mode", default=None,
                        help="Override hybrid_ranking.candidate_mode in config.")
    parser.add_argument("--lambda_hybrid", type=float, default=None)
    parser.add_argument("--lambda_global", type=float, default=None)
    parser.add_argument("--lambda_rdrop", type=float, default=None,
                        help="R-Drop KL weight (symmetric).")
    parser.add_argument("--fgm_epsilon", type=float, default=None,
                        help="FGM perturbation epsilon. Set <=0 to disable.")
    parser.add_argument("--ema_decay", type=float, default=None,
                        help="EMA decay rate. Set <=0 to disable.")
    parser.add_argument("--use_question", type=int, default=None,
                        help="1 to prepend Question to text; 0 to disable. Overrides config.")
    parser.add_argument("--tag", default=None,
                        help="Output sub-directory name for this run.")
    return parser.parse_args()


def move_batch_to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
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
    metrics = compute_classification_metrics(
        labels=labels, predictions=predictions, id2label=id2label
    )
    return metrics, predictions, labels


def _dedupe_without_gold(ids: Sequence[int], gold_id: int) -> Tuple[int, ...]:
    out: List[int] = []
    for i in ids:
        i = int(i)
        if i == gold_id or i in out:
            continue
        out.append(i)
    return tuple(out)


def _validate_label_list(labels: Sequence[str], label2id: Dict[str, int], name: str) -> None:
    missing = sorted({str(l) for l in labels if str(l) not in label2id})
    if missing:
        raise ValueError(f"{name} contains labels absent from label2id: {', '.join(missing)}")


def build_candidate_map(
    label2id: Dict[str, int],
    circular_order: Sequence[str],
    candidate_mode: str,
    core_groups: Sequence[Sequence[str]],
) -> Dict[int, Tuple[int, ...]]:
    if candidate_mode not in {"adjacent_1", "adjacent_2", "core_plus_adjacent_1"}:
        raise ValueError(
            "candidate_mode must be one of adjacent_1, adjacent_2, core_plus_adjacent_1"
        )
    _validate_label_list(circular_order, label2id, "circular_order")
    if len(set(circular_order)) != len(circular_order):
        raise ValueError("circular_order must not contain duplicate labels.")
    if set(circular_order) != set(label2id):
        missing_from_order = sorted(set(label2id) - set(circular_order))
        extra_in_order = sorted(set(circular_order) - set(label2id))
        raise ValueError(
            f"circular_order coverage mismatch. missing={missing_from_order}, extra={extra_in_order}"
        )
    for i, group in enumerate(core_groups):
        _validate_label_list(group, label2id, f"core_groups[{i}]")

    order_ids = [label2id[str(l)] for l in circular_order]
    radius = 2 if candidate_mode == "adjacent_2" else 1
    candidate_map: Dict[int, Tuple[int, ...]] = {}
    total = len(order_ids)
    for idx, gold_id in enumerate(order_ids):
        cand: List[int] = []
        for offset in range(radius, 0, -1):
            cand.append(order_ids[(idx - offset) % total])
        for offset in range(1, radius + 1):
            cand.append(order_ids[(idx + offset) % total])
        candidate_map[gold_id] = _dedupe_without_gold(cand, gold_id)
    if candidate_mode == "core_plus_adjacent_1":
        for group in core_groups:
            group_ids = [label2id[str(l)] for l in group]
            for gold_id in group_ids:
                candidate_map[gold_id] = _dedupe_without_gold(
                    list(candidate_map.get(gold_id, ())) + group_ids, gold_id
                )
    return candidate_map


def compute_total_loss(
    out: Dict[str, torch.Tensor],
    lambda_hybrid: float,
    lambda_global: float,
    epoch: int,
    start_epoch: int,
) -> torch.Tensor:
    ce = out["ce_loss"]
    if epoch < start_epoch:
        return ce
    return (
        ce
        + float(lambda_hybrid) * out["hybrid_ranking_loss"]
        + float(lambda_global) * out["global_fallback_ranking_loss"]
    )


def symmetric_kl(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    log_p1 = F.log_softmax(logits1, dim=-1)
    log_p2 = F.log_softmax(logits2, dim=-1)
    p1 = log_p1.exp()
    p2 = log_p2.exp()
    kl_12 = F.kl_div(log_p1, p2, reduction="batchmean")
    kl_21 = F.kl_div(log_p2, p1, reduction="batchmean")
    return 0.5 * (kl_12 + kl_21)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch: int,
    lambda_hybrid: float,
    lambda_global: float,
    start_epoch: int,
    lambda_rdrop: float,
    fgm,
    ema,
) -> Dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "ce": 0.0,
        "hybrid": 0.0,
        "global": 0.0,
        "rdrop_kl": 0.0,
        "adv": 0.0,
    }
    step_count = 0
    for batch in dataloader:
        tensors = move_batch_to_device(batch, device)
        optimizer.zero_grad()

        out1 = model(**tensors)
        loss_clean1 = compute_total_loss(
            out1, lambda_hybrid, lambda_global, epoch, start_epoch
        )

        if lambda_rdrop > 0.0:
            out2 = model(**tensors)
            loss_clean2 = compute_total_loss(
                out2, lambda_hybrid, lambda_global, epoch, start_epoch
            )
            kl = symmetric_kl(out1["logits"], out2["logits"])
            loss_clean = 0.5 * (loss_clean1 + loss_clean2) + lambda_rdrop * kl
            kl_value = float(kl.detach().item())
        else:
            loss_clean = loss_clean1
            kl_value = 0.0

        loss_clean.backward()

        if fgm is not None:
            fgm.attack()
            out_adv = model(**tensors)
            loss_adv = compute_total_loss(
                out_adv, lambda_hybrid, lambda_global, epoch, start_epoch
            )
            loss_adv.backward()
            fgm.restore()
            adv_value = float(loss_adv.detach().item())
        else:
            adv_value = 0.0

        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        totals["loss"] += float(loss_clean.detach().item())
        totals["ce"] += float(out1["ce_loss"].detach().item())
        totals["hybrid"] += float(out1["hybrid_ranking_loss"].detach().item())
        totals["global"] += float(out1["global_fallback_ranking_loss"].detach().item())
        totals["rdrop_kl"] += kl_value
        totals["adv"] += adv_value
        step_count += 1

    denom = max(step_count, 1)
    return {f"train_{k}": v / denom for k, v in totals.items()}


def save_checkpoint(path: Path, model, epoch: int, metrics: Dict[str, object]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_metrics": metrics,
        },
        path,
    )


def main() -> None:
    verify_canonical_metrics()
    args = parse_args()
    config = load_yaml(args.config)
    config["train"]["seed"] = args.seed

    # Apply CLI overrides for the regularization knobs.
    reg_cfg = config.get("regularization", {}) or {}
    lambda_rdrop = float(args.lambda_rdrop if args.lambda_rdrop is not None
                         else reg_cfg.get("r_drop_lambda", 0.0))
    fgm_epsilon = float(args.fgm_epsilon if args.fgm_epsilon is not None
                        else reg_cfg.get("fgm_epsilon", 0.0))
    ema_decay = float(args.ema_decay if args.ema_decay is not None
                      else reg_cfg.get("ema_decay", 0.0))

    if args.use_question is not None:
        config["data"]["use_question"] = bool(int(args.use_question))

    ranking_cfg = config.get("hybrid_ranking", {}) or {}
    candidate_mode = str(
        args.candidate_mode
        if args.candidate_mode is not None
        else ranking_cfg.get("candidate_mode", "adjacent_2")
    )
    lambda_hybrid = float(
        args.lambda_hybrid
        if args.lambda_hybrid is not None
        else ranking_cfg.get("lambda_hybrid", 0.1)
    )
    lambda_global = float(
        args.lambda_global
        if args.lambda_global is not None
        else ranking_cfg.get("lambda_global", 0.05)
    )
    ranking_margin = float(ranking_cfg.get("margin", 1.0))
    top_k_hybrid = int(ranking_cfg.get("top_k_hybrid", 1))
    top_k_global = int(ranking_cfg.get("top_k_global", 1))
    start_epoch = int(ranking_cfg.get("start_epoch", 2))

    warmup_checkpoint_path = Path(args.warmup_ckpt)
    if not warmup_checkpoint_path.exists():
        raise FileNotFoundError(f"Warm-up checkpoint not found: {warmup_checkpoint_path}")

    set_seed(args.seed)
    ensure_local_model_path(str(config["model"]["pretrained_model_name_or_path"]))
    tag = str(args.tag or f"{candidate_mode}_rdrop{lambda_rdrop}_fgm{fgm_epsilon}_ema{ema_decay}_q{int(bool(config['data'].get('use_question', False)))}")
    run_paths = create_expI_run_paths(
        str(config["experiment"]["output_root"]), args.seed, tag
    )
    device = resolve_device(str(config["runtime"]["device"]))

    copy_config_file(args.config, run_paths["artifacts"] / "config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(
        str(config["model"]["pretrained_model_name_or_path"]), local_files_only=True
    )

    dataset_bundle = build_svda_datasets(config)
    label2id = dataset_bundle["label2id"]
    id2label = dataset_bundle["id2label"]
    candidate_map = build_candidate_map(
        label2id=label2id,
        circular_order=ranking_cfg.get("circular_order", []),
        candidate_mode=candidate_mode,
        core_groups=ranking_cfg.get("core_groups", []),
    )

    num_labels = len(label2id)
    if num_labels != int(config["model"]["num_labels"]):
        raise ValueError(
            f"Config expects {config['model']['num_labels']} labels, got {num_labels}."
        )
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

    model = RobertaForSVDAClassificationI(
        pretrained_model_name_or_path=str(config["model"]["pretrained_model_name_or_path"]),
        num_labels=num_labels,
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
        candidate_map=candidate_map,
        ranking_margin=ranking_margin,
        top_k_hybrid=top_k_hybrid,
        top_k_global=top_k_global,
    )
    warmup_checkpoint = torch.load(warmup_checkpoint_path, map_location=device)
    model.load_encoder_state_dict_from_checkpoint(warmup_checkpoint)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    total_steps = len(train_loader) * int(config["train"]["num_epochs"])
    warmup_steps = math.ceil(total_steps * float(config["train"]["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    fgm = FGM(model, emb_name="word_embeddings", epsilon=fgm_epsilon) if fgm_epsilon > 0 else None
    ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None

    training_log_path = run_paths["logs"] / "svda_finetune_log.jsonl"
    best_metric_name = str(config["eval"]["primary_metric"])
    best_score = float("-inf")
    best_epoch = -1
    best_payload: Dict[str, object] = {}
    epochs_without_improvement = 0
    best_checkpoint_path = run_paths["svda_checkpoints"] / "best.pt"
    best_ema_checkpoint_path = run_paths["svda_checkpoints"] / "best_ema.pt"
    used_ema_for_best = False

    for epoch in range(1, int(config["train"]["num_epochs"]) + 1):
        train_losses = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            lambda_hybrid=lambda_hybrid,
            lambda_global=lambda_global,
            start_epoch=start_epoch,
            lambda_rdrop=lambda_rdrop,
            fgm=fgm,
            ema=ema,
        )

        raw_dev_metrics, _, _ = evaluate(model, dev_loader, device, id2label)

        ema_dev_metrics: Dict[str, object] = {}
        if ema is not None:
            ema.apply_shadow(model)
            ema_dev_metrics, _, _ = evaluate(model, dev_loader, device, id2label)
            ema.restore(model)

        # Pick whichever (raw or EMA) gave the better primary metric this epoch.
        if ema_dev_metrics and ema_dev_metrics[best_metric_name] >= raw_dev_metrics[best_metric_name]:
            epoch_best_metrics = ema_dev_metrics
            epoch_used_ema = True
        else:
            epoch_best_metrics = raw_dev_metrics
            epoch_used_ema = False

        current_score = float(epoch_best_metrics[best_metric_name])
        is_best = current_score > best_score
        if is_best:
            best_score = current_score
            best_epoch = epoch
            best_payload = epoch_best_metrics
            used_ema_for_best = epoch_used_ema
            epochs_without_improvement = 0
            if epoch_used_ema and ema is not None:
                ema.apply_shadow(model)
                save_checkpoint(best_ema_checkpoint_path, model, epoch, epoch_best_metrics)
                ema.restore(model)
            else:
                save_checkpoint(best_checkpoint_path, model, epoch, epoch_best_metrics)
        else:
            epochs_without_improvement += 1

        log_entry = {
            "epoch": epoch,
            **train_losses,
            "lambda_hybrid": lambda_hybrid,
            "lambda_global": lambda_global,
            "lambda_rdrop": lambda_rdrop,
            "fgm_epsilon": fgm_epsilon,
            "ema_decay": ema_decay,
            "candidate_mode": candidate_mode,
            "start_epoch": start_epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "raw_dev_accuracy": raw_dev_metrics["accuracy"],
            "raw_dev_macro_precision": raw_dev_metrics["macro_precision"],
            "raw_dev_macro_recall": raw_dev_metrics["macro_recall"],
            "raw_dev_macro_f1": raw_dev_metrics["macro_f1"],
            "ema_dev_accuracy": ema_dev_metrics.get("accuracy", None) if ema_dev_metrics else None,
            "ema_dev_macro_precision": ema_dev_metrics.get("macro_precision", None) if ema_dev_metrics else None,
            "ema_dev_macro_recall": ema_dev_metrics.get("macro_recall", None) if ema_dev_metrics else None,
            "ema_dev_macro_f1": ema_dev_metrics.get("macro_f1", None) if ema_dev_metrics else None,
            "selected_used_ema": epoch_used_ema,
            "is_best": is_best,
        }
        append_jsonl(training_log_path, log_entry)
        print(
            f"Epoch {epoch}: "
            f"raw_f1={raw_dev_metrics['macro_f1']:.6f} raw_acc={raw_dev_metrics['accuracy']:.6f}"
            + (
                f" | ema_f1={ema_dev_metrics['macro_f1']:.6f} ema_acc={ema_dev_metrics['accuracy']:.6f}"
                if ema_dev_metrics else ""
            )
            + (" *" if is_best else "")
        )

        if epochs_without_improvement >= int(config["train"]["early_stopping_patience"]):
            break

    if used_ema_for_best:
        chosen_ckpt = best_ema_checkpoint_path
    else:
        chosen_ckpt = best_checkpoint_path

    checkpoint = torch.load(chosen_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_dev_metrics, _, _ = evaluate(model, dev_loader, device, id2label)
    num_value_dims = int(warmup_checkpoint.get("num_value_dims", 0))

    metrics_payload = {
        "run_dir": str(run_paths["svda_run_dir"]).replace("\\", "/"),
        "seed": args.seed,
        "candidate_mode": candidate_mode,
        "lambda_hybrid": lambda_hybrid,
        "lambda_global": lambda_global,
        "lambda_rdrop": lambda_rdrop,
        "fgm_epsilon": fgm_epsilon,
        "ema_decay": ema_decay,
        "use_question": bool(config["data"].get("use_question", False)),
        "best_epoch": int(checkpoint["epoch"]),
        "best_dev_accuracy": final_dev_metrics["accuracy"],
        "best_dev_macro_precision": final_dev_metrics["macro_precision"],
        "best_dev_macro_recall": final_dev_metrics["macro_recall"],
        "best_dev_macro_f1": final_dev_metrics["macro_f1"],
        "ema_dev_accuracy": best_payload.get("accuracy") if used_ema_for_best else None,
        "ema_dev_macro_precision": best_payload.get("macro_precision") if used_ema_for_best else None,
        "ema_dev_macro_recall": best_payload.get("macro_recall") if used_ema_for_best else None,
        "ema_dev_macro_f1": best_payload.get("macro_f1") if used_ema_for_best else None,
        "selected_used_ema": bool(used_ema_for_best),
        "ranking_margin": ranking_margin,
        "top_k_hybrid": top_k_hybrid,
        "top_k_global": top_k_global,
        "start_epoch": start_epoch,
        "num_train_examples": len(dataset_bundle["train_dataset"]),
        "num_dev_examples": len(dataset_bundle["dev_dataset"]),
        "num_value_dims": num_value_dims,
        "device": str(device),
        "primary_metric": best_metric_name,
        "warmup_checkpoint": str(warmup_checkpoint_path).replace("\\", "/"),
        "best_checkpoint": str(chosen_ckpt).replace("\\", "/"),
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
        try:
            save_confusion_matrix_png(
                final_dev_metrics["confusion_matrix"],
                final_dev_metrics["labels"],
                run_paths["svda_finetune"] / "confusion_matrix.png",
            )
        except Exception as exc:
            print(f"[warn] confusion-matrix PNG skipped: {exc}")

    summary_path = refresh_expI_summary_csv(str(config["experiment"]["output_root"]))
    print(
        "ExpI fine-tune finished. "
        f"Best dev macro_f1={metrics_payload['best_dev_macro_f1']:.6f} "
        f"acc={metrics_payload['best_dev_accuracy']:.6f} "
        f"macro_precision={metrics_payload['best_dev_macro_precision']:.6f} "
        f"(epoch={metrics_payload['best_epoch']}, ema={used_ema_for_best})"
    )
    print(f"Summary written to {summary_path}")
    print(f"Run directory: {run_paths['svda_run_dir']}")


if __name__ == "__main__":
    main()
