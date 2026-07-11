"""ExpJ training: FocalLoss + Sibling-Aware ranking + Class-balanced sampler
+ R-Drop + FGM + EMA, built on the FULCRA-warmup encoder.
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

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from data import build_eval_dataloader, build_svda_datasets, build_train_dataloader
from ema import EMA
from fgm import FGM
from metrics import compute_classification_metrics, verify_canonical_metrics
from models import RobertaForSVDAClassificationJ
from utils import (
    append_jsonl,
    build_sibling_pairs,
    copy_config_file,
    create_expJ_run_paths,
    ensure_local_model_path,
    load_yaml,
    refresh_expJ_summary_csv,
    resolve_device,
    save_confusion_matrix_csv,
    save_json,
    save_label_mappings,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ExpJ: Focal + Sibling Margin + Class-Balanced + R-Drop + FGM + EMA.")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--warmup_ckpt", required=True)
    p.add_argument("--candidate_mode", default=None)
    p.add_argument("--lambda_hybrid", type=float, default=None)
    p.add_argument("--lambda_global", type=float, default=None)
    p.add_argument("--lambda_rdrop", type=float, default=None)
    p.add_argument("--fgm_epsilon", type=float, default=None)
    p.add_argument("--ema_decay", type=float, default=None)
    p.add_argument("--focal_gamma", type=float, default=None)
    p.add_argument("--sibling_margin", type=float, default=None)
    p.add_argument("--class_balance_power", type=float, default=None)
    p.add_argument("--tag", default=None)
    return p.parse_args()


def move_batch_to_device(batch, device):
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
    }


def evaluate(model, dataloader, device, id2label):
    model.eval()
    preds: List[int] = []
    labels: List[int] = []
    with torch.no_grad():
        for batch in dataloader:
            t = move_batch_to_device(batch, device)
            out = model(**t)
            preds.extend(out["logits"].argmax(-1).cpu().tolist())
            labels.extend(t["labels"].cpu().tolist())
    metrics = compute_classification_metrics(labels=labels, predictions=preds, id2label=id2label)
    return metrics, preds, labels


def _dedupe(ids, gold):
    out: List[int] = []
    for i in ids:
        i = int(i)
        if i == gold or i in out:
            continue
        out.append(i)
    return tuple(out)


def build_candidate_map(label2id, circular_order, candidate_mode, core_groups):
    if candidate_mode not in {"adjacent_1", "adjacent_2", "core_plus_adjacent_1"}:
        raise ValueError("bad candidate_mode")
    if set(circular_order) != set(label2id):
        raise ValueError("circular_order mismatch with labels")
    order_ids = [label2id[str(l)] for l in circular_order]
    radius = 2 if candidate_mode == "adjacent_2" else 1
    cmap: Dict[int, Tuple[int, ...]] = {}
    n = len(order_ids)
    for idx, gold in enumerate(order_ids):
        cand: List[int] = []
        for off in range(radius, 0, -1):
            cand.append(order_ids[(idx - off) % n])
        for off in range(1, radius + 1):
            cand.append(order_ids[(idx + off) % n])
        cmap[gold] = _dedupe(cand, gold)
    if candidate_mode == "core_plus_adjacent_1":
        for group in core_groups:
            gids = [label2id[str(l)] for l in group]
            for gold in gids:
                cmap[gold] = _dedupe(list(cmap.get(gold, ())) + gids, gold)
    return cmap


def compute_clean_total(out, lambda_hybrid, lambda_global, epoch, start_epoch):
    base = out["focal_loss"]
    if epoch < start_epoch:
        return base
    return base + float(lambda_hybrid) * out["hybrid_ranking_loss"] + float(lambda_global) * out["global_fallback_ranking_loss"]


def symmetric_kl(logits1, logits2):
    lp1 = F.log_softmax(logits1, dim=-1)
    lp2 = F.log_softmax(logits2, dim=-1)
    return 0.5 * (F.kl_div(lp1, lp2.exp(), reduction="batchmean") + F.kl_div(lp2, lp1.exp(), reduction="batchmean"))


def train_one_epoch(
    model, loader, optimizer, scheduler, device, epoch,
    lambda_hybrid, lambda_global, start_epoch,
    lambda_rdrop, fgm, ema,
):
    model.train()
    totals = {k: 0.0 for k in ("loss", "focal", "hybrid", "global", "rdrop_kl", "adv")}
    steps = 0
    for batch in loader:
        tensors = move_batch_to_device(batch, device)
        optimizer.zero_grad()

        out1 = model(**tensors)
        loss1 = compute_clean_total(out1, lambda_hybrid, lambda_global, epoch, start_epoch)

        if lambda_rdrop > 0.0:
            out2 = model(**tensors)
            loss2 = compute_clean_total(out2, lambda_hybrid, lambda_global, epoch, start_epoch)
            kl = symmetric_kl(out1["logits"], out2["logits"])
            total_clean = 0.5 * (loss1 + loss2) + lambda_rdrop * kl
            kl_v = float(kl.detach().item())
        else:
            total_clean = loss1
            kl_v = 0.0

        total_clean.backward()

        if fgm is not None:
            fgm.attack()
            out_adv = model(**tensors)
            adv = compute_clean_total(out_adv, lambda_hybrid, lambda_global, epoch, start_epoch)
            adv.backward()
            fgm.restore()
            adv_v = float(adv.detach().item())
        else:
            adv_v = 0.0

        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        totals["loss"] += float(total_clean.detach().item())
        totals["focal"] += float(out1["focal_loss"].detach().item())
        totals["hybrid"] += float(out1["hybrid_ranking_loss"].detach().item())
        totals["global"] += float(out1["global_fallback_ranking_loss"].detach().item())
        totals["rdrop_kl"] += kl_v
        totals["adv"] += adv_v
        steps += 1

    denom = max(steps, 1)
    return {f"train_{k}": v / denom for k, v in totals.items()}


def save_checkpoint(path: Path, model, epoch: int, metrics: Dict[str, object]) -> None:
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": epoch, "best_metrics": metrics},
        path,
    )


def main() -> None:
    verify_canonical_metrics()
    args = parse_args()
    config = load_yaml(args.config)
    config["train"]["seed"] = args.seed

    reg = config.get("regularization", {}) or {}
    focal_cfg = config.get("focal", {}) or {}
    sibling_cfg = config.get("sibling", {}) or {}

    lambda_rdrop = float(args.lambda_rdrop if args.lambda_rdrop is not None else reg.get("r_drop_lambda", 1.0))
    fgm_epsilon = float(args.fgm_epsilon if args.fgm_epsilon is not None else reg.get("fgm_epsilon", 1.0))
    ema_decay = float(args.ema_decay if args.ema_decay is not None else reg.get("ema_decay", 0.999))
    focal_gamma = float(args.focal_gamma if args.focal_gamma is not None else focal_cfg.get("gamma", 2.0))
    sibling_margin = float(args.sibling_margin if args.sibling_margin is not None else sibling_cfg.get("sibling_margin", 1.5))
    class_balance_power = float(args.class_balance_power if args.class_balance_power is not None else config.get("sampling", {}).get("class_balance_power", 0.5))

    ranking_cfg = config.get("hybrid_ranking", {}) or {}
    candidate_mode = str(args.candidate_mode if args.candidate_mode is not None else ranking_cfg.get("candidate_mode", "adjacent_2"))
    lambda_hybrid = float(args.lambda_hybrid if args.lambda_hybrid is not None else ranking_cfg.get("lambda_hybrid", 0.1))
    lambda_global = float(args.lambda_global if args.lambda_global is not None else ranking_cfg.get("lambda_global", 0.05))
    default_margin = float(ranking_cfg.get("margin", 1.0))
    top_k_hybrid = int(ranking_cfg.get("top_k_hybrid", 1))
    top_k_global = int(ranking_cfg.get("top_k_global", 1))
    start_epoch = int(ranking_cfg.get("start_epoch", 2))

    warmup_ckpt = Path(args.warmup_ckpt)
    if not warmup_ckpt.exists():
        raise FileNotFoundError(f"Warm-up ckpt missing: {warmup_ckpt}")

    set_seed(args.seed)
    ensure_local_model_path(str(config["model"]["pretrained_model_name_or_path"]))
    tag = str(args.tag or f"focal{focal_gamma}_sib{sibling_margin}_cb{class_balance_power}_rdrop{lambda_rdrop}_adj2")
    run_paths = create_expJ_run_paths(str(config["experiment"]["output_root"]), args.seed, tag)
    device = resolve_device(str(config["runtime"]["device"]))

    copy_config_file(args.config, run_paths["artifacts"] / "config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(
        str(config["model"]["pretrained_model_name_or_path"]), local_files_only=True
    )
    bundle = build_svda_datasets(config)
    label2id = bundle["label2id"]
    id2label = bundle["id2label"]
    candidate_map = build_candidate_map(
        label2id=label2id,
        circular_order=ranking_cfg.get("circular_order", []),
        candidate_mode=candidate_mode,
        core_groups=ranking_cfg.get("core_groups", []),
    )
    sibling_pairs = build_sibling_pairs(label2id, sibling_cfg.get("sibling_groups", []))
    num_labels = len(label2id)
    save_label_mappings(label2id, id2label, run_paths["artifacts"])

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = build_train_dataloader(
        bundle["train_dataset"], tokenizer=tokenizer,
        batch_size=int(config["train"]["batch_size"]),
        max_length=int(config["model"]["max_length"]),
        num_workers=int(config["train"]["num_workers"]),
        sampling=str(config.get("sampling", {}).get("strategy", "weighted")),
        class_balance_power=class_balance_power,
        epochs_target_size=int(config["train"].get("epochs_target_size", len(bundle["train_dataset"]))),
        generator=g,
    )
    dev_loader = build_eval_dataloader(
        bundle["dev_dataset"], tokenizer=tokenizer,
        batch_size=int(config["train"]["batch_size"]),
        max_length=int(config["model"]["max_length"]),
        num_workers=int(config["train"]["num_workers"]),
    )

    model = RobertaForSVDAClassificationJ(
        pretrained_model_name_or_path=str(config["model"]["pretrained_model_name_or_path"]),
        num_labels=num_labels,
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
        focal_gamma=focal_gamma,
        candidate_map=candidate_map,
        sibling_pairs=sibling_pairs,
        default_margin=default_margin,
        sibling_margin=sibling_margin,
        top_k_hybrid=top_k_hybrid,
        top_k_global=top_k_global,
    )
    warmup_ckpt_data = torch.load(warmup_ckpt, map_location=device)
    model.load_encoder_state_dict_from_checkpoint(warmup_ckpt_data)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    total_steps = len(train_loader) * int(config["train"]["num_epochs"])
    warmup_steps = math.ceil(total_steps * float(config["train"]["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    fgm = FGM(model, emb_name="word_embeddings", epsilon=fgm_epsilon) if fgm_epsilon > 0 else None
    ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None

    log_path = run_paths["logs"] / "svda_finetune_log.jsonl"
    best_metric = str(config["eval"]["primary_metric"])
    best_score = float("-inf")
    best_epoch = -1
    best_payload: Dict[str, object] = {}
    patience = 0
    best_raw_ckpt = run_paths["svda_checkpoints"] / "best.pt"
    best_ema_ckpt = run_paths["svda_checkpoints"] / "best_ema.pt"
    used_ema = False

    for epoch in range(1, int(config["train"]["num_epochs"]) + 1):
        losses_avg = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, scheduler=scheduler,
            device=device, epoch=epoch,
            lambda_hybrid=lambda_hybrid, lambda_global=lambda_global, start_epoch=start_epoch,
            lambda_rdrop=lambda_rdrop, fgm=fgm, ema=ema,
        )
        raw_metrics, _, _ = evaluate(model, dev_loader, device, id2label)
        ema_metrics: Dict[str, object] = {}
        if ema is not None:
            ema.apply_shadow(model)
            ema_metrics, _, _ = evaluate(model, dev_loader, device, id2label)
            ema.restore(model)

        if ema_metrics and ema_metrics[best_metric] >= raw_metrics[best_metric]:
            epoch_best = ema_metrics
            epoch_used_ema = True
        else:
            epoch_best = raw_metrics
            epoch_used_ema = False

        cur = float(epoch_best[best_metric])
        is_best = cur > best_score
        if is_best:
            best_score = cur
            best_epoch = epoch
            best_payload = epoch_best
            used_ema = epoch_used_ema
            patience = 0
            if epoch_used_ema and ema is not None:
                ema.apply_shadow(model)
                save_checkpoint(best_ema_ckpt, model, epoch, epoch_best)
                ema.restore(model)
            else:
                save_checkpoint(best_raw_ckpt, model, epoch, epoch_best)
        else:
            patience += 1

        append_jsonl(log_path, {
            "epoch": epoch, **losses_avg,
            "focal_gamma": focal_gamma, "sibling_margin": sibling_margin,
            "class_balance_power": class_balance_power,
            "lambda_hybrid": lambda_hybrid, "lambda_global": lambda_global,
            "lambda_rdrop": lambda_rdrop,
            "fgm_epsilon": fgm_epsilon, "ema_decay": ema_decay,
            "candidate_mode": candidate_mode, "start_epoch": start_epoch,
            "raw_dev_accuracy": raw_metrics["accuracy"],
            "raw_dev_macro_precision": raw_metrics["macro_precision"],
            "raw_dev_macro_recall": raw_metrics["macro_recall"],
            "raw_dev_macro_f1": raw_metrics["macro_f1"],
            "ema_dev_accuracy": ema_metrics.get("accuracy") if ema_metrics else None,
            "ema_dev_macro_precision": ema_metrics.get("macro_precision") if ema_metrics else None,
            "ema_dev_macro_recall": ema_metrics.get("macro_recall") if ema_metrics else None,
            "ema_dev_macro_f1": ema_metrics.get("macro_f1") if ema_metrics else None,
            "selected_used_ema": epoch_used_ema,
            "is_best": is_best,
        })
        line = f"Epoch {epoch}: raw_f1={raw_metrics['macro_f1']:.6f} raw_acc={raw_metrics['accuracy']:.6f}"
        if ema_metrics:
            line += f" | ema_f1={ema_metrics['macro_f1']:.6f} ema_acc={ema_metrics['accuracy']:.6f}"
        if is_best:
            line += " *"
        print(line)

        if patience >= int(config["train"]["early_stopping_patience"]):
            break

    chosen_ckpt = best_ema_ckpt if used_ema else best_raw_ckpt
    ckpt = torch.load(chosen_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    final_metrics, _, _ = evaluate(model, dev_loader, device, id2label)

    payload = {
        "run_dir": str(run_paths["svda_run_dir"]).replace("\\", "/"),
        "seed": args.seed,
        "candidate_mode": candidate_mode,
        "focal_gamma": focal_gamma,
        "lambda_hybrid": lambda_hybrid, "lambda_global": lambda_global,
        "lambda_rdrop": lambda_rdrop,
        "default_margin": default_margin, "sibling_margin": sibling_margin,
        "class_balance_power": class_balance_power,
        "fgm_epsilon": fgm_epsilon, "ema_decay": ema_decay,
        "best_epoch": int(ckpt["epoch"]),
        "best_dev_accuracy": final_metrics["accuracy"],
        "best_dev_macro_precision": final_metrics["macro_precision"],
        "best_dev_macro_recall": final_metrics["macro_recall"],
        "best_dev_macro_f1": final_metrics["macro_f1"],
        "ema_dev_accuracy": best_payload.get("accuracy") if used_ema else None,
        "ema_dev_macro_precision": best_payload.get("macro_precision") if used_ema else None,
        "ema_dev_macro_recall": best_payload.get("macro_recall") if used_ema else None,
        "ema_dev_macro_f1": best_payload.get("macro_f1") if used_ema else None,
        "selected_used_ema": bool(used_ema),
        "top_k_hybrid": top_k_hybrid, "top_k_global": top_k_global,
        "start_epoch": start_epoch,
        "num_train_examples": len(bundle["train_dataset"]),
        "num_dev_examples": len(bundle["dev_dataset"]),
        "device": str(device),
        "primary_metric": best_metric,
        "warmup_checkpoint": str(warmup_ckpt).replace("\\", "/"),
        "best_checkpoint": str(chosen_ckpt).replace("\\", "/"),
    }
    save_json(payload, run_paths["metrics"] / "metrics.json")
    save_json({"per_class_metrics": final_metrics["per_class_metrics"]},
              run_paths["svda_finetune"] / "per_class_metrics.json")
    save_confusion_matrix_csv(
        final_metrics["confusion_matrix"], final_metrics["labels"],
        run_paths["svda_finetune"] / "confusion_matrix.csv",
    )

    summary_path = refresh_expJ_summary_csv(str(config["experiment"]["output_root"]))
    print(
        f"ExpJ fine-tune finished. Best dev macro_f1={payload['best_dev_macro_f1']:.6f} "
        f"acc={payload['best_dev_accuracy']:.6f} "
        f"macro_precision={payload['best_dev_macro_precision']:.6f} "
        f"(epoch={payload['best_epoch']}, ema={used_ema})"
    )
    print(f"Summary written to {summary_path}")
    print(f"Run directory: {run_paths['svda_run_dir']}")


if __name__ == "__main__":
    main()
