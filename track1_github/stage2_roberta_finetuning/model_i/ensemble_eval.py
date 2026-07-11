"""Ensemble inference over N already-trained checkpoints.

For each checkpoint, forward over the dev split, compute softmax with optional
temperature scaling, average probabilities across models, argmax → predictions,
then evaluate via the canonical (immutable) `compute_classification_metrics`.

Each checkpoint just needs:
- the same encoder backbone (RoBERTa-large)
- the same head topology (1024 → hidden_dim → 19)
- weights saved under `model_state_dict`

Both ExpF / ExpF_new / ExpI models satisfy this — their encoder + classifier
state_dict keys are byte-identical.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from data import build_dataloader, build_svda_datasets
from metrics import compute_classification_metrics, verify_canonical_metrics
from models import RobertaForSVDAClassificationI
from utils import load_yaml, resolve_device, set_seed


@dataclass
class CheckpointSpec:
    path: str
    label: str = ""
    use_question: bool = False  # match the input convention this ckpt was trained with
    weight: float = 1.0  # ensemble weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Soft-vote ensemble over saved SVDA checkpoints.")
    parser.add_argument("--config", required=True, help="Base config to take model arch + data paths from.")
    parser.add_argument("--checkpoints_json", required=True,
                        help='JSON file with list of {"path":"...","label":"...","use_question":bool}')
    parser.add_argument("--temperatures", default="0.5,0.75,1.0,1.25,1.5,2.0,2.5",
                        help="Comma-separated temperature values to sweep.")
    parser.add_argument("--output_json", default=None,
                        help="Where to write the per-temperature metric sweep summary.")
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def load_model(spec: CheckpointSpec, base_config: Dict[str, object], device: torch.device) -> torch.nn.Module:
    """Instantiate a minimal classifier and load the checkpoint's encoder+classifier weights."""
    num_labels = int(base_config["model"]["num_labels"])
    model = RobertaForSVDAClassificationI(
        pretrained_model_name_or_path=str(base_config["model"]["pretrained_model_name_or_path"]),
        num_labels=num_labels,
        hidden_dim=int(base_config["model"]["hidden_dim"]),
        dropout=float(base_config["model"]["dropout"]),
        candidate_map={},  # not used at inference
        ranking_margin=1.0,
        top_k_hybrid=1,
        top_k_global=1,
    )
    ckpt = torch.load(spec.path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # Load only encoder.* and classifier.* keys; ignore loss-fn nested state if any.
    kept = {k: v for k, v in state_dict.items()
            if k.startswith("encoder.") or k.startswith("classifier.")}
    missing, unexpected = model.load_state_dict(kept, strict=False)
    # We expect non-encoder/classifier keys to be "missing" (loss buffers etc.).
    enc_cls_missing = [k for k in missing if k.startswith(("encoder.", "classifier."))]
    if enc_cls_missing:
        raise RuntimeError(f"Encoder/classifier weights missing for {spec.path}: {enc_cls_missing[:5]}")
    if unexpected:
        # Some checkpoints contain confusable_loss_fn etc. — those are not parameters in our model.
        pass
    model.to(device)
    model.eval()
    return model


def collect_probs(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    temperature: float,
) -> torch.Tensor:
    """Return (N, C) softmax(logits/T) tensor over dataloader order."""
    probs_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            probs = F.softmax(logits / float(temperature), dim=-1)
            probs_chunks.append(probs.cpu())
    return torch.cat(probs_chunks, dim=0)


def build_input_variants(base_config: Dict[str, object], use_question: bool):
    """Reuse build_svda_datasets twice if needed: with and without `use_question`."""
    cfg = json.loads(json.dumps(base_config))  # deep copy
    cfg["data"]["use_question"] = bool(use_question)
    return build_svda_datasets(cfg)


def main() -> None:
    verify_canonical_metrics()
    args = parse_args()
    base_config = load_yaml(args.config)
    set_seed(args.seed)
    device = resolve_device(str(base_config["runtime"]["device"]))
    temperatures = [float(t) for t in args.temperatures.split(",")]

    specs_raw = json.loads(Path(args.checkpoints_json).read_text())
    specs = [CheckpointSpec(**item) for item in specs_raw]

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_config["model"]["pretrained_model_name_or_path"]), local_files_only=True
    )

    # Pre-build dev dataloaders for each input variant we'll encounter.
    dev_loaders: Dict[bool, object] = {}
    id2label: Dict[int, str] = {}
    labels_seq: List[int] = []
    for variant in {spec.use_question for spec in specs}:
        bundle = build_input_variants(base_config, use_question=variant)
        loader = build_dataloader(
            bundle["dev_dataset"],
            tokenizer=tokenizer,
            batch_size=int(base_config["train"]["batch_size"]),
            max_length=int(base_config["model"]["max_length"]),
            shuffle=False,
            num_workers=0,
        )
        dev_loaders[variant] = loader
        if not id2label:
            id2label = bundle["id2label"]
            labels_seq = [bundle["dev_dataset"][i]["labels"] for i in range(len(bundle["dev_dataset"]))]

    # For each checkpoint, collect probs ONCE per temperature.
    # To avoid repeated forwards: collect logits once, then re-softmax per T.
    logits_per_model: List[torch.Tensor] = []
    for spec in specs:
        print(f"[load] {spec.label or spec.path}")
        model = load_model(spec, base_config, device)
        loader = dev_loaders[spec.use_question]
        logits_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
                logits_chunks.append(logits.cpu())
        logits_per_model.append(torch.cat(logits_chunks, dim=0))
        del model
        torch.cuda.empty_cache()

    # Sanity: ground truth labels (from any of the dataloaders, since they're the same dev split).
    gold_labels = labels_seq

    # Normalize weights.
    weights = torch.tensor([float(s.weight) for s in specs], dtype=torch.float32)
    if (weights <= 0).any():
        raise ValueError("All checkpoint weights must be positive.")
    weights = weights / weights.sum()

    # Sweep temperatures.
    sweep_rows: List[Dict[str, object]] = []
    best_overall = None
    for T in temperatures:
        # Weighted soft-vote: Σ w_i * softmax(z_i / T)
        probs_sum = None
        for w, logits in zip(weights.tolist(), logits_per_model):
            probs = F.softmax(logits / T, dim=-1) * float(w)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        probs_avg = probs_sum
        predictions = probs_avg.argmax(dim=-1).tolist()
        metrics = compute_classification_metrics(
            labels=gold_labels, predictions=predictions, id2label=id2label
        )
        row = {
            "temperature": T,
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
        }
        print(
            f"[T={T:>5.2f}]  acc={row['accuracy']:.6f}  "
            f"macro_p={row['macro_precision']:.6f}  "
            f"macro_r={row['macro_recall']:.6f}  "
            f"macro_f1={row['macro_f1']:.6f}"
        )
        sweep_rows.append(row)
        if (best_overall is None) or (row["macro_f1"] > best_overall["macro_f1"]):
            best_overall = row

    # Single-model baseline lines, for reference.
    single_rows: List[Dict[str, object]] = []
    for spec, logits in zip(specs, logits_per_model):
        probs = F.softmax(logits, dim=-1)
        predictions = probs.argmax(dim=-1).tolist()
        metrics = compute_classification_metrics(
            labels=gold_labels, predictions=predictions, id2label=id2label
        )
        single_rows.append({
            "label": spec.label or spec.path,
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
        })

    summary = {
        "checkpoints": [{"path": s.path, "label": s.label, "use_question": s.use_question}
                         for s in specs],
        "single_model_metrics": single_rows,
        "ensemble_sweep": sweep_rows,
        "best_ensemble": best_overall,
    }
    print("\n=== Single-model baselines (T=1.0) ===")
    for row in single_rows:
        print(f"  {row['label']:<70}  acc={row['accuracy']:.6f}  f1={row['macro_f1']:.6f}")
    print("\n=== Best ensemble ===")
    print(best_overall)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\nWrote sweep summary → {args.output_json}")


if __name__ == "__main__":
    main()
