import argparse
import math
from typing import Dict

import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import build_fulcra_dataloader, build_fulcra_datasets
from models import RobertaForFULCRARegression
from utils import (
    append_jsonl,
    copy_config_file,
    create_expB_run_directories,
    ensure_local_model_path,
    load_yaml,
    resolve_device,
    save_json,
    save_value_type_mappings,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FULCRA signed-vector warm-up for Experiment B.")
    parser.add_argument("--config", required=True, help="Path to the Experiment B YAML config file.")
    parser.add_argument("--seed", required=True, type=int, help="Random seed for a single run.")
    return parser.parse_args()


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
    }


def train_one_epoch(model, dataloader, optimizer, scheduler, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        tensors = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        outputs = model(**tensors)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


def evaluate_loss(model, dataloader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            tensors = move_batch_to_device(batch, device)
            outputs = model(**tensors)
            total_loss += outputs["loss"].item()
    return total_loss / max(len(dataloader), 1)


def save_best_checkpoint(checkpoint_path, model, epoch: int, val_loss: float, num_value_dims: int) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_val_loss": val_loss,
            "num_value_dims": num_value_dims,
        },
        checkpoint_path,
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    config["train"]["seed"] = args.seed

    set_seed(args.seed)
    ensure_local_model_path(config["model"]["pretrained_model_name_or_path"])
    run_paths = create_expB_run_directories(config["experiment"]["output_root"], args.seed)
    device = resolve_device(config["runtime"]["device"])

    copy_config_file(args.config, run_paths["artifacts"] / "config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        local_files_only=True,
    )
    dataset_bundle = build_fulcra_datasets(config, seed=args.seed)
    save_value_type_mappings(
        dataset_bundle["value_type2id"],
        dataset_bundle["id2value_type"],
        run_paths["artifacts"],
    )

    train_loader = build_fulcra_dataloader(
        dataset_bundle["train_dataset"],
        tokenizer=tokenizer,
        batch_size=int(config["train"]["batch_size"]),
        max_length=int(config["model"]["max_length"]),
        shuffle=True,
        num_workers=int(config["train"]["num_workers"]),
    )
    val_loader = build_fulcra_dataloader(
        dataset_bundle["val_dataset"],
        tokenizer=tokenizer,
        batch_size=int(config["train"]["batch_size"]),
        max_length=int(config["model"]["max_length"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
    )

    model = RobertaForFULCRARegression(
        pretrained_model_name_or_path=config["model"]["pretrained_model_name_or_path"],
        num_value_dims=int(dataset_bundle["num_value_dims"]),
        hidden_dim=int(config["fulcra_model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    total_training_steps = len(train_loader) * int(config["fulcra_train"]["num_epochs"])
    warmup_steps = math.ceil(total_training_steps * float(config["train"]["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    training_log_path = run_paths["logs"] / "fulcra_warmup_log.jsonl"
    best_val_loss = float("inf")
    best_epoch = -1
    best_checkpoint = run_paths["fulcra_checkpoints"] / "best.pt"

    for epoch in range(1, int(config["fulcra_train"]["num_epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = evaluate_loss(model, val_loader, device)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            save_best_checkpoint(best_checkpoint, model, epoch, val_loss, int(dataset_bundle["num_value_dims"]))

        append_jsonl(
            training_log_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "is_best": is_best,
            },
        )
        print(
            "Epoch {epoch}: val_loss={val_loss:.6f}".format(
                epoch=epoch,
                val_loss=val_loss,
            )
        )

    save_json(
        {
            "run_dir": str(run_paths["run_dir"]).replace("\\", "/"),
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "num_train_examples": len(dataset_bundle["train_dataset"]),
            "num_val_examples": len(dataset_bundle["val_dataset"]),
            "num_value_dims": int(dataset_bundle["num_value_dims"]),
            "device": str(device),
            "best_checkpoint": str(best_checkpoint).replace("\\", "/"),
        },
        run_paths["fulcra_warmup"] / "metrics.json",
    )
    save_json(
        {
            "seed": args.seed,
            "run_dir": str(run_paths["run_dir"]).replace("\\", "/"),
            "stage": "fulcra_warmup",
            "device": str(device),
            "num_value_dims": int(dataset_bundle["num_value_dims"]),
            "best_checkpoint": str(best_checkpoint).replace("\\", "/"),
            "train_config": config["train"],
            "fulcra_train_config": config["fulcra_train"],
            "model_config": config["model"],
            "fulcra_data_config": config["fulcra_data"],
        },
        run_paths["artifacts"] / "run_info.json",
    )
    print(f"FULCRA warm-up finished. Best val loss: {best_val_loss:.6f}")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Run directory: {run_paths['run_dir']}")


if __name__ == "__main__":
    main()
