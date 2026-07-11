#!/usr/bin/env python3
"""Train a LoRA DPO adapter from GPT-5 reranked preference pairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


DEFAULT_BASE_MODEL = (
    "/home/lanxin/NLPCC Task2/Qlora_Sft_Model/Meta-Llama-3.1-8B-Instruct"
)
DEFAULT_SFT_ADAPTER = (
    "/home/lanxin/NLPCC Task2/Qlora_Sft_Model/nlpcc_task2_checkpoint-1100"
)
DEFAULT_TRAIN_FILE = (
    "/home/lanxin/NLPCC Task2/Rerank/gpt5_rerank_outputs/dpo_pairs.jsonl"
)
DEFAULT_OUTPUT_DIR = "/home/lanxin/NLPCC Task2/DPO/dpo_adapter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline LoRA-DPO training with TRL DPOTrainer."
    )
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--sft_adapter", default=DEFAULT_SFT_ADAPTER)
    parser.add_argument("--train_file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def load_pairs(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    rows = []
    pair_type_counter: Counter[str] = Counter()
    skipped = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = str(item.get("prompt", "")).strip()
            chosen = str(item.get("chosen", "")).strip()
            rejected = str(item.get("rejected", "")).strip()
            if not prompt or not chosen or not rejected or chosen == rejected:
                skipped += 1
                continue
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
            pair_type_counter[str(item.get("pair_type", "unknown"))] += 1
            if limit is not None and len(rows) >= limit:
                break

    if not rows:
        raise ValueError(f"No valid DPO pairs found in {path}")

    print(f"Loaded valid DPO pairs: {len(rows)}")
    print(f"Skipped invalid pairs: {skipped}")
    print("Pair type distribution:")
    for pair_type, count in pair_type_counter.most_common():
        print(f"  {pair_type}: {count}")
    return rows


def make_bnb_config(args: argparse.Namespace) -> BitsAndBytesConfig | None:
    if not args.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_policy_or_ref_model(
    base_model_path: str,
    adapter_path: str,
    quantization_config: BitsAndBytesConfig | None,
    trust_remote_code: bool,
    is_trainable: bool,
):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=is_trainable)
    model.config.use_cache = False
    if is_trainable:
        model.train()
        model.print_trainable_parameters()
    else:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    return model


def write_run_config(output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    write_run_config(output_dir, args)
    print(
        "Note: installed trl 1.3.0 DPOConfig does not expose max_prompt_length; "
        f"recording requested max_prompt_length={args.max_prompt_length} only."
    )

    rows = load_pairs(Path(args.train_file), limit=args.limit)
    train_dataset = Dataset.from_list(rows)

    tokenizer = load_tokenizer(args.sft_adapter)
    bnb_config = make_bnb_config(args)

    print("Loading policy model")
    policy_model = load_policy_or_ref_model(
        base_model_path=args.base_model,
        adapter_path=args.sft_adapter,
        quantization_config=bnb_config,
        trust_remote_code=args.trust_remote_code,
        is_trainable=True,
    )

    print("Loading frozen reference model")
    ref_model = load_policy_or_ref_model(
        base_model_path=args.base_model,
        adapter_path=args.sft_adapter,
        quantization_config=bnb_config,
        trust_remote_code=args.trust_remote_code,
        is_trainable=False,
    )

    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        use_cache=False,
    )

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Done. DPO adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
