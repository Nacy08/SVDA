#!/usr/bin/env python3
"""Train a LoRA DPO adapter from Plan_DPO round2 preference pairs.

The round2 pair file contains model outputs that may echo generation
instructions such as "Do not repeat..." or "Response:".  This script keeps
the DPO prompt identical to test-time input and strips those echoed wrappers
from chosen/rejected responses before training.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path("/home/lanxin/NLPCC Task2")
PLAN_DPO_ROOT = ROOT / "Plan_DPO"

DEFAULT_BASE_MODEL = ROOT / "Qlora_Sft_Model" / "Meta-Llama-3.1-8B-Instruct"
DEFAULT_SFT_ADAPTER = ROOT / "DPO" / "dpo_adapter" / "checkpoint-1600"
DEFAULT_TRAIN_FILE = PLAN_DPO_ROOT / "dpo_pairs_round2_selected_6000_ckpt1600.jsonl"
DEFAULT_OUTPUT_DIR = PLAN_DPO_ROOT / "plan_dpo_adapter_round2_ckpt1600"

PROMPT_ALLOWED_FIELDS = ("Scenario", "Question", "Value")
ORIGINAL_PROMPT = """You are the person described in the scenario.
Answer the question from your own perspective.
Your response should be natural, meaningful, and aligned with the target human value.
Do not merely repeat the value name. Do not explain the value label. Output only the response.

Scenario: {Scenario}
Question: {Question}
Target value: {Value}

Response:"""

INSTRUCTION_LINE_RE = re.compile(
    r"^\s*(?:"
    r"do\s+not\b.*|"
    r"don't\b.*|"
    r"output\s+only\b.*|"
    r"remove\s+all\b.*|"
    r"exclude\s+header\b.*|"
    r"the\s+response\s+should\s+be\b.*|"
    r"the\s+response\s+should\s+align\b.*|"
    r"the\s+response\s+will\b.*|"
    r"the\s+output\s+should\s+be\b.*|"
    r"the\s+output\s+should\s+follow\b.*|"
    r"the\s+output\s+does\s+not\s+include\b.*|"
    r"the\s+output\s+does\s+not\s+contain\b.*|"
    r"the\s+output\s+format\b.*|"
    r"the\s+answer\s+should\s+be\b.*|"
    r"the\s+target\s+human\s+value\s+is\b.*|"
    r"write\s+one\s+natural\b.*|"
    r"answer\s+the\s+question\b.*|"
    r"target(?:\s+human)?\s+value\s*:.*|"
    r"scenario\s*:.*|"
    r"question\s*:.*|"
    r"plan\s*:.*"
    r")\s*$",
    re.IGNORECASE,
)

RESPONSE_LABEL_RE = re.compile(
    r"^\s*(?:assistant\s*:|answer\s*:|response\s*:|final\s*:)\s*",
    re.IGNORECASE,
)

ANY_RESPONSE_LABEL_RE = re.compile(
    r"(?:^|\n|\b)(?:assistant|answer|response|final)\s*:\s*",
    re.IGNORECASE,
)

LEADING_CONTROL_SENTENCE_RE = re.compile(
    r"^\s*(?:"
    r"do\s+not\b[^.!?\n]*[.!?]\s*|"
    r"don't\b[^.!?\n]*[.!?]\s*|"
    r"output\s+only\b[^.!?\n]*[.!?]\s*|"
    r"remove\s+all\b[^.!?\n]*[.!?]\s*|"
    r"exclude\s+header\b[^.!?\n]*[.!?]\s*|"
    r"the\s+response\s+should\s+be\b[^.!?\n]*[.!?]\s*|"
    r"the\s+response\s+should\s+align\b[^.!?\n]*[.!?]\s*|"
    r"the\s+response\s+will\b[^.!?\n]*[.!?]\s*|"
    r"the\s+output\s+should\s+be\b[^.!?\n]*[.!?]\s*|"
    r"the\s+output\s+should\s+follow\b[^.!?\n]*[.!?]\s*|"
    r"the\s+output\s+does\s+not\s+include\b[^.!?\n]*[.!?]\s*|"
    r"the\s+output\s+does\s+not\s+contain\b[^.!?\n]*[.!?]\s*|"
    r"the\s+output\s+format\b[^.!?\n]*[.!?]\s*|"
    r"the\s+answer\s+should\s+be\b[^.!?\n]*[.!?]\s*|"
    r"the\s+target\s+human\s+value\s+is\b[^.!?\n]*[.!?]\s*|"
    r"do\s+i\b[^.!?\n]*[.!?]\s*|"
    r"target(?:\s+human)?\s+value\s*:[^.!?\n]*[.!?]?\s*"
    r")",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Plan_DPO round2 LoRA-DPO training with clean inputs."
    )
    parser.add_argument("--base_model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--sft_adapter", default=str(DEFAULT_SFT_ADAPTER))
    parser.add_argument("--train_file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only load, clean, and report data statistics; do not train.",
    )
    parser.add_argument(
        "--write_clean_jsonl",
        default=None,
        help="Optional path to write cleaned {prompt, chosen, rejected} rows.",
    )

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


def squeeze_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def clean_response(text: Any) -> str:
    """Remove echoed prompt/control text while preserving the real answer."""
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""

    # Prefer the final labeled answer span, e.g. "metadata...\nResponse: real answer".
    label_matches = list(ANY_RESPONSE_LABEL_RE.finditer(raw))
    if label_matches:
        raw = raw[label_matches[-1].end() :].strip()

    lines = [line.strip() for line in raw.split("\n")]
    kept: list[str] = []
    for line in lines:
        if not line:
            if kept and kept[-1]:
                kept.append("")
            continue
        if INSTRUCTION_LINE_RE.match(line):
            peeled = line
            previous_line = None
            while peeled and peeled != previous_line:
                previous_line = peeled
                peeled = LEADING_CONTROL_SENTENCE_RE.sub("", peeled, count=1).strip()
            if not peeled or INSTRUCTION_LINE_RE.match(peeled):
                continue
            line = peeled
        kept.append(line)

    cleaned = "\n".join(kept).strip()

    # Some rows put the label at the beginning of the remaining text.
    cleaned = RESPONSE_LABEL_RE.sub("", cleaned, count=1).strip()

    # If there is no label, peel off leading control sentences while preserving
    # same-line answers such as "Do not repeat the label. Choose the team...".
    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned
        cleaned = LEADING_CONTROL_SENTENCE_RE.sub("", cleaned, count=1).strip()

    # Drop accidental fenced blocks without treating response prose as code.
    cleaned = re.sub(r"^```(?:\w+)?\s*", "", cleaned).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return squeeze_ws(cleaned)


def normalize_prompt_boundary(prompt: str) -> str:
    """Keep a stable tokenization boundary between prompt and completion."""
    prompt = prompt.replace("\r\n", "\n").replace("\r", "\n").rstrip()
    if not prompt:
        return ""
    return prompt + "\n"


def build_clean_prompt(item: dict[str, Any]) -> str:
    """Use only task fields, never plans, judgments, rankings, or metadata."""
    if all(str(item.get(field, "")).strip() for field in PROMPT_ALLOWED_FIELDS):
        prompt = ORIGINAL_PROMPT.format(
            Scenario=str(item.get("Scenario", "")).strip(),
            Question=str(item.get("Question", "")).strip(),
            Value=str(item.get("Value", "")).strip(),
        )
        return normalize_prompt_boundary(prompt)
    return normalize_prompt_boundary(str(item.get("prompt", "")))


def dirty_marker_count(text: str) -> int:
    lower = text.lower()
    markers = (
        "do not repeat",
        "do not include",
        "do not explain",
        "output only",
        "response:",
        "target value:",
        "exclude header metadata",
        "remove all other metadata",
    )
    return sum(1 for marker in markers if marker in lower)


def load_pairs(path: Path, limit: int | None = None) -> tuple[list[dict[str, str]], Counter[str]]:
    rows: list[dict[str, str]] = []
    stats: Counter[str] = Counter()
    pair_type_counter: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            stats["raw_rows"] += 1
            item = json.loads(line)
            prompt = build_clean_prompt(item)
            chosen_raw = str(item.get("chosen", ""))
            rejected_raw = str(item.get("rejected", ""))
            chosen = clean_response(chosen_raw)
            rejected = clean_response(rejected_raw)

            if chosen != chosen_raw.strip():
                stats["chosen_cleaned"] += 1
            if rejected != rejected_raw.strip():
                stats["rejected_cleaned"] += 1
            stats["raw_dirty_markers"] += dirty_marker_count(chosen_raw)
            stats["raw_dirty_markers"] += dirty_marker_count(rejected_raw)
            stats["clean_dirty_markers"] += dirty_marker_count(chosen)
            stats["clean_dirty_markers"] += dirty_marker_count(rejected)

            if not prompt or not chosen or not rejected:
                stats["skipped_empty"] += 1
                continue
            if chosen == rejected:
                stats["skipped_same_response"] += 1
                continue

            key = (prompt, chosen, rejected)
            if key in seen:
                stats["skipped_duplicate"] += 1
                continue
            seen.add(key)
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
            pair_type_counter[str(item.get("pair_type", "unknown"))] += 1

            if limit is not None and len(rows) >= limit:
                stats["stopped_at_limit"] = line_no
                break

    if not rows:
        raise ValueError(f"No valid DPO pairs found in {path}")

    stats["valid_rows"] = len(rows)
    print(f"Loaded valid DPO pairs: {len(rows)}")
    print(f"Skipped empty pairs: {stats['skipped_empty']}")
    print(f"Skipped same-response pairs: {stats['skipped_same_response']}")
    print(f"Skipped duplicate pairs: {stats['skipped_duplicate']}")
    print(f"Chosen responses cleaned: {stats['chosen_cleaned']}")
    print(f"Rejected responses cleaned: {stats['rejected_cleaned']}")
    print(f"Raw dirty marker hits: {stats['raw_dirty_markers']}")
    print(f"Clean dirty marker hits: {stats['clean_dirty_markers']}")
    print("Pair type distribution:")
    for pair_type, count in pair_type_counter.most_common():
        print(f"  {pair_type}: {count}")
    return rows, stats


def write_clean_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote cleaned training rows to {path}")


def make_bnb_config(args: argparse.Namespace):
    import torch
    from transformers import BitsAndBytesConfig

    if not args.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(adapter_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_policy_or_ref_model(
    base_model_path: str,
    adapter_path: str,
    quantization_config: Any,
    trust_remote_code: bool,
    is_trainable: bool,
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

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


def write_run_config(output_dir: Path, args: argparse.Namespace, stats: Counter[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = vars(args).copy()
    payload["data_stats"] = dict(stats)
    payload["prompt_allowed_fields"] = list(PROMPT_ALLOWED_FIELDS)
    payload["response_cleaning"] = {
        "strip_instruction_echo": True,
        "strip_response_label": True,
        "rebuild_prompt_from_scenario_question_value": True,
    }
    config_path = output_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def make_dpo_config(args: argparse.Namespace, output_dir: Path):
    from trl import DPOConfig

    return DPOConfig(
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    rows, stats = load_pairs(Path(args.train_file), limit=args.limit)
    if args.write_clean_jsonl:
        write_clean_jsonl(Path(args.write_clean_jsonl), rows)
    write_run_config(output_dir, args, stats)

    print("\nClean sample:")
    print(json.dumps(rows[0], ensure_ascii=False, indent=2))

    if args.dry_run:
        print("\nDry run complete. No model was loaded and no training was started.")
        return

    from datasets import Dataset
    from trl import DPOTrainer

    train_dataset = Dataset.from_list(rows)
    tokenizer = load_tokenizer(args.sft_adapter)
    bnb_config = make_bnb_config(args)

    print("Loading trainable policy model")
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

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=make_dpo_config(args, output_dir),
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Done. DPO adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
