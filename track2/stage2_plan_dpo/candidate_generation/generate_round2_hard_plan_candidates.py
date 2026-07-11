#!/usr/bin/env python3
"""Generate round2 hard-value plan-guided candidates for DPO data construction."""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import time
from pathlib import Path
from typing import Any


ROOT = Path("/home/lanxin/NLPCC Task2")
PLAN_DPO_ROOT = ROOT / "Plan_DPO"
DEFAULT_INPUT_FILE = PLAN_DPO_ROOT / "Plan" / "train_plan_gpt5mini.jsonl"
DEFAULT_OUTPUT_FILE = PLAN_DPO_ROOT / "hard_plan_g16_predictions" / "predictions.jsonl"
DEFAULT_BASE_MODEL = ROOT / "Qlora_Sft_Model" / "Meta-Llama-3.1-8B-Instruct"
DEFAULT_ADAPTER = ROOT / "DPO" / "dpo_adapter"

HIGH_VALUES = [
    "Face",
    "Power–resources",
    "Power–dominance",
    "Hedonism",
    "Humility",
    "Benevolence–dependability",
    "Security–personal",
]
MID_VALUES = [
    "Benevolence–caring",
    "Stimulation",
    "Conformity–interpersonal",
    "Tradition",
]
LOW_VALUES = [
    "Conformity–rules",
    "Universalism–concern",
    "Achievement",
]
HARD_VALUES = HIGH_VALUES + MID_VALUES + LOW_VALUES
VALUE_TO_TIER = {value: "high" for value in HIGH_VALUES}
VALUE_TO_TIER.update({value: "mid" for value in MID_VALUES})
VALUE_TO_TIER.update({value: "low" for value in LOW_VALUES})

ORIGINAL_PROMPT = """You are the person described in the scenario.
Answer the question from your own perspective.
Your response should be natural, meaningful, and aligned with the target human value.
Do not merely repeat the value name. Do not explain the value label. Output only the response.

Scenario: {Scenario}
Question: {Question}
Target value: {Value}

Response:"""

PLAN_GUIDED_PROMPT = """Scenario: {Scenario}
Question: {Question}
Target Value: {Value}

Plan:
{plan}

Write one natural, meaningful response that directly answers the question and strongly reflects the target value.
Output only the response."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate plan-guided G=16 candidates for hard-value round2 DPO."
    )
    parser.add_argument("--input_file", default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--output_file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--model_name_or_path", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--adapter_path", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--num_candidates", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(path: Path, start: int = 0, limit: int | None = None) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            if row_index < start or not line.strip():
                continue
            row = json.loads(line)
            if row.get("Value") not in VALUE_TO_TIER:
                continue
            rows.append((row_index, row))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_completed_row_indices(output_file: Path) -> set[int]:
    completed: set[int] = set()
    if not output_file.exists():
        return completed
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                completed.add(int(row.get("row_index")))
            except (TypeError, ValueError):
                continue
    return completed


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_run_config(path: Path, args: argparse.Namespace, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = vars(args).copy()
    payload.update(extra)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def plan_to_text(plan: Any) -> str:
    if plan is None:
        return ""
    if isinstance(plan, str):
        return plan.strip()
    if isinstance(plan, (dict, list)):
        return json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True).strip()
    return str(plan).strip()


def is_bad_candidate(text: str, value: str) -> bool:
    if not text.strip():
        return True
    lower = text.lower()
    value_lower = value.lower()
    if f"this reflects {value_lower}" in lower:
        return True
    if "target value" in lower or "value label" in lower:
        return True
    if value_lower in lower and any(
        marker in lower
        for marker in (
            "reflects",
            "aligns with",
            "is about",
            "means",
            "demonstrates",
            "represents",
        )
    ):
        return True
    return False


def dedupe_and_filter_candidates(candidates: list[dict[str, Any]], value: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    clean: list[dict[str, Any]] = []
    for candidate in candidates:
        text = str(candidate.get("text", "")).strip()
        key = re.sub(r"\s+", " ", text)
        if not key or key in seen or is_bad_candidate(text, value):
            continue
        seen.add(key)
        clean.append({"sample_id": len(clean) + 1, "text": text})
    return clean


def build_original_prompt(row: dict[str, Any]) -> str:
    return ORIGINAL_PROMPT.format(
        Scenario=row.get("Scenario", ""),
        Question=row.get("Question", ""),
        Value=row.get("Value", ""),
    )


def build_plan_prompt(row: dict[str, Any], plan_text: str) -> str:
    return PLAN_GUIDED_PROMPT.format(
        Scenario=row.get("Scenario", ""),
        Question=row.get("Question", ""),
        Value=row.get("Value", ""),
        plan=plan_text,
    )


def make_bnb_config(load_in_4bit: bool) -> BitsAndBytesConfig | None:
    import torch
    from transformers import BitsAndBytesConfig

    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=make_bnb_config(args.load_in_4bit),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()
    return model, tokenizer


def generate_candidates(
    model,
    tokenizer,
    prompt: str,
    num_candidates: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    input_length = inputs["input_ids"].shape[-1]
    input_ids = inputs["input_ids"].repeat(num_candidates, 1)
    attention_mask = inputs["attention_mask"].repeat(num_candidates, 1)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    candidates = []
    for sample_idx, sample_ids in enumerate(output_ids, start=1):
        generated_ids = sample_ids[input_length:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        candidates.append({"sample_id": sample_idx, "text": text})
    return candidates


def unload_model(model) -> None:
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    skipped_file = output_file.parent / "skipped_rows.jsonl"
    run_config_file = output_file.parent / "run_config.json"

    if not args.resume:
        output_file.write_text("", encoding="utf-8")
        skipped_file.write_text("", encoding="utf-8")

    completed = load_completed_row_indices(output_file) if args.resume else set()
    rows = load_rows(input_file, start=args.start, limit=args.limit)
    write_run_config(
        run_config_file,
        args,
        {
            "hard_values": HARD_VALUES,
            "generation_mode": "hard_plan_g16",
            "loaded_hard_rows": len(rows),
            "resume_completed": len(completed),
        },
    )

    print(f"Loaded hard-value rows: {len(rows)} from {input_file}", flush=True)
    print(f"Writing predictions to {output_file}", flush=True)
    print(f"Loading adapter: {args.adapter_path}", flush=True)
    model, tokenizer = load_model_and_tokenizer(args)
    adapter_name = Path(args.adapter_path).name
    started_at = time.time()
    written = 0
    skipped = 0

    try:
        for local_idx, (row_index, row) in enumerate(rows, start=1):
            if row_index in completed:
                continue
            plan_text = plan_to_text(row.get("plan"))
            if not plan_text:
                append_jsonl(
                    skipped_file,
                    {"row_index": row_index, "reason": "missing_or_empty_plan", **row},
                )
                skipped += 1
                continue
            plan_prompt = build_plan_prompt(row, plan_text)
            raw_candidates = generate_candidates(
                model=model,
                tokenizer=tokenizer,
                prompt=plan_prompt,
                num_candidates=args.num_candidates,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
            )
            candidates = dedupe_and_filter_candidates(raw_candidates, row.get("Value", ""))
            if not candidates:
                append_jsonl(
                    skipped_file,
                    {"row_index": row_index, "reason": "no_valid_candidates", **row},
                )
                skipped += 1
                continue
            output = {
                "Scenario": row.get("Scenario", ""),
                "Question": row.get("Question", ""),
                "Value": row.get("Value", ""),
                "Consistent Value Response": row.get("Consistent Value Response", ""),
                "Contrastive Response": row.get("Contrastive Response", ""),
                "prompt": build_original_prompt(row),
                "plan": row.get("plan"),
                "value_tier": VALUE_TO_TIER[row.get("Value")],
                "generation_mode": "hard_plan_g16",
                "adapter_name": adapter_name,
                "adapter_path": args.adapter_path,
                "base_model_path": args.model_name_or_path,
                "row_index": row_index,
                "num_requested_candidates": args.num_candidates,
                "num_valid_candidates": len(candidates),
                "candidates": candidates,
            }
            append_jsonl(output_file, output)
            written += 1
            if local_idx % 25 == 0 or local_idx == len(rows):
                print(f"{local_idx}/{len(rows)} written={written} skipped={skipped}", flush=True)
    finally:
        unload_model(model)

    elapsed = time.time() - started_at
    write_run_config(
        run_config_file,
        args,
        {
            "hard_values": HARD_VALUES,
            "generation_mode": "hard_plan_g16",
            "loaded_hard_rows": len(rows),
            "resume_completed": len(completed),
            "written_rows_this_run": written,
            "skipped_rows_this_run": skipped,
            "elapsed_seconds": round(elapsed, 3),
        },
    )
    print(f"Done. written={written}, skipped={skipped}, elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
