#!/usr/bin/env python3
"""Run final inference with the Plan_DPO round2 adapter.

This script mirrors the existing DPO/Plan_DPO generation style, but writes a
single final prediction field by default for validation/testing.  It can also
sample multiple candidates per row when --num_candidates > 1.
"""

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

DEFAULT_BASE_MODEL = ROOT / "Qlora_Sft_Model" / "Meta-Llama-3.1-8B-Instruct"
DEFAULT_ADAPTER = PLAN_DPO_ROOT / "plan_dpo_adapter_round2_ckpt1600"
DEFAULT_INPUT_FILE = ROOT / "Data" / "dev.jsonl"
DEFAULT_OUTPUT_FILE = PLAN_DPO_ROOT / "final_dev_predictions_round2_ckpt1600" / "final_predictions.jsonl"

PROMPT = """You are the person described in the scenario.
Answer the question from your own perspective.
Your response should be natural, meaningful, and aligned with the target human value.
Do not merely repeat the value name. Do not explain the value label. Output only the response.

Scenario: {Scenario}
Question: {Question}
Target value: {Value}

Response:
"""

RESPONSE_LABEL_RE = re.compile(
    r"^\s*(?:assistant\s*:|answer\s*:|response\s*:|final\s*:)\s*",
    re.IGNORECASE,
)
ANY_RESPONSE_LABEL_RE = re.compile(
    r"(?:^|\n|\b)(?:assistant|answer|response|final)\s*:\s*",
    re.IGNORECASE,
)
LEADING_CONTROL_RE = re.compile(
    r"^\s*(?:"
    r"do\s+not\b[^.!?\n]*[.!?]\s*|"
    r"don't\b[^.!?\n]*[.!?]\s*|"
    r"output\s+only\b[^.!?\n]*[.!?]\s*|"
    r"remove\s+all\b[^.!?\n]*[.!?]\s*|"
    r"the\s+response\s+should\b[^.!?\n]*[.!?]\s*|"
    r"the\s+output\s+should\b[^.!?\n]*[.!?]\s*|"
    r"target(?:\s+human)?\s+value\s*:[^.!?\n]*[.!?]?\s*"
    r")",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final validation/test predictions from the Plan_DPO adapter."
    )
    parser.add_argument("--base_model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--adapter_path", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--input_file", default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--output_file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--num_candidates", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use sampling. Default false for stable final validation output.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 4-bit NF4 quantization. Disable for CPU inference.",
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help='Transformers device_map, e.g. "auto" or "cpu".',
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_rows(path: Path, start: int = 0, limit: int | None = None) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            if row_index < start or not line.strip():
                continue
            rows.append((row_index, json.loads(line)))
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


def build_prompt(row: dict[str, Any]) -> str:
    return PROMPT.format(
        Scenario=str(row.get("Scenario", "")).strip(),
        Question=str(row.get("Question", "")).strip(),
        Value=str(row.get("Value", "")).strip(),
    )


def clean_generated_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    label_matches = list(ANY_RESPONSE_LABEL_RE.finditer(cleaned))
    if label_matches:
        cleaned = cleaned[label_matches[-1].end() :].strip()
    cleaned = RESPONSE_LABEL_RE.sub("", cleaned, count=1).strip()
    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned
        cleaned = LEADING_CONTROL_RE.sub("", cleaned, count=1).strip()
    cleaned = re.sub(r"^```(?:\w+)?\s*", "", cleaned).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    cleaned = re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", cleaned)).strip()
    return cleaned


def make_bnb_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None
    import torch
    from transformers import BitsAndBytesConfig

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

    adapter_path = Path(args.adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if args.device_map != "cpu" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=make_bnb_config(args.load_in_4bit),
        device_map=args.device_map,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model, tokenizer


def model_device(model):
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def generate_candidates(
    model,
    tokenizer,
    prompt: str,
    num_candidates: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    device = model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_length = inputs["input_ids"].shape[-1]
    input_ids = inputs["input_ids"].repeat(num_candidates, 1)
    attention_mask = inputs["attention_mask"].repeat(num_candidates, 1)
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})

    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)

    candidates = []
    seen: set[str] = set()
    for sample_idx, sample_ids in enumerate(output_ids, start=1):
        generated_ids = sample_ids[input_length:]
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        text = clean_generated_text(raw_text)
        if text in seen:
            continue
        seen.add(text)
        candidates.append({"sample_id": sample_idx, "text": text, "raw_text": raw_text.strip()})
    return candidates


def unload_model(model) -> None:
    try:
        import torch

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    run_config_file = output_file.parent / "run_config.json"

    if not args.resume:
        output_file.write_text("", encoding="utf-8")
    completed = load_completed_row_indices(output_file) if args.resume else set()
    rows = read_rows(input_file, start=args.start, limit=args.limit)
    adapter_path = Path(args.adapter_path)
    adapter_name = adapter_path.name

    write_run_config(
        run_config_file,
        args,
        {
            "mode": "final_inference",
            "loaded_rows": len(rows),
            "resume_completed": len(completed),
        },
    )

    print(f"Loaded rows: {len(rows)} from {input_file}", flush=True)
    print(f"Loading adapter: {adapter_path}", flush=True)
    print(f"Writing predictions to {output_file}", flush=True)
    model, tokenizer = load_model_and_tokenizer(args)

    started_at = time.time()
    written = 0
    skipped = 0
    try:
        for local_idx, (row_index, row) in enumerate(rows, start=1):
            if row_index in completed:
                skipped += 1
                continue
            prompt = build_prompt(row)
            candidates = generate_candidates(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                num_candidates=args.num_candidates,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
            )
            prediction = candidates[0]["text"] if candidates else ""
            output = {
                "row_index": row_index,
                "Scenario": row.get("Scenario", ""),
                "Question": row.get("Question", ""),
                "Value": row.get("Value", ""),
                "Consistent Value Response": row.get("Consistent Value Response", ""),
                "Contrastive Response": row.get("Contrastive Response", ""),
                "prompt": prompt,
                "prediction": prediction,
                "candidates": candidates,
                "adapter_name": adapter_name,
                "adapter_path": str(adapter_path),
                "base_model_path": args.base_model,
                "num_requested_candidates": args.num_candidates,
                "num_valid_candidates": len(candidates),
                "generation_mode": "plan_dpo_round2_final",
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
            "mode": "final_inference",
            "loaded_rows": len(rows),
            "resume_completed": len(completed),
            "written_rows_this_run": written,
            "skipped_rows_this_run": skipped,
            "elapsed_seconds": round(elapsed, 3),
        },
    )
    print(f"Done. written={written}, skipped={skipped}, elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
