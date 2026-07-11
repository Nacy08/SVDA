#!/usr/bin/env python3
"""Generate dev-set candidates from the trained DPO LoRA adapter."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_BASE_MODEL = (
    "/home/lanxin/NLPCC Task2/Qlora_Sft_Model/Meta-Llama-3.1-8B-Instruct"
)
DEFAULT_ADAPTER = "/home/lanxin/NLPCC Task2/DPO/dpo_adapter"
DEFAULT_INPUT_FILE = "/home/lanxin/NLPCC Task2/Data/dev.jsonl"
DEFAULT_OUTPUT_FILE = (
    "/home/lanxin/NLPCC Task2/DPO/dev_candidates_dpo_adapter/predictions.jsonl"
)

PROMPT = """You are the person described in the scenario.
Answer the question from your own perspective.
Your response should be natural, meaningful, and aligned with the target human value.
Do not merely repeat the value name. Do not explain the value label. Output only the response.

Scenario: {Scenario}
Question: {Question}
Target value: {Value}

Response:"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate five dev-set candidates from the trained DPO adapter for "
            "downstream GPT-5.1 reranking."
        )
    )
    parser.add_argument(
        "--base_model",
        default=DEFAULT_BASE_MODEL,
        help="Path to the base causal LM.",
    )
    parser.add_argument(
        "--adapter_path",
        default=DEFAULT_ADAPTER,
        help="Path to the trained DPO LoRA adapter.",
    )
    parser.add_argument(
        "--input_file",
        default=DEFAULT_INPUT_FILE,
        help="Validation/dev JSONL file.",
    )
    parser.add_argument(
        "--output_file",
        default=DEFAULT_OUTPUT_FILE,
        help="Output predictions.jsonl path for downstream reranking.",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=5,
        help="Number of sampled candidates per input row.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p value.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
        help="Generation repetition penalty.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=160,
        help="Maximum number of new tokens per candidate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the base model with 4-bit NF4 quantization.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the base model.",
    )
    return parser.parse_args()


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_prompt(example: dict[str, Any]) -> str:
    return PROMPT.format(
        Scenario=example.get("Scenario", ""),
        Question=example.get("Question", ""),
        Value=example.get("Value", ""),
    )


def make_bnb_config(load_in_4bit: bool) -> BitsAndBytesConfig | None:
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(
    base_model: str,
    adapter_path: str,
    load_in_4bit: bool,
    trust_remote_code: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=make_bnb_config(load_in_4bit),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
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


def write_run_config(path: Path, config: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")


def unload_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_file, limit=args.limit)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    run_config_path = output_file.parent / "run_config.json"

    adapter_path = Path(args.adapter_path)
    adapter_name = adapter_path.name
    started_at = time.time()

    print(f"Loaded {len(rows)} rows from {args.input_file}", flush=True)
    print(f"Loading DPO adapter: {adapter_path}", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Writing candidates to {output_file}", flush=True)
    with output_file.open("w", encoding="utf-8") as out_f:
        for idx, row in enumerate(rows, start=1):
            prompt = build_prompt(row)
            candidates = generate_candidates(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                num_candidates=args.num_candidates,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
            )
            output = dict(row)
            output.update(
                {
                    "prompt": prompt,
                    "candidates": candidates,
                    "adapter_name": adapter_name,
                    "adapter_path": str(adapter_path),
                    "base_model_path": args.base_model,
                    "num_samples": args.num_candidates,
                }
            )
            out_f.write(json.dumps(output, ensure_ascii=False) + "\n")

            if idx % 25 == 0 or idx == len(rows):
                print(f"{idx}/{len(rows)}", flush=True)

    elapsed_seconds = round(time.time() - started_at, 3)
    write_run_config(
        run_config_path,
        {
            "adapter_name": adapter_name,
            "adapter_path": str(adapter_path),
            "base_model_path": args.base_model,
            "input_file": args.input_file,
            "output_file": str(output_file),
            "sample_count": len(rows),
            "num_candidates_per_example": args.num_candidates,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "load_in_4bit": args.load_in_4bit,
            "bnb_4bit_quant_type": "nf4" if args.load_in_4bit else None,
            "bnb_4bit_compute_dtype": "bfloat16" if args.load_in_4bit else None,
            "bnb_4bit_use_double_quant": True if args.load_in_4bit else None,
            "downstream_rerank_model": "gpt-5.1",
            "elapsed_seconds": elapsed_seconds,
        },
    )
    unload_model(model)
    print(f"Done in {elapsed_seconds}s", flush=True)
    print(
        "Rerank with: python \"/home/lanxin/NLPCC Task2/Rerank/qwen_rerank.py\" "
        f"--input_file \"{output_file}\" --output_dir "
        "\"/home/lanxin/NLPCC Task2/Rerank/gpt51_dev_rerank_outputs\" "
        "--model \"gpt-5.1\" --resume",
        flush=True,
    )


if __name__ == "__main__":
    main()
