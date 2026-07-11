#!/usr/bin/env python3
"""Rerank DPO dev candidates with GPT-5.1 and export final predictions.

Input is the predictions.jsonl produced by the DPO candidate generation script.
Each line should contain Scenario, Question, Value, candidates, and optionally
the official consistent/contrastive responses. The script writes:
  1. rerank_results.jsonl: GPT-5.1 judgment details for each source row.
  2. final_predictions.jsonl: one selected prediction per source row for eval.
  3. failed_samples.jsonl: rows that failed after retries.
  4. run_config.json: arguments used for this run.

The default API settings follow CloseAI's OpenAI-compatible documentation:
https://doc.closeai-asia.com/tutorial/api/openai.html
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local
from typing import Any

DEFAULT_BASE_URL = "https://api.openai-proxy.org/v1"
DEFAULT_MODEL = "gpt-5.1"
DEFAULT_INPUT = "/home/lanxin/NLPCC Task2/DPO/dev_candidates_dpo_adapter/predictions.jsonl"
DEFAULT_OUTPUT_DIR = "/home/lanxin/NLPCC Task2/Rerank/gpt51_dpo_rerank_outputs"
DEFAULT_TIMEOUT = 300.0
THREAD_LOCAL = local()


SYSTEM_PROMPT = """You are a strict preference judge for a human-value response generation task.
You will receive a Scenario, a Question, a Target Value, a reference value-consistent response, and several sampled response candidates.

Official criterion:
1. A candidate must meaningfully answer the Question.
2. Among meaningful candidates, prefer the one that is more naturally and strongly consistent with the Target Value.
3. Prefer concrete, scenario-specific, human-like answers over generic value slogans.

Judging rules:
- Choose only from the sampled response candidates.
- Do not choose a candidate merely because it is longer.
- Do not reward direct explanations such as "This reflects Achievement" or repeated value labels.
- Penalize candidates that are off-topic, unnatural, empty, evasive, or inconsistent with the target value.
- Use the reference value-consistent response only as guidance for the intended value alignment; do not select it because it is not a candidate.

Return valid JSON only. Do not include markdown fences."""


USER_PROMPT_TEMPLATE = """Scenario:
{scenario}

Question:
{question}

Target Value:
{value}

Reference value-consistent response:
{gold_response}

Sampled response candidates:
{candidate_block}

Select the single best sampled response candidate for the next evaluation stage.

Return this JSON schema exactly:
{{
  "best_candidate": "sample_3",
  "confidence": "high",
  "reason": "sample_3 directly answers the question and is the most natural, specific, and strongly aligned with the target value."
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank DPO-generated dev candidates with GPT-5.1."
    )
    parser.add_argument(
        "--input_file",
        default=DEFAULT_INPUT,
        help="DPO candidate predictions.jsonl.",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for rerank_results.jsonl and final_predictions.jsonl.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help="OpenAI judge model. Default: gpt-5.1 or OPENAI_MODEL.",
    )
    parser.add_argument(
        "--base_url",
        default=os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or DEFAULT_BASE_URL,
        help=(
            "OpenAI-compatible API base URL. CloseAI default: "
            "https://api.openai-proxy.org/v1. Must include /v1."
        ),
    )
    parser.add_argument(
        "--api_key_env",
        default="OPENAI_API_KEY",
        help="Environment variable that stores the API key.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge temperature. Keep 0 for reproducible labels.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Concurrent API worker threads. Use 1 for smoke tests.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum API retries per sample.",
    )
    parser.add_argument(
        "--retry_sleep",
        type=float,
        default=3.0,
        help="Base seconds to sleep before retry. Exponential backoff is used.",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=256,
        help="Maximum tokens for the judge JSON response.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=(
            "Request timeout in seconds. CloseAI notes ChatCompletion requests "
            "can run up to about 5 minutes, so the default is 300."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to process for smoke tests.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based start row offset.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip examples already present in rerank_results.jsonl or failed_samples.jsonl.",
    )
    return parser.parse_args()


def load_jsonl_with_indices(
    path: Path, limit: int | None = None, start: int = 0
) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            if row_index < start or not line.strip():
                continue
            rows.append((row_index, json.loads(line)))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_completed_keys(*paths: Path) -> set[str]:
    completed = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row.get("source_key")
                if key:
                    completed.add(str(key))
    return completed


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if not base_url.endswith("/v1"):
        raise ValueError(
            f"Invalid base_url={base_url!r}. CloseAI/OpenAI-compatible APIs need a /v1 suffix."
        )
    return base_url


def preflight_api_config(args: argparse.Namespace) -> None:
    if not os.getenv(args.api_key_env):
        raise SystemExit(
            f"Missing API key. Please set {args.api_key_env}, for example:\n"
            f"  export {args.api_key_env}=sk-..."
        )
    try:
        import openai  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "openai is not installed. Install it with: pip install openai"
        ) from exc
    try:
        normalize_base_url(args.base_url)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def make_client(args: argparse.Namespace):
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Please set {args.api_key_env}, for example:\n"
            f"  export {args.api_key_env}=sk-..."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai is not installed. Install it with: pip install openai"
        ) from exc

    return OpenAI(
        base_url=normalize_base_url(args.base_url),
        api_key=api_key,
        timeout=args.timeout,
    )


def get_thread_client(args: argparse.Namespace):
    client = getattr(THREAD_LOCAL, "client", None)
    if client is None:
        client = make_client(args)
        THREAD_LOCAL.client = client
    return client


def hash_text(text: str) -> str:
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return f"{value:08x}"


def source_key(row: dict[str, Any], row_index: int) -> str:
    adapter = row.get("adapter_name", "unknown_adapter")
    scenario = row.get("Scenario", "")
    question = row.get("Question", "")
    value = row.get("Value", "")
    return f"{adapter}:{row_index}:{hash_text(scenario + question + value)}"


def candidate_text(candidate: Any) -> str:
    if isinstance(candidate, dict):
        return str(candidate.get("text", "")).strip()
    return str(candidate).strip()


def candidate_id(candidate: Any, fallback_index: int) -> str:
    if isinstance(candidate, dict):
        raw_id = candidate.get("sample_id") or candidate.get("candidate_id") or fallback_index
    else:
        raw_id = fallback_index
    text_id = str(raw_id).strip()
    if text_id.startswith("sample_"):
        return text_id
    return f"sample_{text_id}"


def build_candidate_items(row: dict[str, Any]) -> dict[str, str]:
    items: dict[str, str] = {}
    for idx, candidate in enumerate(row.get("candidates", []), start=1):
        cid = candidate_id(candidate, idx)
        text = candidate_text(candidate)
        if text and cid not in items:
            items[cid] = text
    return items


def build_candidate_block(candidate_items: dict[str, str]) -> str:
    lines = []
    for response_id, text in candidate_items.items():
        lines.append(f"[{response_id}] {text}")
    return "\n\n".join(lines)


def build_user_prompt(row: dict[str, Any], candidate_items: dict[str, str]) -> str:
    return USER_PROMPT_TEMPLATE.format(
        scenario=row.get("Scenario", ""),
        question=row.get("Question", ""),
        value=row.get("Value", ""),
        gold_response=row.get("Consistent Value Response", ""),
        candidate_block=build_candidate_block(candidate_items),
    )


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            return json.loads(match.group(0))
        raise


def validate_judgment(
    judgment: dict[str, Any], expected_candidate_ids: set[str]
) -> dict[str, Any]:
    if not isinstance(judgment, dict):
        raise ValueError("judgment is not a JSON object")

    best_candidate = str(judgment.get("best_candidate", "")).strip()
    if best_candidate not in expected_candidate_ids:
        raise ValueError(
            "best_candidate must be one of the sampled candidate ids; "
            f"got={best_candidate!r}, expected={sorted(expected_candidate_ids)}"
        )

    confidence = str(judgment.get("confidence", "")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        raise ValueError("confidence must be high, medium, or low")

    return {
        "best_candidate": best_candidate,
        "confidence": confidence,
        "reason": str(judgment.get("reason", "")).strip(),
    }


def create_chat_completion(client, request: dict[str, Any]):
    return client.chat.completions.create(**request)


def call_judge(
    client,
    args: argparse.Namespace,
    user_prompt: str,
    expected_candidate_ids: set[str],
) -> dict[str, Any]:
    last_error: Exception | None = None
    use_response_format = True
    use_temperature = True
    use_max_completion_tokens = True

    for attempt in range(args.max_retries):
        try:
            request: dict[str, Any] = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if use_temperature:
                request["temperature"] = args.temperature
            if use_response_format:
                request["response_format"] = {"type": "json_object"}
            if use_max_completion_tokens:
                request["max_completion_tokens"] = args.max_completion_tokens
            else:
                request["max_tokens"] = args.max_completion_tokens

            response = create_chat_completion(client, request)
            content = response.choices[0].message.content
            return validate_judgment(extract_json(content), expected_candidate_ids)
        except Exception as exc:  # API errors vary across SDK/provider versions.
            last_error = exc
            error_text = str(exc).lower()
            if use_response_format and (
                "response_format" in error_text or "json_object" in error_text
            ):
                use_response_format = False
                print(
                    "API does not appear to support response_format; retrying without it.",
                    file=sys.stderr,
                )
                continue
            if use_temperature and "temperature" in error_text and "unsupported" in error_text:
                use_temperature = False
                print("API rejected temperature; retrying without it.", file=sys.stderr)
                continue
            if use_max_completion_tokens and "max_completion_tokens" in error_text:
                use_max_completion_tokens = False
                print(
                    "API rejected max_completion_tokens; retrying with max_tokens.",
                    file=sys.stderr,
                )
                continue

            sleep_seconds = args.retry_sleep * (2**attempt) + random.random()
            print(f"API call failed ({attempt + 1}/{args.max_retries}): {exc}", file=sys.stderr)
            time.sleep(sleep_seconds)

    raise RuntimeError(f"GPT-5.1 rerank failed after retries: {last_error}")


def final_prediction_row(
    source_row: dict[str, Any],
    best_candidate: str,
    final_prediction: str,
    rerank_reason: str,
    judge_model: str,
) -> dict[str, Any]:
    return {
        "Scenario": source_row.get("Scenario", ""),
        "Question": source_row.get("Question", ""),
        "Value": source_row.get("Value", ""),
        "Consistent Value Response": source_row.get("Consistent Value Response", ""),
        "Contrastive Response": source_row.get("Contrastive Response", ""),
        "prediction": final_prediction,
        "best_candidate": best_candidate,
        "rerank_reason": rerank_reason,
        "judge_model": judge_model,
        "adapter_name": source_row.get("adapter_name", ""),
    }


def process_row(
    args: argparse.Namespace,
    row: dict[str, Any],
    local_idx: int,
    total_rows: int,
    row_index: int,
    key: str,
) -> dict[str, Any]:
    candidate_items = build_candidate_items(row)
    if not candidate_items:
        return {
            "status": "skipped",
            "local_idx": local_idx,
            "total_rows": total_rows,
            "message": "skip row without candidates",
        }

    user_prompt = build_user_prompt(row, candidate_items)
    try:
        client = get_thread_client(args)
        judgment = call_judge(client, args, user_prompt, set(candidate_items))
    except Exception as exc:
        failed_row = dict(row)
        failed_row.update(
            {
                "source_key": key,
                "row_index": row_index,
                "judge_model": args.model,
                "judge_base_url": args.base_url,
                "error": str(exc),
                "max_retries": args.max_retries,
            }
        )
        return {
            "status": "failed",
            "local_idx": local_idx,
            "total_rows": total_rows,
            "failed_row": failed_row,
        }

    best_candidate = judgment["best_candidate"]
    selected_text = candidate_items[best_candidate]
    rerank_row = dict(row)
    rerank_row.update(
        {
            "source_key": key,
            "row_index": row_index,
            "judge_model": args.model,
            "judge_base_url": args.base_url,
            "best_candidate": best_candidate,
            "final_prediction": selected_text,
            "confidence": judgment["confidence"],
            "reason": judgment.get("reason", ""),
            "judgment": judgment,
        }
    )

    return {
        "status": "success",
        "local_idx": local_idx,
        "total_rows": total_rows,
        "rerank_row": rerank_row,
        "best_candidate": best_candidate,
        "confidence": judgment["confidence"],
    }


def load_rerank_results(path: Path) -> dict[str, dict[str, Any]]:
    results = {}
    if not path.exists():
        return results
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("source_key")
            if key:
                results[str(key)] = row
    return results


def rebuild_final_predictions(
    indexed_rows: list[tuple[int, dict[str, Any]]],
    rerank_path: Path,
    final_path: Path,
    judge_model: str,
) -> int:
    reranked = load_rerank_results(rerank_path)
    final_rows: list[dict[str, Any]] = []
    for row_index, source_row in indexed_rows:
        key = source_key(source_row, row_index)
        rerank_row = reranked.get(key)
        if not rerank_row:
            continue
        final_rows.append(
            final_prediction_row(
                source_row=source_row,
                best_candidate=str(rerank_row.get("best_candidate", "")),
                final_prediction=str(rerank_row.get("final_prediction", "")),
                rerank_reason=str(rerank_row.get("reason", "")),
                judge_model=str(rerank_row.get("judge_model", judge_model)),
            )
        )
    write_jsonl(final_path, final_rows)
    return len(final_rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rerank_path = output_dir / "rerank_results.jsonl"
    final_path = output_dir / "final_predictions.jsonl"
    failed_path = output_dir / "failed_samples.jsonl"
    config_path = output_dir / "run_config.json"

    indexed_rows = load_jsonl_with_indices(input_path, limit=args.limit, start=args.start)
    completed = load_completed_keys(rerank_path, failed_path) if args.resume else set()
    work_items = [
        (row_index, row)
        for row_index, row in indexed_rows
        if source_key(row, row_index) not in completed
    ]

    write_json(
        config_path,
        {
            "input_file": str(input_path),
            "output_dir": str(output_dir),
            "rerank_results": str(rerank_path),
            "final_predictions": str(final_path),
            "failed_samples": str(failed_path),
            "model": args.model,
            "base_url": args.base_url,
            "api_key_env": args.api_key_env,
            "temperature": args.temperature,
            "num_workers": args.num_workers,
            "max_retries": args.max_retries,
            "retry_sleep": args.retry_sleep,
            "max_completion_tokens": args.max_completion_tokens,
            "timeout": args.timeout,
            "limit": args.limit,
            "start": args.start,
            "resume": args.resume,
            "loaded_rows": len(indexed_rows),
            "skipped_completed": len(indexed_rows) - len(work_items),
        },
    )

    print(f"Loaded {len(indexed_rows)} rows from {input_path}")
    print(f"Writing rerank details to {rerank_path}")
    print(f"Writing final predictions to {final_path}")
    if args.resume:
        print(f"Resume enabled; skipped {len(indexed_rows) - len(work_items)} completed rows")
    if work_items:
        preflight_api_config(args)

    success_count = 0
    failed_count = 0
    skipped_count = 0
    total = len(work_items)
    workers = max(1, args.num_workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_meta = {}
        for local_idx, (row_index, row) in enumerate(work_items, start=1):
            key = source_key(row, row_index)
            future = executor.submit(process_row, args, row, local_idx, total, row_index, key)
            future_to_meta[future] = (local_idx, row_index, key)

        for future in as_completed(future_to_meta):
            result = future.result()
            status = result["status"]
            local_idx = result["local_idx"]
            total_rows = result["total_rows"]

            if status == "success":
                append_jsonl(rerank_path, result["rerank_row"])
                success_count += 1
                print(
                    f"{local_idx}/{total_rows} success "
                    f"best={result['best_candidate']} confidence={result['confidence']}"
                )
            elif status == "failed":
                append_jsonl(failed_path, result["failed_row"])
                failed_count += 1
                print(f"{local_idx}/{total_rows} failed", file=sys.stderr)
            else:
                skipped_count += 1
                print(f"{local_idx}/{total_rows} skipped: {result['message']}")

    final_count = rebuild_final_predictions(indexed_rows, rerank_path, final_path, args.model)
    print(
        "Done: "
        f"success={success_count}, failed={failed_count}, skipped={skipped_count}, "
        f"final_predictions={final_count}"
    )


if __name__ == "__main__":
    main()
