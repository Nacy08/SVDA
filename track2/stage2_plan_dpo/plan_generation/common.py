import json
import os
import re
import sys
import time
from pathlib import Path

import yaml


SYSTEM_PROMPT = "You are a value-aligned response generator."

PLAN_PROMPT = """You are given a scenario, a question, and a Schwartz's Basic Human Value
Generate a concise value-aware plan for writing a response.

Important rules:
1. Do not write the final response.
2. Do not use any reference answer.
3. Identify key scenario details that are useful for reflecting the target value.
4. Identify value-specific keywords that should guide the response.
6. Keep the plan concise and specific.

Scenario:
{scenario}

Question:
{question}

Target Value:
{value}

Value Meaning:
{value_definition}

Return JSON with the following fields:
{{
  "scenario_keywords": ["..."],
  "question_keywords": ["..."],
  "decision_direction": "...",
  "generation_focus": "..."
}}"""

SFT_USER_PROMPT = """Given a scenario, a question, a target human value, and a value-aware plan, write a natural response that directly answers the question and clearly reflects the target value.

Requirements:
1. Directly answer the question.
2. Use the scenario information.
3. Make the target value the main reason for the response.
4. Use the plan naturally, but do not mechanically list keywords.
5. Avoid competing values that weaken the target value.
6. Keep the response concise, specific, and natural.

Scenario:
{scenario}

Question:
{question}

Target Value:
{value}

Value Meaning:
{value_definition}

Value-aware Plan:
- Scenario Keywords: {scenario_keywords}
- Question Keywords: {question_keywords}
- Decision Direction: {decision_direction}
- Generation Focus: {generation_focus}

Response:"""

RERANK_PROMPT = """You are given a scenario, a question, a target human value, and several candidate responses.
Your task is to choose the candidate response that best satisfies the following criteria:
1. It directly answers the question.
2. It is consistent with the target value.
3. It is grounded in the scenario.
4. It is clear, natural, and not overly verbose.
5. It avoids contradictions or irrelevant content.

Scenario:
{scenario}

Question:
{question}

Target Value:
{value}

Value Meaning:
{value_definition}

Candidate Responses:
{candidates}

Return JSON with fields:
{{
  "best_candidate": "A",
  "reason": "brief reason"
}}"""


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "scenario_keywords": {"type": "array", "items": {"type": "string"}},
        "question_keywords": {"type": "array", "items": {"type": "string"}},
        "decision_direction": {"type": "string"},
        "generation_focus": {"type": "string"},
    },
    "required": [
        "scenario_keywords",
        "question_keywords",
        "decision_direction",
        "generation_focus",
    ],
    "additionalProperties": False,
}

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "best_candidate": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
        "reason": {"type": "string"},
    },
    "required": ["best_candidate", "reason"],
    "additionalProperties": False,
}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path, limit=None, add_id=False):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if add_id and row.get("id") is None:
                row["id"] = str(idx)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def append_jsonl(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def completed_ids(path):
    done = set()
    path = Path(path)
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id") is not None:
                done.add(str(row["id"]))
    return done


def value_definition(row, value_definitions):
    existing = row.get("Value Definition")
    if existing:
        return existing
    return value_definitions.get(row.get("Value", ""), "")


def normalize_plan(plan):
    plan = plan or {}
    return {
        "scenario_keywords": [str(x) for x in plan.get("scenario_keywords", [])],
        "question_keywords": [str(x) for x in plan.get("question_keywords", [])],
        "decision_direction": str(plan.get("decision_direction", "")),
        "generation_focus": str(plan.get("generation_focus", "")),
    }


def format_plan_prompt(row, value_definitions):
    return PLAN_PROMPT.format(
        scenario=row.get("Scenario", ""),
        question=row.get("Question", ""),
        value=row.get("Value", ""),
        value_definition=value_definition(row, value_definitions),
    )


def format_sft_user_prompt(row, value_definitions):
    plan = normalize_plan(row.get("plan", {}))
    return SFT_USER_PROMPT.format(
        scenario=row.get("Scenario", ""),
        question=row.get("Question", ""),
        value=row.get("Value", ""),
        value_definition=value_definition(row, value_definitions),
        scenario_keywords=", ".join(plan["scenario_keywords"]),
        question_keywords=", ".join(plan["question_keywords"]),
        decision_direction=plan["decision_direction"],
        generation_focus=plan["generation_focus"],
    )


def extract_json(text):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def response_text(response):
    text = getattr(response, "output_text", None)
    if text:
        return text
    if hasattr(response, "choices"):
        return response.choices[0].message.content
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(value)
    return "\n".join(chunks)


def call_openai_json(client, model, prompt, schema, schema_name, temperature, max_output_tokens, timeout=None):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=max_output_tokens,
        timeout=timeout,
    )
    text = response_text(response)
    if not text.strip():
        choice = response.choices[0] if getattr(response, "choices", None) else None
        finish_reason = getattr(choice, "finish_reason", None)
        usage = getattr(response, "usage", None)
        raise RuntimeError(
            "empty model output; "
            f"finish_reason={finish_reason}; "
            f"usage={usage}; "
            "increase --max_output_tokens if finish_reason is 'length'"
        )
    return extract_json(text)


def error_chain(exc):
    messages = [f"{type(exc).__name__}: {exc}"]
    cause = getattr(exc, "__cause__", None)
    while cause:
        messages.append(f"caused by {type(cause).__name__}: {cause}")
        cause = getattr(cause, "__cause__", None)
    return " | ".join(messages)


def retry_json_call(fn, max_retries=3, sleep_base=1.0):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(), None
        except Exception as exc:  # noqa: BLE001 - keep batch jobs resumable.
            last_error = error_chain(exc)
            if attempt < max_retries:
                time.sleep(min(sleep_base * (2 ** (attempt - 1)), 8))
    return None, last_error


def require_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing API key. Please set environment variable {name}.")
    return value


def checkpoint_name(path):
    path = Path(path)
    return path.name or re.sub(r"[^A-Za-z0-9_.-]+", "_", str(path))


def add_project_root_to_path():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
