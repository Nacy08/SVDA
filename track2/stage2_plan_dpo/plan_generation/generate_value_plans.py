#!/usr/bin/env python3
import argparse
import json
import os
import time
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path

import httpx
from openai import OpenAI

from common import (
    PLAN_SCHEMA,
    append_jsonl,
    call_openai_json,
    completed_ids,
    format_plan_prompt,
    load_json,
    load_yaml,
    read_jsonl,
    require_env,
    retry_json_call,
    value_definition,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate value-aware plans with gpt-5-mini.")
    parser.add_argument("--config", default="/root/Plan_Qlora/configs/plan_sft.yaml")
    parser.add_argument("--split", choices=["train", "dev", "both"], default="both")
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--ca_bundle", default=None)
    parser.add_argument("--ssl_verify", choices=["true", "false"], default=None)
    parser.add_argument("--no_trust_env", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_output_tokens", type=int, default=1024)
    return parser.parse_args()


def split_paths(cfg, split):
    if split == "train":
        return cfg["train_raw"], cfg["train_plan"]
    if split == "dev":
        return cfg["dev_raw"], cfg["dev_plan"]
    raise ValueError(split)


def generate_split(args, cfg, split, client, value_definitions):
    input_file, output_file = split_paths(cfg, split)
    input_file = args.input_file or input_file
    output_file = args.output_file or output_file
    output_path = Path(output_file)
    log_path = Path(cfg.get("log_dir", "/root/Plan_Qlora/logs")) / f"{split}_plan_failures.jsonl"
    model = args.model or cfg.get("plan_model", "gpt-5-mini")
    rows = read_jsonl(input_file, limit=args.limit, add_id=True)
    done = completed_ids(output_path)

    print(f"[{split}] loaded {len(rows)} rows from {input_file}")
    print(f"[{split}] skipping {len(done)} completed rows in {output_file}")

    max_retries = int(cfg.get("max_retries", 3))
    timeout = int(cfg.get("request_timeout", 120))
    request_interval = float(cfg.get("request_interval", 0.0))
    prompt_version = cfg.get("plan_prompt_version", "plan_v1")

    for idx, row in enumerate(rows, start=1):
        row_id = str(row["id"])
        if row_id in done:
            continue

        prompt = format_plan_prompt(row, value_definitions)

        def call():
            return call_openai_json(
                client=client,
                model=model,
                prompt=prompt,
                schema=PLAN_SCHEMA,
                schema_name="value_aware_plan",
                temperature=None,
                max_output_tokens=args.max_output_tokens,
                timeout=timeout,
            )

        plan, error = retry_json_call(call, max_retries=max_retries)
        if error:
            append_jsonl(
                log_path,
                {"id": row_id, "split": split, "error": error, "prompt": prompt},
            )
            print(f"[{split}] failed id={row_id}: {error}")
            continue

        output = dict(row)
        output["Value Definition"] = value_definition(row, value_definitions)
        output["plan"] = plan
        output["plan_model"] = model
        output["plan_prompt_version"] = prompt_version
        append_jsonl(output_path, output)
        done.add(row_id)

        if idx % 25 == 0 or idx == len(rows):
            print(f"[{split}] {idx}/{len(rows)}")
        if request_interval > 0:
            time.sleep(request_interval)


def normalize_openai_base_url(base_url):
    if not base_url:
        return None
    base_url = base_url.strip()
    if "://" not in base_url:
        scheme = "http" if looks_like_http_only_host(base_url) else "https"
        base_url = f"{scheme}://{base_url}"
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    for suffix in ("/chat/completions", "/responses", "/completions", "/models"):
        if path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
    if not path:
        path = "/v1"
    elif not path.endswith("/v1"):
        path = f"{path}/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def looks_like_http_only_host(base_url):
    host = base_url.split("/", 1)[0].split(":", 1)[0].lower()
    if host in {"localhost", "127.0.0.1", "0.0.0.0"}:
        return True
    if host.startswith(("10.", "192.168.")):
        return True
    if host.startswith("172."):
        parts = host.split(".")
        if len(parts) > 1 and parts[1].isdigit() and 16 <= int(parts[1]) <= 31:
            return True
    return False


def env_bool(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def build_http_client(args):
    ssl_verify = args.ssl_verify or os.environ.get("OPENAI_SSL_VERIFY")
    verify = env_bool("OPENAI_SSL_VERIFY") if ssl_verify is None else ssl_verify == "true"
    ca_bundle = args.ca_bundle or os.environ.get("OPENAI_CA_BUNDLE")
    if ca_bundle:
        verify = ca_bundle

    proxy = args.proxy or os.environ.get("OPENAI_PROXY")
    trust_env = not args.no_trust_env and env_bool("OPENAI_TRUST_ENV", default=True)

    if verify is False:
        print("[openai] SSL certificate verification disabled")
    if ca_bundle:
        print(f"[openai] using CA bundle={ca_bundle}")
    if proxy:
        print(f"[openai] using proxy={proxy}")
    if not trust_env:
        print("[openai] ignoring proxy/SSL settings from environment")

    return httpx.Client(verify=verify, proxy=proxy, trust_env=trust_env)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    api_key = require_env("OPENAI_API_KEY")
    value_definitions = load_json(cfg["value_definitions"])
    client_kwargs = {"api_key": api_key, "http_client": build_http_client(args)}
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    base_url = normalize_openai_base_url(base_url)
    if base_url:
        client_kwargs["base_url"] = base_url
        print(f"[openai] using base_url={base_url}")
    else:
        print("[openai] using default base_url=https://api.openai.com/v1")
    client = OpenAI(**client_kwargs)
    splits = ["train", "dev"] if args.split == "both" else [args.split]
    for split in splits:
        generate_split(args, cfg, split, client, value_definitions)


if __name__ == "__main__":
    main()
