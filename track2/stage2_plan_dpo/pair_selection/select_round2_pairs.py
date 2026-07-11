#!/usr/bin/env python3
"""Select final round2 DPO pairs from Qwen raw pairs."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/home/lanxin/NLPCC Task2")
PLAN_DPO_ROOT = ROOT / "Plan_DPO"
DEFAULT_RAW_PAIRS = PLAN_DPO_ROOT / "qwen_rerank_round2_hard_plan_g16" / "dpo_pairs.jsonl"
DEFAULT_OUTPUT = PLAN_DPO_ROOT / "dpo_pairs_round2_selected_6000.jsonl"

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
HARD_VALUES = set(HIGH_VALUES + MID_VALUES + LOW_VALUES)
VALUE_TO_TIER = {value: "high" for value in HIGH_VALUES}
VALUE_TO_TIER.update({value: "mid" for value in MID_VALUES})
VALUE_TO_TIER.update({value: "low" for value in LOW_VALUES})

PAIR_GROUPS = {
    "sample_vs_gold": ["sample_vs_gold"],
    "sample_vs_sample": ["best_sample_vs_worst_sample", "best_sample_vs_lower_sample"],
    "quality_repair": ["gold_vs_bad_candidate", "best_sample_vs_contrastive"],
    "gold_vs_contrastive": ["gold_vs_contrastive"],
}
PAIR_TYPE_TO_GROUP = {
    pair_type: group for group, pair_types in PAIR_GROUPS.items() for pair_type in pair_types
}
GROUP_WEIGHTS = {
    "sample_vs_gold": 2400,
    "sample_vs_sample": 1500,
    "quality_repair": 600,
    "gold_vs_contrastive": 300,
}
TIER_WEIGHTS = {"high": 50, "mid": 35, "low": 15}
ADJACENT_GROUPS = {
    "sample_vs_gold": ["sample_vs_sample", "quality_repair", "gold_vs_contrastive"],
    "sample_vs_sample": ["sample_vs_gold", "quality_repair", "gold_vs_contrastive"],
    "quality_repair": ["sample_vs_sample", "gold_vs_contrastive", "sample_vs_gold"],
    "gold_vs_contrastive": ["quality_repair", "sample_vs_sample", "sample_vs_gold"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select final 6000 round2 DPO pairs.")
    parser.add_argument("--raw_pairs_file", default=str(DEFAULT_RAW_PAIRS))
    parser.add_argument("--replay_pairs_file", default=None)
    parser.add_argument("--output_file", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--target_total", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_len_ratio", type=float, default=0.7)
    parser.add_argument("--max_len_ratio", type=float, default=1.6)
    parser.add_argument("--max_chosen_words", type=int, default=120)
    parser.add_argument("--max_rep4", type=float, default=0.1)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def word_count(text: str) -> int:
    return len(str(text).split())


def len_ratio(chosen: str, rejected: str) -> float:
    return word_count(chosen) / max(word_count(rejected), 1)


def rep4_ratio(text: str) -> float:
    words = str(text).split()
    if len(words) < 4:
        return 0.0
    grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(len(grams), 1)


def pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("prompt", "")).strip(),
        str(row.get("chosen", "")).strip(),
        str(row.get("rejected", "")).strip(),
    )


def value_tier(value: str) -> str:
    return VALUE_TO_TIER.get(value, "non_hard")


def build_meta_map(rerank_results_file: Path) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(rerank_results_file):
        source_key = str(row.get("source_key", ""))
        if not source_key:
            continue
        judgment = row.get("judgment") or {}
        meta[source_key] = {
            "confidence": str(judgment.get("confidence", "")).lower(),
            "best_sample": judgment.get("best_sample", ""),
            "value_tier": row.get("value_tier") or value_tier(str(row.get("Value", ""))),
            "generation_mode": row.get("generation_mode", ""),
            "row_index": row.get("row_index"),
        }
    return meta


def enrich_pair(row: dict[str, Any], meta_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    enriched = dict(row)
    source_key = str(enriched.get("source_key", ""))
    meta = meta_map.get(source_key, {})
    value = str(enriched.get("Value", ""))
    enriched["confidence"] = enriched.get("confidence") or meta.get("confidence", "")
    enriched["value_tier"] = enriched.get("value_tier") or meta.get("value_tier") or value_tier(value)
    enriched["generation_mode"] = enriched.get("generation_mode") or meta.get("generation_mode", "")
    if "source_row_index" not in enriched and meta.get("row_index") is not None:
        enriched["source_row_index"] = meta.get("row_index")
    enriched["pair_group"] = PAIR_TYPE_TO_GROUP.get(str(enriched.get("pair_type", "")), "other")
    return enriched


def valid_pair(row: dict[str, Any], args: argparse.Namespace, stats: Counter[str] | None = None) -> bool:
    chosen = str(row.get("chosen", "")).strip()
    rejected = str(row.get("rejected", "")).strip()
    if not chosen or not rejected or chosen == rejected:
        if stats is not None:
            stats["invalid_empty_or_same"] += 1
        return False
    if word_count(chosen) > args.max_chosen_words:
        if stats is not None:
            stats["overlong_chosen"] += 1
        return False
    ratio = len_ratio(chosen, rejected)
    if ratio < args.min_len_ratio or ratio > args.max_len_ratio:
        if stats is not None:
            stats["bad_len_ratio"] += 1
        return False
    if rep4_ratio(chosen) > args.max_rep4:
        if stats is not None:
            stats["rep4_filtered"] += 1
        return False
    return True


def allocate(total: int, weights: dict[str, int]) -> dict[str, int]:
    weight_sum = sum(weights.values())
    raw = {key: total * weight / weight_sum for key, weight in weights.items()}
    out = {key: math.floor(value) for key, value in raw.items()}
    remainder = total - sum(out.values())
    order = sorted(weights, key=lambda key: raw[key] - out[key], reverse=True)
    for key in order[:remainder]:
        out[key] += 1
    return out


def shuffle_pools(pools: dict[str, list[dict[str, Any]]], rng: random.Random) -> None:
    for rows in pools.values():
        rng.shuffle(rows)


def take_from_pool(
    pool: list[dict[str, Any]],
    need: int,
    selected_keys: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    out = []
    if need <= 0:
        return out
    for row in pool:
        key = pair_key(row)
        if key in selected_keys:
            continue
        selected_keys.add(key)
        out.append(row)
        if len(out) >= need:
            break
    return out


def collect_bucket(
    group: str,
    tier: str,
    need: int,
    pools: dict[tuple[str, str], list[dict[str, Any]]],
    all_hard_pool: list[dict[str, Any]],
    selected_keys: set[tuple[str, str, str]],
    fallback_counts: Counter[str],
) -> list[dict[str, Any]]:
    selected = take_from_pool(pools[(group, tier)], need, selected_keys)
    if len(selected) >= need:
        return selected
    missing = need - len(selected)
    for other_group in ADJACENT_GROUPS[group]:
        added = take_from_pool(pools[(other_group, tier)], missing, selected_keys)
        if added:
            fallback_counts[f"{group}/{tier}:same_tier_{other_group}"] += len(added)
        selected.extend(added)
        missing = need - len(selected)
        if missing <= 0:
            return selected
    for other_tier in ("high", "mid", "low"):
        if other_tier == tier:
            continue
        added = take_from_pool(pools[(group, other_tier)], missing, selected_keys)
        if added:
            fallback_counts[f"{group}/{tier}:same_group_{other_tier}"] += len(added)
        selected.extend(added)
        missing = need - len(selected)
        if missing <= 0:
            return selected
    added = take_from_pool(all_hard_pool, missing, selected_keys)
    if added:
        fallback_counts[f"{group}/{tier}:global_hard"] += len(added)
    selected.extend(added)
    missing = need - len(selected)
    if missing > 0:
        fallback_counts[f"{group}/{tier}:unfilled"] += missing
    return selected


def prepare_rows(
    raw_pairs_file: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta_map = build_meta_map(raw_pairs_file.parent / "rerank_results.jsonl")
    raw_rows = [enrich_pair(row, meta_map) for row in read_jsonl(raw_pairs_file)]
    stats = Counter()
    dedup_seen: set[tuple[str, str, str]] = set()
    filtered = []
    for row in raw_rows:
        key = pair_key(row)
        if key in dedup_seen:
            stats["dedup_removed"] += 1
            continue
        dedup_seen.add(key)
        if not valid_pair(row, args, stats):
            continue
        filtered.append(row)
    return filtered, {
        "raw_pairs": len(raw_rows),
        "filtered_pairs": len(filtered),
        "dedup_removed": stats["dedup_removed"],
        "rep4_filtered": stats["rep4_filtered"],
        "filter_counts": dict(stats),
        "meta_found": len(meta_map),
    }


def add_selection_metadata(
    rows: list[dict[str, Any]],
    source: str,
    target_group: str | None = None,
    target_tier: str | None = None,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["round2_selection_source"] = source
        if target_group is not None:
            item["round2_target_group"] = target_group
        if target_tier is not None:
            item["round2_target_tier"] = target_tier
        out.append(item)
    return out


def select_hard_pairs(
    rows: list[dict[str, Any]],
    target_total: int,
    rng: random.Random,
    selected_keys: set[tuple[str, str, str]],
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, Any]]:
    hard_target = int(round(target_total * 0.8))
    group_quotas = allocate(hard_target, GROUP_WEIGHTS)
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    all_hard_pool = []
    pool_counts = Counter()
    for row in rows:
        value = str(row.get("Value", ""))
        group = str(row.get("pair_group", ""))
        tier = str(row.get("value_tier", ""))
        confidence = str(row.get("confidence", "")).lower()
        if value not in HARD_VALUES or group not in GROUP_WEIGHTS or confidence not in {"high", "medium"}:
            continue
        pools[(group, tier)].append(row)
        pool_counts[f"{group}/{tier}"] += 1
        if confidence in {"high", "medium"} and tier in {"high", "mid", "low"}:
            all_hard_pool.append(row)
    shuffle_pools(pools, rng)
    rng.shuffle(all_hard_pool)

    selected = []
    fallback_counts: Counter[str] = Counter()
    requested = {}
    for group, group_total in group_quotas.items():
        tier_quotas = allocate(group_total, TIER_WEIGHTS)
        for tier, need in tier_quotas.items():
            requested[f"{group}/{tier}"] = need
            bucket_rows = collect_bucket(
                group,
                tier,
                need,
                pools,
                all_hard_pool,
                selected_keys,
                fallback_counts,
            )
            selected.extend(add_selection_metadata(bucket_rows, "hard_new", group, tier))
    stats = {
        "hard_target": hard_target,
        "hard_selected": len(selected),
        "requested_buckets": requested,
        "available_pool_counts": dict(pool_counts),
    }
    return selected, fallback_counts, stats


def select_replay_pairs(
    rows: list[dict[str, Any]],
    replay_pairs_file: str | None,
    target_total: int,
    rng: random.Random,
    selected_keys: set[tuple[str, str, str]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    replay_target = target_total - int(round(target_total * 0.8))
    replay_selected: list[dict[str, Any]] = []
    replay_stats = Counter()

    replay_file_rows: list[dict[str, Any]] = []
    if replay_pairs_file:
        replay_file_rows = read_jsonl(Path(replay_pairs_file))
        rng.shuffle(replay_file_rows)
        clean_replay = []
        local_stats: Counter[str] = Counter()
        for row in replay_file_rows:
            enriched = dict(row)
            enriched["value_tier"] = enriched.get("value_tier") or value_tier(str(enriched.get("Value", "")))
            enriched["pair_group"] = PAIR_TYPE_TO_GROUP.get(str(enriched.get("pair_type", "")), "replay")
            if valid_pair(enriched, args, local_stats):
                clean_replay.append(enriched)
        added = take_from_pool(clean_replay, replay_target, selected_keys)
        replay_selected.extend(add_selection_metadata(added, "replay_file"))
        replay_stats["from_replay_file"] = len(added)
        replay_stats.update({f"replay_file_filter_{key}": value for key, value in local_stats.items()})

    missing = replay_target - len(replay_selected)
    if missing > 0:
        fallback_pool = []
        for row in rows:
            value = str(row.get("Value", ""))
            pair_type = str(row.get("pair_type", ""))
            if value not in HARD_VALUES or pair_type == "gold_vs_contrastive":
                fallback_pool.append(row)
        rng.shuffle(fallback_pool)
        added = take_from_pool(fallback_pool, missing, selected_keys)
        replay_selected.extend(add_selection_metadata(added, "raw_fallback"))
        replay_stats["from_raw_fallback"] = len(added)
    missing = replay_target - len(replay_selected)
    if missing > 0:
        replay_stats["unfilled"] = missing
    return replay_selected, {"replay_target": replay_target, "replay_selected": len(replay_selected), **dict(replay_stats)}


def summarize(rows: list[dict[str, Any]], base_stats: dict[str, Any], fallback_counts: Counter[str], selection_stats: dict[str, Any]) -> dict[str, Any]:
    chosen_lens = [word_count(row.get("chosen", "")) for row in rows]
    rejected_lens = [word_count(row.get("rejected", "")) for row in rows]
    ratios = [len_ratio(row.get("chosen", ""), row.get("rejected", "")) for row in rows]
    return {
        "total_pairs": len(rows),
        "hard_new_pairs": sum(1 for row in rows if row.get("round2_selection_source") == "hard_new"),
        "replay_pairs": sum(1 for row in rows if row.get("round2_selection_source") != "hard_new"),
        "by_pair_type": dict(Counter(str(row.get("pair_type", "")) for row in rows)),
        "by_pair_group": dict(Counter(str(row.get("pair_group", "")) for row in rows)),
        "by_value": dict(Counter(str(row.get("Value", "")) for row in rows)),
        "by_value_tier": dict(Counter(str(row.get("value_tier", "")) for row in rows)),
        "by_confidence": dict(Counter(str(row.get("confidence", "")) for row in rows)),
        "fallback_counts": dict(fallback_counts),
        "avg_chosen_words": statistics.mean(chosen_lens) if chosen_lens else 0,
        "avg_rejected_words": statistics.mean(rejected_lens) if rejected_lens else 0,
        "median_len_ratio": statistics.median(ratios) if ratios else 0,
        "max_len_ratio": max(ratios) if ratios else 0,
        "rep4_filtered": base_stats.get("rep4_filtered", 0),
        "dedup_removed": base_stats.get("dedup_removed", 0),
        "base_stats": base_stats,
        "selection_stats": selection_stats,
    }


def write_stats(stats_path: Path, md_path: Path, stats: dict[str, Any]) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Round2 Pair Selection Stats\n\n")
        f.write(f"- total_pairs: {stats['total_pairs']}\n")
        f.write(f"- hard_new_pairs: {stats['hard_new_pairs']}\n")
        f.write(f"- replay_pairs: {stats['replay_pairs']}\n")
        f.write(f"- avg_chosen_words: {stats['avg_chosen_words']:.2f}\n")
        f.write(f"- avg_rejected_words: {stats['avg_rejected_words']:.2f}\n")
        f.write(f"- median_len_ratio: {stats['median_len_ratio']:.3f}\n")
        f.write(f"- max_len_ratio: {stats['max_len_ratio']:.3f}\n")
        f.write(f"- dedup_removed: {stats['dedup_removed']}\n")
        f.write(f"- rep4_filtered: {stats['rep4_filtered']}\n\n")
        for section in ("by_pair_group", "by_pair_type", "by_value_tier", "by_confidence", "fallback_counts"):
            f.write(f"## {section}\n\n")
            f.write("| key | count |\n|---|---:|\n")
            for key, value in sorted(stats.get(section, {}).items()):
                f.write(f"| {key} | {value} |\n")
            f.write("\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    raw_pairs_file = Path(args.raw_pairs_file)
    output_file = Path(args.output_file)
    stats_json = output_file.with_name("round2_pair_stats.json")
    stats_md = output_file.with_name("round2_pair_stats.md")

    rows, base_stats = prepare_rows(raw_pairs_file, args)
    selected_keys: set[tuple[str, str, str]] = set()
    hard_rows, fallback_counts, hard_stats = select_hard_pairs(
        rows, args.target_total, rng, selected_keys
    )
    replay_rows, replay_stats = select_replay_pairs(
        rows,
        args.replay_pairs_file,
        args.target_total,
        rng,
        selected_keys,
        args,
    )
    selected = hard_rows + replay_rows
    rng.shuffle(selected)
    write_jsonl(output_file, selected)
    stats = summarize(
        selected,
        base_stats,
        fallback_counts,
        {"hard": hard_stats, "replay": replay_stats, "target_total": args.target_total},
    )
    write_stats(stats_json, stats_md, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
