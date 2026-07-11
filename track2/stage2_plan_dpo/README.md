# Plan_DPO Round2 Hard-Value Plan-Guided DPO Data Pipeline

本目录用于第二轮 DPO 数据构造，不包含 DPO 训练脚本。

目标流程：

1. hard-value plan-guided G=16 candidate generation
2. Qwen rerank and raw pair construction
3. select final 6000 DPO pairs

## Data

- plan file: `/home/lanxin/NLPCC Task2/Plan_DPO/Plan/train_plan_gpt5mini.jsonl`
- train size: 3520
- hard-value size: 3068
- hard tiers:
  - high: 1306
  - mid: 1043
  - low: 719

Hard values:

```text
high: Face, Power-resources, Power-dominance, Hedonism, Humility, Benevolence-dependability, Security-personal
mid: Benevolence-caring, Stimulation, Conformity-interpersonal, Tradition
low: Conformity-rules, Universalism-concern, Achievement
```

## 1. Generate Hard-Value Plan-Guided Candidates

Smoke test:

```bash
python "/home/lanxin/NLPCC Task2/Plan_DPO/Sample/generate_round2_hard_plan_candidates.py" \
  --input_file "/home/lanxin/NLPCC Task2/Plan_DPO/Plan/train_plan_gpt5mini.jsonl" \
  --output_file "/home/lanxin/NLPCC Task2/Plan_DPO/hard_plan_g16_predictions_smoke/predictions.jsonl" \
  --model_name_or_path "/home/lanxin/NLPCC Task2/Qlora_Sft_Model/Meta-Llama-3.1-8B-Instruct" \
  --adapter_path "/home/lanxin/NLPCC Task2/DPO/dpo_adapter" \
  --num_candidates 2 \
  --limit 3 \
  --resume
```

Full generation:

```bash
python "/home/lanxin/NLPCC Task2/Plan_DPO/Sample/generate_round2_hard_plan_candidates.py" \
  --input_file "/home/lanxin/NLPCC Task2/Plan_DPO/Plan/train_plan_gpt5mini.jsonl" \
  --output_file "/home/lanxin/NLPCC Task2/Plan_DPO/hard_plan_g16_predictions/predictions.jsonl" \
  --model_name_or_path "/home/lanxin/NLPCC Task2/Qlora_Sft_Model/Meta-Llama-3.1-8B-Instruct" \
  --adapter_path "/home/lanxin/NLPCC Task2/DPO/dpo_adapter" \
  --num_candidates 16 \
  --temperature 0.8 \
  --top_p 0.9 \
  --max_new_tokens 160 \
  --seed 42 \
  --resume
```

Notes:

- Generation uses the existing `plan` field.
- The output `prompt` field intentionally does not include plan, so later DPO training prompt remains consistent with test-time prompt.
- Invalid rows are written to `skipped_rows.jsonl`.

## 2. Qwen Rerank and Raw Pair Construction

Smoke test:

```bash
export OPENAI_API_KEY="your CloseAI key"

python "/home/lanxin/NLPCC Task2/Rerank/qwen_rerank.py" \
  --input_file "/home/lanxin/NLPCC Task2/Plan_DPO/hard_plan_g16_predictions_smoke/predictions.jsonl" \
  --output_dir "/home/lanxin/NLPCC Task2/Plan_DPO/qwen_rerank_round2_hard_plan_g16_smoke" \
  --model qwen3.6-flash \
  --base_url "https://api.openai-proxy.org/v1" \
  --temperature 0.0 \
  --num_workers 1 \
  --min_confidence medium \
  --max_pairs_per_example 4 \
  --min_len_ratio 0.7 \
  --max_len_ratio 1.6 \
  --max_chosen_words 120 \
  --include_sample_vs_gold \
  --resume
```

Full rerank:

```bash
export OPENAI_API_KEY="your CloseAI key"

python "/home/lanxin/NLPCC Task2/Rerank/qwen_rerank.py" \
  --input_file "/home/lanxin/NLPCC Task2/Plan_DPO/hard_plan_g16_predictions/predictions.jsonl" \
  --output_dir "/home/lanxin/NLPCC Task2/Plan_DPO/qwen_rerank_round2_hard_plan_g16" \
  --model qwen3.6-flash \
  --base_url "https://api.openai-proxy.org/v1" \
  --temperature 0.0 \
  --num_workers 4 \
  --min_confidence medium \
  --max_pairs_per_example 4 \
  --min_len_ratio 0.7 \
  --max_len_ratio 1.6 \
  --max_chosen_words 120 \
  --include_sample_vs_gold \
  --resume
```

The existing Qwen script writes:

```text
qwen_rerank_round2_hard_plan_g16/rerank_results.jsonl
qwen_rerank_round2_hard_plan_g16/dpo_pairs.jsonl
```

`dpo_pairs.jsonl` is the raw pair file for round2 selection. `--include_sample_vs_gold` must be enabled because the second-round objective relies on sample > gold pairs.

## 3. Select Final 6000 Pairs

Smoke selection from smoke rerank output:

```bash
python "/home/lanxin/NLPCC Task2/Plan_DPO/Rerank/select_round2_pairs.py" \
  --raw_pairs_file "/home/lanxin/NLPCC Task2/Plan_DPO/qwen_rerank_round2_hard_plan_g16_smoke/dpo_pairs.jsonl" \
  --output_file "/home/lanxin/NLPCC Task2/Plan_DPO/dpo_pairs_round2_selected_smoke.jsonl" \
  --target_total 20 \
  --seed 42
```

Full selection:

```bash
python "/home/lanxin/NLPCC Task2/Plan_DPO/Rerank/select_round2_pairs.py" \
  --raw_pairs_file "/home/lanxin/NLPCC Task2/Plan_DPO/qwen_rerank_round2_hard_plan_g16/dpo_pairs.jsonl" \
  --replay_pairs_file "/home/lanxin/NLPCC Task2/Rerank/qwen_rerank_outputs/dpo_pairs.jsonl" \
  --output_file "/home/lanxin/NLPCC Task2/Plan_DPO/dpo_pairs_round2_selected_6000.jsonl" \
  --target_total 6000 \
  --seed 42
```

Outputs:

```text
/home/lanxin/NLPCC Task2/Plan_DPO/dpo_pairs_round2_selected_6000.jsonl
/home/lanxin/NLPCC Task2/Plan_DPO/round2_pair_stats.json
/home/lanxin/NLPCC Task2/Plan_DPO/round2_pair_stats.md
```

Selection targets:

| Source | Target |
|---|---:|
| hard new pairs | 4800 |
| replay pairs | 1200 |
| total | 6000 |

Hard new pair quotas:

| Group | Target | high | mid | low |
|---|---:|---:|---:|---:|
| sample_vs_gold | 2400 | 1200 | 840 | 360 |
| sample_vs_sample | 1500 | 750 | 525 | 225 |
| quality_repair | 600 | 300 | 210 | 90 |
| gold_vs_contrastive | 300 | 150 | 105 | 45 |

Pair type mapping:

```text
sample_vs_gold:
  sample_vs_gold

sample_vs_sample:
  best_sample_vs_worst_sample
  best_sample_vs_lower_sample

quality_repair:
  gold_vs_bad_candidate
  best_sample_vs_contrastive

gold_vs_contrastive:
  gold_vs_contrastive
```

Filtering:

- hard new pairs require hard value and confidence high/medium.
- de-duplicate `(prompt, chosen, rejected)`.
- require non-empty chosen/rejected and chosen != rejected.
- require `chosen_words <= 120`.
- require `0.7 <= chosen_words / rejected_words <= 1.6`.
- require repeated 4-gram ratio `<= 0.1`.

Fallback order:

1. same tier, adjacent pair group
2. same pair group, other tier
3. all hard raw high/medium pairs
4. record unfilled buckets in stats

Replay:

- Prefer `--replay_pairs_file`.
- If insufficient, backfill from raw non-hard pairs or `gold_vs_contrastive`.
- Replay also uses de-duplication and length filters.
