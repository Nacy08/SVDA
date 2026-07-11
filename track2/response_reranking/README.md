# Qwen Rerank for DPO

`qwen_rerank.py` 用于把 SFT 模型采样得到的多个候选回复交给 `qwen3.6-flash` 排序，并生成后续 DPO 训练需要的 `prompt/chosen/rejected` 偏好对。

接口使用 CloseAI 的 OpenAI 兼容 `/v1/chat/completions`，文档要求 `base_url` 带 `/v1` 后缀。

## 输入与输出

输入文件：

```text
/home/lanxin/NLPCC Task2/Sample/train_samples/nlpcc_task2_checkpoint-1100/predictions.jsonl
```

每行需要包含：

```text
Scenario
Question
Value
Consistent Value Response
Contrastive Response
candidates
```

输出目录中会生成：

```text
rerank_results.jsonl  # Qwen 完整排序与判断
dpo_pairs.jsonl       # DPO 偏好对
failed_samples.jsonl  # 连续重试失败后跳过的样本
run_config.json       # 本次运行参数
```

## 运行命令

先设置 API Key：

```bash
export OPENAI_API_KEY="sk-your-api-key"
```

小规模测试：

```bash
PYTHONUNBUFFERED=1 python "/home/lanxin/NLPCC Task2/Rerank/qwen_rerank.py" \
  --input_file "/home/lanxin/NLPCC Task2/Sample/train_samples/nlpcc_task2_checkpoint-1100/predictions.jsonl" \
  --output_dir "/home/lanxin/NLPCC Task2/Rerank/qwen_rerank_outputs_smoke" \
  --model "qwen3.6-flash" \
  --base_url "https://api.openai-proxy.org/v1" \
  --num_workers 4 \
  --limit 5
```

正式运行：

```bash
PYTHONUNBUFFERED=1 python "/home/lanxin/NLPCC Task2/Rerank/qwen_rerank.py" \
  --input_file "/home/lanxin/NLPCC Task2/Sample/train_samples/nlpcc_task2_checkpoint-1100/predictions.jsonl" \
  --output_dir "/home/lanxin/NLPCC Task2/Rerank/qwen_rerank_outputs" \
  --model "qwen3.6-flash" \
  --base_url "https://api.openai-proxy.org/v1" \
  --num_workers 4 \
  --resume
```

## 常用参数

```text
--num_workers 4
```

并发 API 请求数。`qwen3.6-flash` 限流较高，但建议先用 4；稳定后可试 8 或 16。

```text
--limit 5
```

只处理前 5 条，用于测试 API、输出格式和偏好对质量。

```text
--start 1000 --limit 500
```

从第 1000 条开始处理 500 条，适合分段跑。

```text
--resume
```

跳过 `rerank_results.jsonl` 和 `failed_samples.jsonl` 中已经完成或失败的样本，推荐正式运行时一直开启。

```text
--max_retries 3
```

每条样本最多重试 3 次。连续失败后不会中断全局任务，而是写入 `failed_samples.jsonl` 并继续下一条。

```text
--max_pairs_per_example 4
```

每条样本最多输出 4 个 DPO pair，默认值推荐保留。

## DPO Pair 规则

默认按优先级构造：

```text
1. gold > contrastive
2. best_sample > contrastive
3. best_sample > worst_sample
4. best_sample > lower-ranked sample
5. gold > bad_candidate
```

默认不开启 `sample > gold`。如需更激进偏好对，再加：

```text
--include_sample_vs_gold
```

## 检查输出

```bash
head -n 2 "/home/lanxin/NLPCC Task2/Rerank/qwen_rerank_outputs/dpo_pairs.jsonl"
head -n 1 "/home/lanxin/NLPCC Task2/Rerank/qwen_rerank_outputs/rerank_results.jsonl"
wc -l "/home/lanxin/NLPCC Task2/Rerank/qwen_rerank_outputs/"*.jsonl
```
