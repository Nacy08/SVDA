# TRL LoRA-DPO Training

本目录用于第一阶段离线 DPO：读取 GPT-5 rerank 生成的 `dpo_pairs.jsonl`，在当前 SFT LoRA adapter 基础上继续训练一个 DPO adapter。

## 环境

推荐环境：

```bash
conda activate nlpcc_task2
```

已适配的核心版本：

```text
python 3.12.3
torch 2.8.0+cu128
transformers 5.8.0
peft 0.19.1
trl 1.3.0
bitsandbytes 0.49.2
```

## 输入输出

默认输入：

```text
/home/lanxin/NLPCC Task2/Rerank/gpt5_rerank_outputs/dpo_pairs.jsonl
```

每行需要包含：

```text
prompt
chosen
rejected
pair_type
```

默认输出：

```text
/home/lanxin/NLPCC Task2/DPO/dpo_adapter
```

输出目录会保存 DPO LoRA adapter、tokenizer 和 `run_config.json`。

## Smoke Test

```bash
conda activate nlpcc_task2

CUDA_VISIBLE_DEVICES=2 python "/home/lanxin/NLPCC Task2/DPO/train_dpo_trl.py" \
  --base_model "/home/lanxin/NLPCC Task2/Qlora_Sft_Model/Meta-Llama-3.1-8B-Instruct" \
  --sft_adapter "/home/lanxin/NLPCC Task2/Qlora_Sft_Model/nlpcc_task2_checkpoint-1100" \
  --train_file "/home/lanxin/NLPCC Task2/Rerank/gpt5_rerank_outputs/dpo_pairs.jsonl" \
  --output_dir "/home/lanxin/NLPCC Task2/DPO/dpo_adapter_smoke" \
  --limit 32 \
  --max_steps 5 \
  --max_length 512 \
  --max_prompt_length 512
```

注意：当前环境的 `trl 1.3.0` 中 `DPOConfig` 不暴露 `max_prompt_length`，脚本会记录该参数，但实际截断由 `--max_length` 控制。

## 正式训练

A100 推荐配置：

```bash
conda activate nlpcc_task2

CUDA_VISIBLE_DEVICES=7 python "/home/lanxin/NLPCC Task2/DPO/train_dpo_trl.py" \
  --base_model "/home/lanxin/NLPCC Task2/Qlora_Sft_Model/Meta-Llama-3.1-8B-Instruct" \
  --sft_adapter "/home/lanxin/NLPCC Task2/Qlora_Sft_Model/nlpcc_task2_checkpoint-1100" \
  --train_file "/home/lanxin/NLPCC Task2/Rerank/gpt5_rerank_outputs/dpo_pairs.jsonl" \
  --output_dir "/home/lanxin/NLPCC Task2/DPO/dpo_adapter" \
  --beta 0.1 \
  --learning_rate 3e-6 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --max_prompt_length 512 \
  --bf16 \
  --gradient_checkpointing
```

如果 A100 显存充足，可试：

```text
--max_length 1536
```

## OOM 处理

优先顺序：

```text
1. 确认 --per_device_train_batch_size 1
2. 确认 --gradient_checkpointing
3. 按 1536 -> 1024 -> 768 -> 512 降低 --max_length
4. 不优先降低 gradient_accumulation_steps，它通常不明显降低单步显存峰值
```

`max_length=512` 更适合 smoke test 或 OOM fallback，不建议作为正式默认。

## 检查

静态检查：

```bash
conda run -n nlpcc_task2 python -m py_compile "/home/lanxin/NLPCC Task2/DPO/train_dpo_trl.py"
```

训练完成后检查：

```bash
ls -lh "/home/lanxin/NLPCC Task2/DPO/dpo_adapter"
```

应能看到：

```text
adapter_config.json
adapter_model.safetensors
tokenizer_config.json
```

## DPO 验证集候选生成

DPO 训练完成后，使用 `generate_dpo_dev_candidates.py` 对验证集生成候选文件。这里每条验证样本生成 5 条候选回复，prompt 与 DPO 训练时的 `prompt` 保持一致，输出文件用于后续 `gpt-5.1` rerank。

Smoke test，只跑 3 条验证样本，但每条仍生成 5 条候选：

```bash
conda activate nlpcc_task2

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 python "/home/lanxin/NLPCC Task2/DPO/generate_dpo_dev_candidates.py" \
  --base_model "/home/lanxin/NLPCC Task2/Qlora_Sft_Model/Meta-Llama-3.1-8B-Instruct" \
  --adapter_path "/home/lanxin/NLPCC Task2/DPO/dpo_adapter" \
  --input_file "/home/lanxin/NLPCC Task2/Data/dev.jsonl" \
  --output_file "/home/lanxin/NLPCC Task2/DPO/dev_candidates_dpo_adapter_smoke/predictions.jsonl" \
  --num_candidates 5 \
  --temperature 0.8 \
  --top_p 0.9 \
  --repetition_penalty 1.05 \
  --max_new_tokens 160 \
  --limit 3
```

正式生成完整验证集候选：

```bash
conda activate nlpcc_task2

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=7 python "/home/lanxin/NLPCC Task2/DPO/generate_dpo_dev_candidates.py" \
  --base_model "/home/lanxin/NLPCC Task2/Qlora_Sft_Model/Meta-Llama-3.1-8B-Instruct" \
  --adapter_path "/home/lanxin/NLPCC Task2/DPO/dpo_adapter" \
  --input_file "/home/lanxin/NLPCC Task2/Data/dev.jsonl" \
  --output_file "/home/lanxin/NLPCC Task2/DPO/dev_candidates_dpo_adapter/predictions.jsonl" \
  --num_candidates 5 \
  --temperature 0.8 \
  --top_p 0.9 \
  --repetition_penalty 1.05 \
  --max_new_tokens 160
```

生成后应得到：

```text
/home/lanxin/NLPCC Task2/DPO/dev_candidates_dpo_adapter/predictions.jsonl
/home/lanxin/NLPCC Task2/DPO/dev_candidates_dpo_adapter/run_config.json
```

后续使用 `gpt-5.1` 对候选文件 rerank：

```bash
conda activate nlpcc_task2

PYTHONUNBUFFERED=1 python "/home/lanxin/NLPCC Task2/Rerank/qwen_rerank.py" \
  --input_file "/home/lanxin/NLPCC Task2/DPO/dev_candidates_dpo_adapter/predictions.jsonl" \
  --output_dir "/home/lanxin/NLPCC Task2/Rerank/gpt51_dev_rerank_outputs" \
  --model "gpt-5.1" \
  --resume
```
