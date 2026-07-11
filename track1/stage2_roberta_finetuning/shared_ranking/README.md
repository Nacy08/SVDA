# Experiment F_new: Hybrid Dynamic Confusable Ranking + Global Fallback

`expF_new` is an independent experiment copied from `expF`. It keeps the ExpB/ExpF SVDA fine-tune backbone unchanged:

- RoBERTa-large encoder loaded locally from `/root/autodl-tmp/FacebookAI/roberta-large`
- SVDA classifier head `1024 -> 512 -> 19`
- SVDA input field: `Consistent Value Response`
- FULCRA warm-up is not rerun; pass the existing ExpB warm-up checkpoint to `--warmup_ckpt`

Outputs are written to:

```bash
/root/autodl-tmp/Task1_baseline1/expF_new/outputs
```

## Loss

Stage 2 fine-tuning uses:

```text
total_loss = CE_19class
           + lambda_hybrid * HybridRankingLoss
           + lambda_global * GlobalFallbackRankingLoss
```

`HybridRankingLoss` selects the current highest-logit hard negative from a small candidate set for each gold label. `GlobalFallbackRankingLoss` selects the highest-logit non-gold label globally with a smaller weight.

Default config:

```yaml
candidate_mode: adjacent_1
lambda_hybrid: 0.1
lambda_global: 0.02
margin: 1.0
top_k_hybrid: 1
top_k_global: 1
start_epoch: 2
```

## Run One Configuration

```bash
cd /root/Task1_baseline1/expF_new
python train_svda_from_fulcra.py \
  --config config.yaml \
  --seed 42 \
  --warmup_ckpt /root/Task1_baseline1/expB/outputs/20260428_215642_seed42/fulcra_warmup/checkpoints/best.pt \
  --candidate_mode adjacent_1 \
  --lambda_hybrid 0.1 \
  --lambda_global 0.02
```

Use `--lambda_hybrid 0.0 --lambda_global 0.0` for the CE-only baseline.

## Run Grid

```bash
cd /root/Task1_baseline1/expF_new
python run_lambda_grid.py \
  --config config.yaml \
  --seed 42 \
  --warmup_ckpt /root/Task1_baseline1/expB/outputs/20260428_215642_seed42/fulcra_warmup/checkpoints/best.pt
```

The grid is read from `hybrid_ranking.candidate_mode_grid`, `lambda_hybrid_grid`, and `lambda_global_grid`.

## Aggregate Results

```bash
cd /root/Task1_baseline1/expF_new
python aggregate_results.py
```

This refreshes `/root/autodl-tmp/Task1_baseline1/expF_new/outputs/summary.csv`.
