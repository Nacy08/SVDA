# Experiment I: R-Drop + FGM + EMA on top of ExpF_new ranking stack

ExpI sits on the same FULCRA-warmup encoder as ExpF_new and keeps the
HybridRankingLoss + GlobalFallbackRankingLoss formulation that delivered
the current best (`adjacent_2`, `λ_hybrid=0.1`, `λ_global=0.05`,
`best_dev_accuracy=0.9455`, `best_dev_macro_precision=0.9353`).

On top of that baseline, ExpI adds the standard competition stack:

| Component | Purpose | Reference |
|---|---|---|
| Richer input (`Question + " " + Consistent Value Response`) | Context for ambiguous responses | Competition heuristic |
| R-Drop (symmetric KL between two dropout passes, λ=1.0) | Dropout-induced inconsistency regularizer | Liang et al., NeurIPS 2021 |
| FGM word-embedding perturbation (ε=1.0) | Cheap adversarial robustness | Zhu et al., SDU@AAAI-21 |
| EMA of model parameters (decay=0.999) | Smoother eval target | Standard |

## Evaluation code immutability

`expI/metrics.py` is a thin wrapper that:

1. Reads `/root/Task1_baseline1/expF_new/metrics.py` as the canonical eval file.
2. Verifies its SHA-256 equals
   `ac5f2f8fb769dd16d2f8f0f2046b3b125f727f982b90a308cc9a18e816fd4618`.
3. Re-exports `compute_classification_metrics` from that file.

If anyone edits the canonical file, training aborts before producing scores.
The shell verifier `verify_eval_hash.sh` performs the same check + ensures
all sibling `exp*/metrics.py` copies are byte-identical.

## Run a single configuration

```bash
cd /root/Task1_baseline1/expI
bash verify_eval_hash.sh
/root/miniconda3/envs/svda/bin/python train_svda_from_fulcra.py \
  --config config.yaml \
  --seed 47 \
  --warmup_ckpt /root/Task1_baseline1/expB/outputs/20260428_215642_seed42/fulcra_warmup/checkpoints/best.pt \
  --candidate_mode adjacent_2 \
  --lambda_hybrid 0.1 \
  --lambda_global 0.05 \
  --lambda_rdrop 1.0 \
  --fgm_epsilon 1.0 \
  --ema_decay 0.999 \
  --use_question 1
```

Each epoch logs both raw and EMA dev metrics. The script keeps whichever
gives the higher primary metric per epoch as the "selected" checkpoint;
the final reported numbers come from re-loading that checkpoint and
re-running `compute_classification_metrics` (zero metric-side drift).
