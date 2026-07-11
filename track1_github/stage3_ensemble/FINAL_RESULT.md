# Final SVDA Task 1 result on dev

## Best (3-model logit ensemble, uniform weights, T=0.5)

| metric | value |
|---|---:|
| dev_accuracy | 0.9591 (493/514) |
| dev_macro_precision | 0.9499 |
| dev_macro_recall | 0.9534 |
| dev_macro_f1 | 0.9506 |

## Single-model and pair-wise ensemble comparisons

| Config | acc | macro_p | macro_r | macro_f1 |
|---|---:|---:|---:|---:|
| expF_new baseline (prior best) | 0.9455 | 0.9353 | 0.9383 | 0.9353 |
| expI Run #2 (CE + R-Drop + FGM + EMA) | 0.9533 | 0.9420 | 0.9477 | 0.9441 |
| expJ (+ Focal γ=2 + Sibling Margin 1.5 + Class-Balanced^0.5, adj_2) | 0.9514 | 0.9439 | 0.9436 | 0.9432 |
| expK (+ Focal γ=3 + Sibling Margin 1.5 + Class-Balanced^1.0, adj_1, seed 13) | 0.9553 | 0.9481 | 0.9436 | 0.9445 |
| I+J ensemble (T=any) | 0.9591 | 0.9481 | 0.9533 | 0.9497 |
| I+K ensemble (T=1.0) | 0.9572 | 0.9461 | 0.9517 | 0.9480 |
| J+K ensemble (T=any) | 0.9553 | 0.9469 | 0.9466 | 0.9457 |
| **I+J+K uniform (T=0.5)** | **0.9591** | **0.9499** | **0.9534** | **0.9506** |
| I(1.5)+J(1)+K(1.5) weighted (T=0.5) | 0.9591 | 0.9499 | 0.9534 | 0.9506 |

## Notes

- Δ vs baseline expF_new: **+1.36 pp acc, +1.46 pp macro_p, +1.51 pp macro_r, +1.53 pp macro_f1.**
- Dev errors decrease from 28/514 (expF_new baseline) to **21/514** (final ensemble).
- The 3-model ensemble required two genuinely-different recipes (J and K both use focal + sibling margin) to gain diversity beyond seed variation.
- Pure parameter-averaging (Model Soup) did NOT help: greedy soup degenerates to picking the single best model only. Diversity in loss landscape between models is too high for averaging to find a common minimum.
- Temperature has nearly zero effect (logits are very confident); included for completeness.
