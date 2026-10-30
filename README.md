# LeJEPA and discriminative collapse on Tennessee Eastman

Code for the post **"When Preventing Collapse Destroys Useful Representations: The Hidden Role of Optimization in LeJEPA"** (`<link to the post>`).

The post asks why a LeJEPA encoder, whose embeddings look geometrically healthy, detects industrial faults worse than a supervised encoder whose embeddings use a single direction. This repository contains every experiment behind it. Each one is a single `ldc` command, and `scripts/reproduce.sh` runs them all in order.

## Install

```bash
pip install -e .          # Python >= 3.10, PyTorch >= 2.0
ldc --help
```

## Data

The data is the Tennessee Eastman simulation corpus of Rieth et al. (2017), available on Harvard Dataverse: [doi:10.7910/DVN/6C3JR1](https://doi.org/10.7910/DVN/6C3JR1). Download its four files:

| File | Direct link |
|---|---|
| `TEP_FaultFree_Training.RData` | https://dataverse.harvard.edu/api/access/datafile/3031241 |
| `TEP_FaultFree_Testing.RData` | https://dataverse.harvard.edu/api/access/datafile/3031240 |
| `TEP_Faulty_Training.RData` | https://dataverse.harvard.edu/api/access/datafile/3031242 |
| `TEP_Faulty_Testing.RData` | https://dataverse.harvard.edu/api/access/datafile/3031243 |

Then build the tabular dataset:

```bash
ldc convert --raw path/to/rdata --out-root data/te_rieth
```

This keeps 50 simulation runs per condition and summarizes each sensor over sliding windows of 30 samples (stride 10) by six statistics: mean, standard deviation, skewness, kurtosis, minimum and maximum. That gives 52 × 6 = 312 features per window, about 49,000 training and 84,000 test windows. Label 0 is normal and 1 is fault; about 5% of windows are normal.

## Reproduce the post

```bash
DATA=data/te_rieth GPU=0 bash scripts/reproduce.sh
```

The script runs about 30 trainings of 50 epochs, roughly 25 minutes each on an A100, so plan for half a day on one GPU. It skips finished runs, so you can rerun it to resume. With `SEEDS="42"` it skips the seed check and takes about a third of the time. At the end it writes `results/tables.md` with every table of the post, and figures 1–3.

What each section of the post runs:

| Post section | Command(s) | Output under `runs/` |
|---|---|---|
| Supervised reference | `ldc train-sup --seed S` | `sS/sup/encoder_init.pth`, `summary.json` |
| The paradox, 4. masks | `ldc train-ssl --shared-core F` (from scratch); `ldc mask-audit` | `scratch/share{0,25,50,75}/`, `results/mask_audit.json` |
| 5–6. continuation, loss terms | `ldc train-ssl --init … --alpha A --lam L` | `sS/{inv_only,both}/`, `s42/sig_only/` |
| 7. gradient analysis | `ldc grad --sigreg-target T` | `results/grad_*.json` |
| 8. anisotropic targets | `ldc train-ssl --alpha 0 --sigreg-target T` | `s42/sig_{power1,power2,linear}/` |
| 9. gradient paths | `ldc train-ssl --path {detach_h,freeze_enc,on_h}` | `sS/detach_h/`, `s42/{freeze_enc,on_h}/` |
| 9. affine control | `ldc affine --seed S` | `results/affine_sS.json` |
| 9. projector dimension | `ldc train-ssl --proj-dim K` | `s42/k{2,4,8}/` |
| all | `ldc diagnose --run R --init I`, then `ldc report` | `R/diagnostics.json`, `results/` |

`ldc diagnose` evaluates every checkpoint of a run and writes `diagnostics.json`. For each checkpoint it gives:
- the linear-probe metrics;
- the effective rank and variance of h;
- the Fisher criterion along the starting encoder's discriminant direction w;
- the probe and geometry of the projector output z, when there is a projector.

## The model in one paragraph

The encoder is the tabular transformer of T-JEPA (Thimonier et al., ICLR 2025): each feature is a token, and we use 4 layers, 2 heads and hidden size 32. A window's embedding h is the mean of its feature tokens, and a projector maps h to z with 16 dimensions. LeJEPA (Balestriero & LeCun, 2025) trains on several masked views of each window, each view keeping 15–35% of the sensors. The loss is `α · invariance(z) + λ · SIGReg(z)`, with α = 1 and λ = 0.05. `--path` changes how SIGReg reaches the encoder:

| `--path` | SIGReg acts on | Encoder trained by |
|---|---|---|
| `full` (LeJEPA) | z = g(h) | both losses |
| `detach_h` | g(detach(h)) | invariance only (SIGReg still trains g) |
| `freeze_enc` | z = g(h) | nothing (frozen) |
| `on_h` | h (no projector) | both losses |

## Things to know

- **Metrics.** All probes are class-balanced logistic regressions on standardized embeddings, fit on the training windows and scored on the test windows. Always answering "fault" would score 0.944 accuracy, so compare accuracies only with each other, and prefer ROC-AUC.
- **Effective rank** is the exponential of the entropy of the normalized singular values (`metrics.effective_rank`). It is scale-invariant, which is why `diagnose` also reports the variance. The affine control also reports the participation ratio, (Σλ)² / Σλ².
- **How close to the post's numbers.**
  - **Self-supervised training** matches the original code weight for weight, but the original runs also trained a monitoring probe every 5 epochs. That probe used up random numbers, so the post's runs had different dropout draws. Expect the same picture with different decimals.
  - **Supervised training** is the original procedure: it matches the original run to six decimals after one epoch, then drifts slightly because GPU kernels are not fully deterministic.
  - **Diagnostics** (`diagnose`, `grad`) applied to the original checkpoints reproduce the post's numbers exactly.
- **Inherited training details.** Two quirks of the original code are kept:
  - 2 × 10% of the training rows are carved out and never used;
  - rows are shuffled once, not every epoch.
  
  Gradients are clamped element-wise to ±1.
- **Seeds.** Results vary a lot with the seed, starting with the supervised encoder: the post's seeds 42, 43 and 44 give 0.870, 0.845 and 0.770 ROC-AUC.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

The tests use a small synthetic corpus, run on CPU in under a minute, and need no data.

## References

- Balestriero & LeCun (2025). *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.* [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)
- Thimonier, De Melo Costa, Popineau, Rimmel & Doan (2025). *T-JEPA: Augmentation-Free Self-Supervised Learning for Tabular Data.* ICLR 2025. [arXiv:2410.05016](https://arxiv.org/abs/2410.05016)
- Rieth, Amsel, Tran & Cook (2017). *Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation.* Harvard Dataverse. [doi:10.7910/DVN/6C3JR1](https://doi.org/10.7910/DVN/6C3JR1)
- Downs & Vogel (1993). *A Plant-Wide Industrial Process Control Problem.* Computers & Chemical Engineering 17(3).

## License

MIT. The encoder code derives from T-JEPA (MIT, © 2024 Hugo Thimonier); see `LICENSE`.
