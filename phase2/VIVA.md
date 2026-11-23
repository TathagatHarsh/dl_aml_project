# Phase Two viva and demonstration guide

## Demonstration (8-10 minutes)
1. Open the executed notebook and explain the engine-level split and target cap.
2. Inspect `reports/manifest.json`: training/validation engine IDs, data hashes, seeds, versions.
3. Show causal window padding and training-only standardization in `pipeline.py`.
4. Explain the LSTM and hybrid branches; compare measured ablations.
5. Compare capped and uncapped metrics and early/late stages; discuss overestimation near failure.
6. Run inference from a saved checkpoint without training or accessing labels.

## Questions every member should be able to answer

**Why split engines rather than rows?** Adjacent cycles share engine-specific wear and noise. A row split would let the model recognize a validation engine from training.

**How are labels created?** Training RUL is the engine's failure cycle minus the current cycle. Official test RUL is supplied separately for each last observed cycle. The last test cycle is not the failure cycle.

**Why cap at 125?** Phase One chose a piecewise target to reduce the influence of early-life variation. This changes the task and restricts predictions, so uncapped metrics are also reported. It is an assumption, not a NASA-mandated target.

**What does the LSTM learn?** Input/forget/output gates control a persistent cell state. A 32-unit encoder summarizes each 30-cycle sensor history; a regression head predicts scaled RUL. It cannot see history before the window.

**Why the hybrid?** A learned temporal representation may complement explicit cycle age, rolling slopes, variability and change from initial sensor values. Concatenation lets a joint supervised loss learn the combination. Superiority must be demonstrated, not assumed.

**What do ablations isolate?** LSTM1 removes temporal history while retaining the encoder architecture. Removing dropout tests that regularizer. Comparing Hybrid30 to LSTM30 tests adding engineered features, but also increases parameter count; it is not a parameter-matched causal test.

**How is leakage prevented?** Engine-disjoint train/validation partitions, fit-only-on-training scalers and sensor inspection, causal features, within-engine windows, and validation-only checkpoint/model selection. No test-based tuning.

**Why left padding?** Short histories repeat their earliest observation so all engines remain evaluable. This invents a constant prehistory and may weaken very-short-history performance; it never copies another engine.

**What is FD001's operating regime?** A single operating condition. The regime indicator is constant and provides no predictive information. Global training-only scaling is therefore single-regime scaling. Multi-regime FD002/FD004 require a new regime-aware design and evaluation.

**What is early stopping?** Stop after ten epochs without improved RMSE on one fixed pseudo-truncated endpoint from each validation engine; restore the best weights. One endpoint per engine avoids overweighting long trajectories, but only 20 validation engines gives noisy selection.

**Why RMSE and MAE?** MAE gives typical absolute cycle error; RMSE emphasizes large errors. NASA score penalizes late/optimistic estimates more severely. Stage grouping uses actual uncapped RUL.

**What does a confidence interval mean here?** The paired bootstrap resamples whole test engines and compares seed-ensemble RMSE. It describes uncertainty across these test engines conditional on this training split; it is not uncertainty across retraining datasets or proof of real-world safety.

**What is regularization?** Dropout 0.2 in the head, AdamW weight decay 1e-4, early stopping and a small hidden size. Gradient clipping at norm 1 stabilizes optimization rather than proving generalization.

**Can we claim industrial readiness?** No. FD001 is simulated and single-condition. Domain shift, sensor outages, calibration, uncertainty, cost of errors and prospective real-world evaluation remain unresolved.

## Proposed presentation ownership (confirm actual contributions)
These are suggested Phase Two responsibilities based on the Phase One roster, not a claim that students already performed this work.

| Member | Proposed topic | Evidence to understand |
|---|---|---|
| MS Thejas | Data and preprocessing | Data validation, scaling, causal features, windows |
| Deepak Pandey | Baseline and ablations | Random Forest, LSTM1, dropout comparison |
| Tathagat Harsh | DL architecture and optimization | LSTM gates, hybrid fusion, loss, checkpoints |
| Aryan | Evaluation | Engine-level metrics, stages, bootstrap, limitations |
| Garv Verma | Reproduction and demonstration | Notebook, checkpoint inference, presentation |

Record actual work and commit references before submission. All five members should understand the full pipeline.
