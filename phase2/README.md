# Phase Two: sensor sequences and engineered degradation features

This directory adds the deep-learning work to the completed Phase One project. It is self-contained because the original local data/source files were iCloud placeholders during implementation. FD001 files were retrieved from this project's GitHub data directory; exact SHA256 hashes are recorded with each experiment.

## Measured results

Capped test RMSE / MAE in cycles: Random Forest **12.95 / 9.32**, LSTM30 **14.58 / 10.76**, Hybrid30 **14.51 / 10.43** (neural values averaged over three seeds). Validation selected Random Forest. The hybrid-minus-LSTM seed-ensemble RMSE bootstrap interval is **[-2.54, 2.00]** cycles, so a reliable hybrid improvement is not established. LSTM1 RMSE is **16.75**, supporting the value of temporal context in this experiment.

All seven checks passed, including artifact-level scaler and checkpoint checks. The notebook contains executed outputs.

## Start here

- `Phase2_Engine_RUL.ipynb`: executed predictive-maintenance notebook.
- `submission/report.pdf`: compiled IEEE-format report; `report.tex` is editable LaTeX source and `report.md` is readable without LaTeX.
- `submission/EngineMind_AI_Phase_2_Styled.pptx`: presentation styled after Phase One, including all five names and roll numbers. A matching PDF is included.
- `VIVA.md`: demonstration, theory questions and proposed member responsibilities.
- `reports/`: per-engine predictions, per-seed and stage metrics, ablations, learning curves and provenance.
- `models/`: fitted preprocessing, Random Forest and neural checkpoints.

## Reproduce from the repository root

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r phase2/requirements.txt
python -m unittest discover -s phase2/tests -v
python -m phase2.pipeline
python -m phase2.build_submission
```

Default experiment: 60 maximum epochs, patience 10, three seeds (42, 43, 44), CPU. The script overwrites Phase Two model/report outputs when rerun. It does not change Phase One artifacts. The saved notebook loads measured outputs by default; change `RETRAIN = True` to train from the notebook. Use `requirements-lock.txt` for the exact environment used in the delivered run.

```bash
python -m phase2.predict phase2/data/test_FD001.txt \
  --checkpoint phase2/models/LSTM30_42.pt \
  --output phase2/reports/inference.csv
```

Input is whitespace-delimited C-MAPSS format: engine ID, cycle, 3 settings and 21 sensors. Predictions are capped-task estimates in [0,125]; input requires complete consecutive cycle histories. Only load trusted local model files.

Compile the report using an IEEEtran-capable TeX distribution:

```bash
cd phase2/submission
pdflatex report.tex
pdflatex report.tex
```

## Experimental contract

- Fixed 80/20 split of the original 100 training engines; official test set has 100 separate engines. Training IDs and test IDs are different namespaces.
- Sensor selection, sensor scaling and engineered-feature scaling are fitted only on training engines.
- FD001 has one operating regime. A constant regime indicator is omitted; normalization is fitted within that one regime.
- Training target: min(final training cycle - current cycle, 125). Test labels are official endpoint RUL, never inferred from test maxima.
- Random Forest receives cycle, sensor level, rolling mean/std/slope and change from first observation.
- LSTM30 learns from normalized sensor windows. Hybrid30 fuses its final hidden state with engineered features.
- LSTM1 and LSTM30_no_dropout ablate temporal context and dropout. All experiments share engine partitions and target definitions.
- Early stopping uses one deterministic pseudo-truncated endpoint per validation engine. Architecture selection uses mean validation RMSE, never test RMSE.
- Capped/uncapped RMSE and MAE, signed error, NASA score and early/middle/late stage metrics are exported. Three seeds measure neural initialization variability, not dataset-split variability.
- Paired engine bootstrap compares seed-ensemble hybrid and LSTM predictions; its interval is conditional on one train/validation split.

## Student-document requirement coverage

| Evaluation item | Evidence |
|---|---|
| Architecture logic and theoretical rigor | Notebook, report sections 3-4, `RULNet` |
| Hybrid innovation | Joint engineered-feature branch; measured hybrid comparison |
| Ablation studies | One-cycle context, no dropout, sequence-only vs hybrid |
| DL literature review | NASA, LSTM paper, temporal-convolution study in report |
| Dataset and regularization | Training-only scaling, padding, dropout, AdamW, early stopping |
| Technical validation | Leakage tests, checkpoint reload test, 100 endpoint predictions |
| Repository and code quality | Runnable modules, tests, dependency lock, manifests |
| Presentation | Submission presentation |
| Individual viva | `VIVA.md`; all members should understand the complete pipeline |
| LaTeX/IEEE report | `submission/report.tex` |

The document makes hybrid work optional and extra marks dependent on demonstrated technical value. An implemented hybrid does not guarantee improved metrics or additional marks. Proposed responsibilities are not evidence of completed individual contributions; the team must record actual contributions before submission.

## Limits

This is a simulated single-condition benchmark. The original Phase One XGBoost was trained on all 100 engines and is historical context, not a fair-split comparator. No new production deployment is performed. A future multi-regime dataset requires regime-aware preprocessing and a new evaluation. Do not tune further against the already inspected test set.

The deck authoring source is `build_deck.mjs`. It uses the Codex bundled `@oai/artifact-tool` runtime; its paths are specific to the authoring machine. The exported PPTX is portable and editable without that runtime. The Python training, notebook, report generator and inference modules are independent of the slide runtime.
