"""Build notebook and LaTeX report directly from measured experiment outputs."""
from pathlib import Path
import json
import re
import pandas as pd
import nbformat as nbf
from phase2.pipeline import ROOT


def build():
    r=ROOT/'reports'; out=ROOT/'submission'; out.mkdir(exist_ok=True)
    metrics=pd.read_csv(r/'test_metrics.csv')
    val=pd.read_csv(r/'validation_metrics.csv')
    stages=pd.read_csv(r/'stage_metrics.csv')
    manifest=json.loads((r/'manifest.json').read_text())
    summary=metrics.groupby(['model','target'])[['rmse','mae']].mean().round(3).reset_index()
    table=summary.to_string(index=False)
    stage_means=stages[stages.target=='uncapped'].groupby(['model','stage'])[['rmse','mae']].mean()
    stage_text='\n\n'.join(f'{model}: '+ '; '.join(f'{stage}: RMSE {stage_means.loc[(model,stage),"rmse"]:.2f}, MAE {stage_means.loc[(model,stage),"mae"]:.2f}' for stage in ['late_0_30','middle_31_90','early_over_90'])+' cycles.' for model in ['RandomForest','LSTM30','Hybrid30'])
    capped=summary[summary.target=='capped'].set_index('model')
    ablation_text=f'Against LSTM30, the one-cycle ablation changes mean capped test RMSE by {capped.loc["LSTM1","rmse"]-capped.loc["LSTM30","rmse"]:+.3f} cycles; removing dropout changes it by {capped.loc["LSTM30_no_dropout","rmse"]-capped.loc["LSTM30","rmse"]:+.3f} cycles. Positive differences are worse. These are descriptive results from the pre-specified comparisons, not a reason to retune against the test set.'
    ci=manifest['hybrid_minus_lstm_ensemble_rmse_bootstrap_95ci']
    conclusion=('The paired interval is entirely below zero, supporting a hybrid improvement on these test engines.' if ci[1]<0 else
                'The paired interval is entirely above zero: the hybrid is worse on these test engines.' if ci[0]>0 else
                'The paired interval includes zero; a reliable hybrid advantage is not established.')
    report=f'''# Predictive Maintenance via Sensor Features and Sequence Models
## Project 7 - Phase Two report

### Abstract
We estimate remaining useful life (RUL) on NASA C-MAPSS FD001 using engineered-feature Random Forest regression, a 30-cycle LSTM, and a joint feature-sequence hybrid. All models use the same 80 training engines, 20 validation engines and 100 official test engines. We retain the Phase One 125-cycle target cap and report both capped and uncapped errors. Three random initializations quantify neural training variability. Validation selects **{manifest['selected_by_validation']}**. {conclusion} This is an academic benchmark, not an operational maintenance policy.

### 1. Motivation and literature
Phase One uses fixed rolling statistics. Such summaries can discard temporal order and interactions, motivating a sequence encoder. NASA's run-to-failure simulator supplies multivariate trajectories with known terminal labels [1]. Hochreiter and Schmidhuber introduced gated memory cells to support learning across time [2]. Our LSTM uses this gating mechanism within a finite window; it does not claim whole-lifetime memory. Bai, Kolter and Koltun show temporal convolutions are a credible alternative across sequence tasks [3]; their general benchmark does not establish superiority on FD001. A temporal CNN is a future comparison, not an implemented result.

### 2. Dataset, target and split
FD001 contains 100 run-to-failure training engines and 100 truncated test engines, with 21 sensors and 3 settings. Training RUL is max cycle within engine minus current cycle. Only training trajectories use that formula; official test endpoint labels come from RUL_FD001.txt. Targets are min(RUL,125), explicitly inherited from Phase One.

Engine IDs are shuffled using NumPy seed 2026, with 80/20 train/validation allocation persisted in the manifest. Official test IDs inhabit a separate dataset namespace. One pseudo-truncated endpoint per validation engine is selected with a fixed RNG independently of model predictions. All training cycles are used. The final benchmark intentionally retains the 80-engine fitted models so that every model has exactly the same training information; it is not directly comparable to the Phase One model trained on all 100 engines.

The loader rejects missing/nonfinite values, invalid IDs, duplicate cycles and gaps. Flat sensors are detected only on the 80 training engines (standard deviation <=1e-8). StandardScaler is fitted only there. FD001 has one operating regime, so per-regime normalization reduces to global training-only normalization and a constant regime feature is omitted. Sensor inspection is exported. The single-regime assumption must not be extended blindly to FD002/FD004.

### 3. Features and architecture
The regression baseline receives cycle age, current sensor readings, trailing means, standard deviations and least-squares slopes over 5 and 10 cycles, and each sensor's deviation from its first observation. All features are causal and reset at engine boundaries. Initial-value deviations can be sensitive to a noisy first measurement.

The LSTM input is a 30 x S normalized sensor matrix. Histories shorter than 30 repeat the first available row on the left. A single 32-unit LSTM supplies its final hidden state to Linear(32,32), ReLU, Dropout(0.2), Linear(32,1), Sigmoid; predictions are multiplied by 125. The hybrid adds a Linear(F,16), ReLU branch on standardized engineered features, concatenated with the LSTM state before the same regression head (input width 48). This is joint feature fusion rather than an arbitrary average of model predictions. It tests whether explicit age and degradation summaries complement learned sequence features.

For LSTM gates i, f, o and candidate g, cell state c_t=f_t*c_(t-1)+i_t*g_t and hidden state h_t=o_t*tanh(c_t). Sigmoid gates determine information retention. The loss is mean squared error on y/125. The bounded output supports the capped task but cannot represent RUL above 125; uncapped early-life error therefore has a structural floor.

### 4. Optimization and regularization
AdamW uses learning rate 0.001 and weight decay 0.0001, batch size 256 and gradient norm clipping at 1. Maximum epochs: {manifest['epochs_max']}; patience: {manifest['patience']}. The best validation RMSE checkpoint is restored. Seeds {manifest['seeds']} are used for each DL configuration, with deterministic CPU operations and 4 Torch threads. Architecture selection uses mean validation RMSE across seeds, not official test results. The Random Forest uses 200 trees, leaf size 3, max-features 0.8 and seed 42. It is a fresh fair-split AML comparator rather than reuse of the Phase One all-engine checkpoint.

### 5. Ablation and evaluation protocol
LSTM1 removes temporal context while retaining the recurrent architecture. LSTM30_no_dropout changes only head dropout to zero. Hybrid30 adds the engineered branch to LSTM30; parameter count also changes, so this is not a parameter-matched proof that features alone cause a difference. Other regularizers remain enabled in the no-dropout ablation. Each model predicts exactly the last observed cycle of each official test engine.

RMSE=sqrt(mean((prediction-truth)^2)); MAE=mean(abs(prediction-truth)). NASA score sums exp(-error/13)-1 for negative errors and exp(error/10)-1 otherwise. Positive error overestimates remaining life. Stage bands use uncapped truth: late 0-30, middle 31-90, early >90 cycles. These are analysis definitions, not certified maintenance thresholds.

### 6. Results
Mean test metrics across seeds (cycles; Random Forest has one run):

```
{table}
```

Per-seed metrics and standard deviations are in reports/test_metrics.csv and reports/summary.csv. Learning curves record both training and validation RMSE; plots show the first seed for clarity. Engine-level predictions and stage metrics are exported, not inferred from plots.

For seed-averaged predictions, a paired 2,000-resample engine bootstrap estimates the Hybrid30-minus-LSTM30 capped RMSE difference with a 95% percentile interval [{ci[0]:.3f}, {ci[1]:.3f}] cycles. Negative means the hybrid is better. {conclusion} This ensemble comparison differs from mean per-seed RMSE and is labeled accordingly. The interval conditions on the fitted training split; it does not include split-selection uncertainty.

{ablation_text}

Uncapped stage errors (mean across seeds):

{stage_text}

### 7. Limitations and conclusions
Sequence modeling addresses the information discarded by rolling summaries, but predictive gains must be read from the measured comparison. The optional hybrid requirement is satisfied by implementing and testing a technically motivated combination; additional marks are at the evaluator's discretion, and a gain is not assumed. Only one engine split and a small validation set are used. Overlapping training windows are correlated and long-lived engines contribute more samples. Three seeds quantify initialization variability only. No broad hyperparameter search, external dataset validation, uncertainty calibration or real-engine trial is claimed. The test set must not be reused for subsequent tuning. FD001's simulated single-condition setting restricts generalization.

### 8. Reproducibility and contribution record
Run python -m phase2.pipeline, then python -m phase2.build_submission. The manifest stores engine IDs, endpoints, target cap, file SHA256 values, runtime versions and seeds. Saved models and preprocessing enable label-free inference via phase2.predict. Unit tests cover causal features, within-engine padding, targets, malformed inputs, gradients and checkpoint reloads. The notebook exposes the experiment, tables and figures. Proposed member responsibilities and viva explanations are in VIVA.md; actual Phase Two contributions must be confirmed by the team and should not be fabricated.

### References
[1] A. Saxena and K. Goebel, Turbofan Engine Degradation Simulation Data Set, NASA Ames Prognostics Data Repository, 2008. https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

[2] S. Hochreiter and J. Schmidhuber, Long Short-Term Memory, Neural Computation, 9(8), 1735-1780, 1997. https://doi.org/10.1162/neco.1997.9.8.1735

[3] S. Bai, J. Z. Kolter and V. Koltun, An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling, 2018. https://arxiv.org/abs/1803.01271
'''
    (out/'report.md').write_text(report)
    def esc(s):
        for a,b in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]: s=s.replace(a,b)
        return s.replace('**','').replace('^',r'\textasciicircum{}').replace('<=',r'$\leq$').replace('>',r'$>$')
    tex=[r'\documentclass[conference]{IEEEtran}',r'\usepackage[T1]{fontenc}',r'\usepackage{graphicx}',r'\usepackage{url}',r'\begin{document}',r'\title{Predictive Maintenance via Sensor Features and Sequence Models}',r'\author{\IEEEauthorblockN{MS Thejas, Deepak Pandey, Tathagat Harsh, Aryan, Garv Verma}\IEEEauthorblockA{CSAI Project 7 -- Phase Two}}',r'\maketitle']
    in_code=False
    for para in re.sub(r'^(#{1,3} .+)\n', r'\1\n\n', report, flags=re.M).split('\n\n'):
        para=para.strip()
        if not para or para.startswith(('# ', '## ')): continue
        if para.startswith('### Abstract'):
            tex.append(r'\begin{abstract}'); continue
        if para.startswith('### 1.'):
            tex.append(r'\end{abstract}')
        if para.startswith('### '):
            title=para.split('\n')[0][4:]
            if title[:1].isdigit(): title=title.split('. ',1)[1]
            tex.append(r'\section{'+esc(title)+'}')
        elif para.startswith('```'):
            tex.extend([r'\begin{table}[ht]',r'\centering\caption{Mean test errors in cycles. C: capped; U: uncapped.}',r'\begin{tabular}{lcrr}\hline Model & Target & RMSE & MAE\\\hline'])
            for row in summary.itertuples():
                tex.append(f'{esc(row.model)} & {row.target[0].upper()} & {row.rmse:.3f} & {row.mae:.3f} '+r'\\')
            tex.extend([r'\hline\end{tabular}\end{table}'])
        else:
            chunks=re.split(r'(https?://\S+)', para)
            tex.append(''.join(r'\url{'+c+'}' if c.startswith('http') else esc(c) for c in chunks))
    tex.extend([r'\begin{figure}[ht]\centering\includegraphics[width=\columnwidth]{../reports/comparison.png}\caption{Capped test RMSE; error bars show standard deviation across initializations, not a confidence interval.}\end{figure}',r'\end{document}'])
    (out/'report.tex').write_text('\n\n'.join(tex))
    md=nbf.v4.new_markdown_cell; code=nbf.v4.new_code_cell
    cells=[md('# Project 7: Predictive Maintenance via Sensor Features and Sequence Models\nPhase Two: engineered regression, LSTM and hybrid feature fusion.\n\nRun cells in order. Saved results are loaded by default; set RETRAIN=True for the full experiment.'),
           code("from pathlib import Path\nimport sys\nroot = Path.cwd()\nif not (root / 'phase2').exists():\n    root = root.parent if root.name == 'phase2' else root.parents[1]\nassert (root / 'phase2/pipeline.py').exists()\nsys.path.insert(0, str(root))\nfrom phase2.pipeline import *\nfrom IPython.display import display, Image\nRETRAIN = False\nif RETRAIN:\n    run()\nr = ROOT / 'reports'"),
           md('## Data quality and target definition\nRUL is failure cycle minus current cycle for training engines only. The inherited cap is 125. Test endpoint labels are supplied separately. FD001 uses one regime.'),
           code("train = load_data(ROOT / 'data/train_FD001.txt')\ndisplay(train.describe())\nprint('Training engines:', train.engine_id.nunique(), 'Rows:', len(train))\ndisplay(pd.read_csv(r / 'sensor_inspection.csv'))"),
           md('## Engine split and leakage controls\nScalers and flat-sensor selection use only the 80 training engines. Padding and rolling features never cross engine boundaries. One fixed validation endpoint per engine mimics truncated evaluation.'),
           code("manifest = json.loads((r / 'manifest.json').read_text())\nassert set(manifest['train_engine_ids']).isdisjoint(manifest['validation_engine_ids'])\ndisplay(manifest)"),
           md('## Architecture and training\n30 × S sensors → LSTM(32) → regression head. Hybrid adds engineered features → Linear(16) → concatenation → regression head. MSE on RUL/125; AdamW, dropout, clipping and early stopping. LSTM1 and no-dropout runs isolate temporal history and head regularization.'),
           code("prep = joblib.load(ROOT / 'models/preprocessing.joblib')\nfor f in [0, len(prep['feature_columns'])]:\n    net = RULNet(len(prep['sensors']), f)\n    print(net)\n    print('Parameters:', sum(p.numel() for p in net.parameters()))\ndisplay(pd.read_csv(r / 'validation_metrics.csv'))"),
           md('## Fair test comparison\nSame test endpoints, cap and training engines for every model. Mean and standard deviation across three neural seeds; Random Forest has one seed. Phase One all-engine metrics are historical context, not the fair comparison.'),
           code("scores = pd.read_csv(r / 'test_metrics.csv')\ndisplay(scores.groupby(['model','target'])[['rmse','mae']].agg(['mean','std']))\ndisplay(Image(filename=str(r / 'comparison.png')))\ndisplay(Image(filename=str(r / 'diagnostics.png')))"),
           md('## Error by degradation stage\nStages are defined from uncapped truth: late ≤30, middle 31–90, early >90. Positive bias means overestimated remaining life.'),
           code("display(pd.read_csv(r / 'stage_metrics.csv').groupby(['model','target','stage'])[['rmse','mae','bias']].mean())\ndisplay(pd.read_csv(r / 'engine_predictions.csv').head(15))\ndisplay(Image(filename=str(r / 'stages_and_lifetimes.png')))"),
           md('## Hybrid evidence and limitations\n'+conclusion+'\n\nThe paired interval uses seed-ensemble predictions and resamples engines, not windows. Single split, small validation set, simulation domain, target cap and overlapping windows limit interpretation.'),
           code("print('Validation-selected model:', manifest['selected_by_validation'])\nprint('Hybrid minus LSTM ensemble RMSE 95% interval:', manifest['hybrid_minus_lstm_ensemble_rmse_bootstrap_95ci'])"),
           md('## Saved-checkpoint inference\nNo test labels are needed to predict RUL from sensor histories.'),
           code("from phase2.predict import predict_file\npredictions = predict_file(ROOT / 'data/test_FD001.txt', ROOT / 'models/LSTM30_42.pt', r / 'inference.csv')\ndisplay(predictions.head())\nassert len(predictions) == 100"),
           md('## References\n1. [NASA C-MAPSS repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)\n2. [Hochreiter & Schmidhuber, 1997](https://doi.org/10.1162/neco.1997.9.8.1735)\n3. [Bai et al., 2018](https://arxiv.org/abs/1803.01271)\n\nSee submission/report.tex for the IEEE-format source and VIVA.md for individual preparation.')]
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nbf.write(nb,ROOT/'Phase2_Engine_RUL.ipynb')
    payload={'summary':summary.to_dict('records'),'selected':manifest['selected_by_validation'],'ci':ci,'conclusion':conclusion,'stages':stage_means.reset_index().to_dict('records')}
    (out/'presentation_data.json').write_text(json.dumps(payload,indent=2))
    print(report)

if __name__=='__main__': build()
