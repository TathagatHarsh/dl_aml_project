"""Reproducible FD001 experiment. Run: python -m phase2.pipeline."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import random
import time
import importlib.metadata
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CAP = 125.0
SENSORS = [f'sensor_{i}' for i in range(1, 22)]
COLUMNS = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + SENSORS


def load_data(path):
    df = pd.read_csv(path, sep=r'\s+', header=None)
    if df.shape[1] != 26 or df.empty:
        raise ValueError('Expected nonempty C-MAPSS data with 26 columns')
    df.columns = COLUMNS
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError('Nonfinite input')
    for c in ['engine_id', 'cycle']:
        if (df[c] < 1).any() or (df[c] % 1 != 0).any():
            raise ValueError('Engine IDs and cycles must be positive integers')
    df = df.sort_values(['engine_id', 'cycle']).reset_index(drop=True)
    if df.duplicated(['engine_id', 'cycle']).any():
        raise ValueError('Duplicate engine cycle')
    if not df.groupby('engine_id').cycle.diff().dropna().eq(1).all():
        raise ValueError('Nonconsecutive cycles')
    return df


def targets(df):
    return (df.groupby('engine_id').cycle.transform('max') - df.cycle).to_numpy(float)


def engineer(df, sensors):
    """Causal features only. FD001 has one regime (constant regime label omitted)."""
    parts = []
    for _, g in df.groupby('engine_id', sort=True):
        s = g[sensors]
        out = {'cycle': g.cycle.to_numpy()}
        for c in sensors:
            out[c] = s[c].to_numpy()
            out[c + '_health_delta'] = (s[c] - s[c].iloc[0]).to_numpy()
            for w in (5, 10):
                out[f'{c}_mean_{w}'] = s[c].rolling(w, min_periods=1).mean().to_numpy()
                out[f'{c}_std_{w}'] = s[c].rolling(w, min_periods=1).std(ddof=0).to_numpy()
                out[f'{c}_slope_{w}'] = s[c].rolling(w, min_periods=1).apply(slope, raw=True).to_numpy()
        parts.append(pd.DataFrame(out, index=g.index))
    return pd.concat(parts).sort_index()


def slope(a):
    if len(a) < 2:
        return 0.0
    t = np.arange(len(a), dtype=float)
    t -= t.mean()
    return float(t @ (a - a.mean()) / (t @ t))


def windows(df, values, length):
    if length < 1:
        raise ValueError('Window must be positive')
    result = np.empty((len(df), length, values.shape[1]), dtype=np.float32)
    for idx in df.groupby('engine_id', sort=True).indices.values():
        v = values[idx]
        padded = np.pad(v, ((length - 1, 0), (0, 0)), mode='edge')
        for j, row in enumerate(idx):
            result[row] = padded[j:j + length]
    return result


def validation_endpoints(df, seed=2026):
    """One reproducible pseudo-truncation per validation engine, before failure."""
    rng = np.random.default_rng(seed)
    indices = []
    for idx in df.groupby('engine_id', sort=True).indices.values():
        indices.append(idx[rng.integers(min(30, len(idx)) - 1, len(idx))])
    return np.asarray(indices)


class RULNet(nn.Module):
    def __init__(self, sensors, features=0, hidden=32, dropout=.2):
        super().__init__()
        self.encoder = nn.LSTM(sensors, hidden, batch_first=True)
        self.features = features
        self.feature_branch = nn.Sequential(nn.Linear(features, 16), nn.ReLU()) if features else None
        self.head = nn.Sequential(nn.Linear(hidden + (16 if features else 0), 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x, f):
        _, (h, _) = self.encoder(x)
        z = h[-1]
        if self.features:
            z = torch.cat([z, self.feature_branch(f)], dim=1)
        return self.head(z).squeeze(-1)


def predict(model, x, f):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i+512]), torch.from_numpy(f[i:i+512])).numpy()
                               for i in range(0, len(x), 512)]) * CAP


def fit_net(x, f, y, vx, vf, vy, *, hybrid, dropout, seed, epochs, patience):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = RULNet(x.shape[-1], f.shape[-1] if hybrid else 0, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(f),
                                    torch.tensor(y / CAP, dtype=torch.float32)),
                        batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best, state, stale, history = float('inf'), None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        for bx, bf, by in loader:
            opt.zero_grad()
            loss = ((model(bx, bf) - by) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            total += loss.item() * len(by)
        vp = predict(model, vx, vf)
        rmse = float(np.sqrt(np.mean((vp - vy) ** 2)))
        history.append({'epoch': epoch, 'train_rmse': np.sqrt(total / len(y)) * CAP, 'validation_rmse': rmse})
        if rmse < best - 1e-5:
            best, state, stale = rmse, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'  epoch {epoch}: validation RMSE {rmse:.3f}', flush=True)
        if stale >= patience:
            break
    model.load_state_dict(state)
    return model, history


def metrics(y, p):
    d = p - y
    return {'n': len(y), 'rmse': float(np.sqrt(np.mean(d ** 2))), 'mae': float(np.mean(abs(d))),
            'bias': float(d.mean()), 'nasa_score': float(np.expm1(np.where(d < 0, -d / 13, d / 10)).sum())}


def run(epochs=60, seeds=(42, 43, 44), patience=10, data_dir=None):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    data_dir = Path(data_dir) if data_dir else ROOT / 'data'
    reports, models = ROOT / 'reports', ROOT / 'models'
    reports.mkdir(exist_ok=True); models.mkdir(exist_ok=True)
    raw = load_data(data_dir / 'train_FD001.txt')
    ids = np.random.default_rng(2026).permutation(raw.engine_id.unique())
    train_ids, val_ids = sorted(ids[20:].tolist()), sorted(ids[:20].tolist())
    tr = raw[raw.engine_id.isin(train_ids)].reset_index(drop=True)
    va = raw[raw.engine_id.isin(val_ids)].reset_index(drop=True)
    assert set(train_ids).isdisjoint(val_ids)
    variability = tr[SENSORS].std()
    sensors = variability[variability > 1e-8].index.tolist()
    variability.rename('training_std').to_csv(reports / 'sensor_inspection.csv')
    sensor_scaler = StandardScaler().fit(tr[sensors])
    tf, vaf = engineer(tr, sensors), engineer(va, sensors)
    feature_scaler = StandardScaler().fit(tf)
    f, vf_all = [feature_scaler.transform(a).astype('float32') for a in (tf, vaf)]
    z, vz = [sensor_scaler.transform(a[sensors]).astype('float32') for a in (tr, va)]
    y, vy_all = [np.minimum(targets(a), CAP) for a in (tr, va)]
    vi = validation_endpoints(va)
    vy, vf = vy_all[vi], vf_all[vi]
    joblib.dump({'sensor_scaler': sensor_scaler, 'feature_scaler': feature_scaler,
                 'sensors': sensors, 'feature_columns': tf.columns.tolist(), 'cap': CAP}, models / 'preprocessing.joblib')
    manifest = {'seed_split': 2026, 'train_engine_ids': train_ids, 'validation_engine_ids': val_ids,
                'validation_endpoints': va.iloc[vi][['engine_id', 'cycle']].to_dict('records'),
                'cap': CAP, 'seeds': list(seeds), 'epochs_max': epochs, 'patience': patience,
                'regime': 'FD001 single operating condition; training-only global standardization',
                'data_source': 'https://github.com/TathagatHarsh/dl_aml_project/tree/main/data',
                'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data_dir.glob('*FD001.txt')},
                'versions': {p: importlib.metadata.version(p) for p in ['torch','numpy','pandas','scikit-learn']}}
    validation, trained, histories = [], {}, []
    start = time.monotonic()
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=.8, n_jobs=4, random_state=42)
    rf.fit(tf, y)
    joblib.dump(rf, models / 'random_forest.joblib')
    validation.append({'model': 'RandomForest', 'seed': 42, **metrics(vy, np.clip(rf.predict(vaf.iloc[vi]), 0, CAP))})
    # Each ablation changes one factor relative to LSTM30.
    specs = [('LSTM30',30,False,.2), ('Hybrid30',30,True,.2),
             ('LSTM1',1,False,.2), ('LSTM30_no_dropout',30,False,0.)]
    for name, length, hybrid, drop in specs:
        x, vx = windows(tr,z,length), windows(va,vz,length)[vi]
        for seed in seeds:
            print(f'{name}, seed {seed}', flush=True)
            model, history = fit_net(x,f,y,vx,vf,vy,hybrid=hybrid,dropout=drop,
                                     seed=seed,epochs=epochs,patience=patience)
            checkpoint = {'state_dict': model.state_dict(), 'sensors': len(sensors),
                          'features': f.shape[1] if hybrid else 0, 'hidden':32, 'dropout':drop,
                          'window':length, 'seed':seed, 'cap':CAP}
            torch.save(checkpoint, models / f'{name}_{seed}.pt')
            trained[name,seed] = (model,length)
            validation.append({'model':name,'seed':seed,**metrics(vy,predict(model,vx,vf))})
            histories.extend([{'model':name,'seed':seed,**h} for h in history])
    val = pd.DataFrame(validation)
    val.to_csv(reports / 'validation_metrics.csv',index=False)
    pd.DataFrame(histories).to_csv(reports / 'learning_curves.csv',index=False)
    selected = val.groupby('model').rmse.mean().idxmin()
    manifest['selected_by_validation'] = selected
    # Test labels are parsed only after training and model selection finish.
    test = load_data(data_dir / 'test_FD001.txt')
    truth = np.loadtxt(data_dir / 'RUL_FD001.txt').reshape(-1)
    if not np.isfinite(truth).all() or (truth < 0).any():
        raise ValueError('Test RUL must be finite and nonnegative')
    ti = test.groupby('engine_id',sort=True).tail(1).index.to_numpy()
    if len(truth) != len(ti) or not np.array_equal(test.iloc[ti].engine_id, np.arange(1,len(truth)+1)):
        raise ValueError('Test RUL labels must match consecutively numbered test engines')
    tfeat = engineer(test,sensors)
    tscaled = feature_scaler.transform(tfeat).astype('float32')
    tsensor = sensor_scaler.transform(test[sensors]).astype('float32')
    predictions = [('RandomForest',42,np.clip(rf.predict(tfeat.iloc[ti]),0,CAP))]
    for (name,seed),(model,length) in trained.items():
        predictions.append((name,seed,predict(model,windows(test,tsensor,length)[ti],tscaled[ti])))
    records, scores, stages = [], [], []
    for name,seed,p in predictions:
        for target, yy in [('capped',np.minimum(truth,CAP)),('uncapped',truth)]:
            scores.append({'model':name,'seed':seed,'target':target,**metrics(yy,p)})
            for stage,mask in [('late_0_30',truth<=30),('middle_31_90',(truth>30)&(truth<=90)),('early_over_90',truth>90)]:
                if mask.any():
                    stages.append({'model':name,'seed':seed,'target':target,'stage':stage,**metrics(yy[mask],p[mask])})
        for unit,cycle,actual,pred in zip(test.iloc[ti].engine_id,test.iloc[ti].cycle,truth,p):
            records.append({'model':name,'seed':seed,'engine_id':int(unit),'cycle':int(cycle),
                            'actual_rul':actual,'capped_rul':min(actual,CAP),'predicted_rul':float(pred)})
    scores = pd.DataFrame(scores)
    scores.to_csv(reports / 'test_metrics.csv',index=False)
    pd.DataFrame(stages).to_csv(reports / 'stage_metrics.csv',index=False)
    pd.DataFrame(records).to_csv(reports / 'engine_predictions.csv',index=False)
    summary = scores.groupby(['model','target'])[['rmse','mae']].agg(['mean','std'])
    summary.to_csv(reports / 'summary.csv')
    # Paired engine bootstrap: uncertainty in hybrid improvement, not individual windows.
    pred = pd.DataFrame(records)
    a = pred[pred.model=='LSTM30'].groupby('engine_id').predicted_rul.mean().to_numpy()
    b = pred[pred.model=='Hybrid30'].groupby('engine_id').predicted_rul.mean().to_numpy()
    yy = np.minimum(truth,CAP)
    rng = np.random.default_rng(2026)
    boot = rng.integers(0,len(yy),(2000,len(yy)))
    diffs = np.sqrt(((b[boot]-yy[boot])**2).mean(1))-np.sqrt(((a[boot]-yy[boot])**2).mean(1))
    manifest['hybrid_minus_lstm_ensemble_rmse_bootstrap_95ci'] = np.quantile(diffs,[.025,.975]).tolist()
    manifest['elapsed_seconds'] = time.monotonic()-start
    (reports / 'manifest.json').write_text(json.dumps(manifest,indent=2))
    plot_results(scores,pred,pd.DataFrame(histories),reports)
    plot_stages(pd.DataFrame(stages),raw,reports)
    print(summary.to_string(),flush=True)
    return scores


def plot_results(scores,pred,hist,reports):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig,ax=plt.subplots(figsize=(9,5))
    s=scores[scores.target=='capped'].groupby('model').rmse.agg(['mean','std'])
    ax.bar(s.index,s['mean'],yerr=s['std'].fillna(0),color='#197c91',capsize=5)
    ax.set(ylabel='Test RMSE (cycles, capped at 125)',title='Identical engine split and test endpoints')
    ax.tick_params(axis='x',rotation=15); fig.tight_layout(); fig.savefig(reports/'comparison.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in ['LSTM30','Hybrid30']:
        g=hist[(hist.model==name)&(hist.seed==hist.seed.min())]
        axes[0].plot(g.epoch,g.validation_rmse,label=name)
        p=pred[(pred.model==name)&(pred.seed==pred.seed.min())]
        axes[1].scatter(p.capped_rul,p.predicted_rul,s=16,alpha=.6,label=name)
    axes[0].set(xlabel='Epoch',ylabel='Validation RMSE (cycles)',title='Early stopping on held-out engines')
    axes[1].plot([0,125],[0,125],'k--'); axes[1].set(xlabel='True capped RUL',ylabel='Predicted RUL',title='100 test engines')
    for ax in axes: ax.legend()
    fig.tight_layout(); fig.savefig(reports/'diagnostics.png',dpi=160); plt.close(fig)


def plot_stages(stages, raw, reports):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    stages[stages.target=='uncapped'].pivot_table(index='stage', columns='model', values='mae').plot.bar(ax=axes[0])
    axes[0].set(title='Error by actual degradation stage', ylabel='Uncapped MAE (cycles)', xlabel='')
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].legend(fontsize=7)
    axes[1].hist(raw.groupby('engine_id').cycle.max(), bins=15, color='#197c91')
    axes[1].set(title='Training engine lifetimes', xlabel='Failure cycle', ylabel='Engine count')
    fig.tight_layout(); fig.savefig(reports/'stages_and_lifetimes.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--data-dir',type=Path)
    args=p.parse_args()
    if args.epochs<1 or args.patience<1: p.error('epochs and patience must be positive')
    run(args.epochs,args.seeds,args.patience,args.data_dir)
