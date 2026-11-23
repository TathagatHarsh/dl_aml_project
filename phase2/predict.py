"""Predict each engine's last observed RUL from a saved Phase Two checkpoint."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
import torch
from phase2.pipeline import ROOT, RULNet, load_data, engineer, windows, predict

def predict_file(input_path, checkpoint, output):
    prep=joblib.load(ROOT/'models/preprocessing.joblib')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=RULNet(saved['sensors'],saved['features'],saved['hidden'],saved['dropout'])
    model.load_state_dict(saved['state_dict'])
    df=load_data(input_path)
    features=engineer(df,prep['sensors'])[prep['feature_columns']]
    f=prep['feature_scaler'].transform(features).astype('float32')
    z=prep['sensor_scaler'].transform(df[prep['sensors']]).astype('float32')
    idx=df.groupby('engine_id').tail(1).index.to_numpy()
    p=predict(model,windows(df,z,saved['window'])[idx],f[idx])
    result=df.iloc[idx][['engine_id','cycle']].copy(); result['predicted_rul']=p
    result.to_csv(output,index=False)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('input',type=Path)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'models/LSTM30_42.pt')
    p.add_argument('--output',type=Path,default=ROOT/'reports/inference.csv')
    a=p.parse_args(); print(predict_file(a.input,a.checkpoint,a.output).to_string(index=False))
