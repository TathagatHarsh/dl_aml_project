"""Run after training to audit the delivered experiment and saved inference."""
import json
import unittest
import numpy as np
import pandas as pd
import joblib
from phase2.pipeline import ROOT, load_data
from phase2.predict import predict_file

@unittest.skipUnless((ROOT/'reports/manifest.json').exists(), 'Full experiment not yet run')
class ArtifactTests(unittest.TestCase):
    def test_split_and_scaler_fit(self):
        m=json.loads((ROOT/'reports/manifest.json').read_text())
        self.assertEqual(len(m['train_engine_ids']),80)
        self.assertEqual(len(m['validation_engine_ids']),20)
        self.assertFalse(set(m['train_engine_ids']) & set(m['validation_engine_ids']))
        df=load_data(ROOT/'data/train_FD001.txt')
        tr=df[df.engine_id.isin(m['train_engine_ids'])]
        prep=joblib.load(ROOT/'models/preprocessing.joblib')
        np.testing.assert_allclose(prep['sensor_scaler'].mean_,tr[prep['sensors']].mean())

    def test_endpoint_metrics_and_checkpoint(self):
        p=pd.read_csv(ROOT/'reports/engine_predictions.csv')
        sizes=p.groupby(['model','seed']).size()
        m=json.loads((ROOT/'reports/manifest.json').read_text())
        self.assertEqual(len(sizes),1+4*len(m['seeds']))
        self.assertTrue((sizes==100).all())
        self.assertFalse(p.duplicated(['model','seed','engine_id']).any())
        expected=p[(p.model=='LSTM30') & (p.seed==42)].sort_values('engine_id')
        actual=predict_file(ROOT/'data/test_FD001.txt',ROOT/'models/LSTM30_42.pt',ROOT/'reports/inference.csv')
        np.testing.assert_allclose(actual.predicted_rul,expected.predicted_rul,atol=1e-4)
        scores=pd.read_csv(ROOT/'reports/test_metrics.csv')
        row=scores[(scores.model=='LSTM30')&(scores.seed==42)&(scores.target=='capped')].iloc[0]
        self.assertAlmostEqual(row.rmse,np.sqrt(np.mean((expected.predicted_rul-expected.capped_rul)**2)),places=5)

if __name__=='__main__': unittest.main()
