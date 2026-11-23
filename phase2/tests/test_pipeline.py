import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from phase2.pipeline import windows, engineer, targets, slope, RULNet, load_data, COLUMNS, metrics

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.df=pd.DataFrame({'engine_id':[1,1,1,2,2],'cycle':[1,2,3,1,2],'sensor_2':[1.,3.,5.,100.,104.]})

    def test_windows_never_cross_engines_and_pad_left(self):
        w=windows(self.df,self.df[['sensor_2']].to_numpy(),3)
        np.testing.assert_array_equal(w[3,:,0],[100,100,100])
        np.testing.assert_array_equal(w[4,:,0],[100,100,104])
        np.testing.assert_array_equal(w[2,:,0],[1,3,5])

    def test_features_are_causal_and_engine_local(self):
        a=engineer(self.df,['sensor_2'])
        changed=self.df.copy(); changed.loc[2,'sensor_2']=999
        b=engineer(changed,['sensor_2'])
        np.testing.assert_array_equal(a.iloc[:2],b.iloc[:2])
        self.assertEqual(a.loc[3,'sensor_2_health_delta'],0)
        self.assertEqual(a.loc[3,'sensor_2_slope_10'],0)
        self.assertAlmostEqual(slope(np.array([2.,5.,8.])),3.)
        np.testing.assert_array_equal(targets(self.df),[2,1,0,1,0])

    def test_checkpoint_roundtrip_and_gradients(self):
        for features in [0,4]:
            m=RULNet(2,features); x=torch.randn(3,5,2); f=torch.randn(3,4)
            m(x,f).sum().backward()
            self.assertIsNotNone(m.encoder.weight_ih_l0.grad)
            m.eval()
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'model.pt'; torch.save(m.state_dict(),p)
                n=RULNet(2,features); n.load_state_dict(torch.load(p,weights_only=True)); n.eval()
                torch.testing.assert_close(m(x,f),n(x,f))

    def test_invalid_data_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.txt'
            a=np.ones((2,len(COLUMNS))); np.savetxt(p,a)
            with self.assertRaises(ValueError): load_data(p)

    def test_metric_units(self):
        m=metrics(np.array([1.,5.]),np.array([4.,1.]))
        self.assertEqual(m['mae'],3.5)
        self.assertAlmostEqual(m['rmse'],np.sqrt(12.5))

if __name__=='__main__': unittest.main()
