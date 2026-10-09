import numpy as np
import pandas as pd
from data_provider.data_loader import ETTDataset

def test_data_loader_integrity():
    # Generate 1000 steps of synthetic sequential data with 7 features
    df = pd.DataFrame(np.arange(7000).reshape(1000, 7), columns=[f"OT_{i}" for i in range(7)])
    df.to_csv("data/synthetic_test.csv", index=False)

    lookback, pred_len = 96, 96
    train_ds = ETTDataset("data/synthetic_test.csv", lookback_len=lookback, pred_len=pred_len, split='train')
    test_ds = ETTDataset("data/synthetic_test.csv", lookback_len=lookback, pred_len=pred_len, split='test')

    # Verify input [L, N] and target [S, N] dimensions
    x, y = train_ds[0]
    assert x.shape == (lookback, 7)
    assert y.shape == (pred_len, 7)

    # Verify temporal continuity: the first target timestamp must immediately follow the last lookback timestamp
    x_raw = train_ds.scaler.inverse_transform(x.numpy())
    y_raw = train_ds.scaler.inverse_transform(y.numpy())
    np.testing.assert_allclose(y_raw[0] - x_raw[-1], 7.0, rtol=1e-5) # Since step difference is 1 row = 7 values

    # Verify that test data is scaled using training statistics only (not test statistics)
    train_mean = train_ds.scaler.mean
    test_mean = np.mean(test_ds.data, axis=0)
    assert not np.allclose(test_mean, 0.0, atol=1e-1), "Test set appears normalized on its own mean (temporal leakage risk)"