import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from data_provider.data_loader import ETTDataset
from models.PatchTST import PatchTST
from models.iTransformer import iTransformer
from models.Autoformer import Autoformer

def train_and_eval(model, train_loader, test_loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Midterm hyperparameter[cite: 87]
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        for seq_x, seq_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(seq_x), seq_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    mse_total, mae_total = [], []
    with torch.no_grad():
        for seq_x, seq_y in test_loader:
            pred = model(seq_x)
            mse_total.append(torch.mean((pred - seq_y)**2).item())
            mae_total.append(torch.mean(torch.abs(pred - seq_y)).item())
            
    return np.mean(mse_total), np.mean(mae_total)

if __name__ == "__main__":
    seeds = [42, 123, 999, 2026, 473] # 5 seeds for statistical reliability[cite: 87]
    dataset = ETTDataset('data/ETTh1.csv', lookback_len=96, pred_len=96)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(ETTDataset('data/ETTh1.csv', split='test'), batch_size=32)
    
    metrics = []
    for seed in seeds:
        torch.manual_seed(seed)
        # Swap model here to test baselines
        model = iTransformer(num_variates=7, lookback_len=96, pred_len=96)
        mse, mae = train_and_eval(model, train_loader, test_loader)
        metrics.append((mse, mae))
        
    mses, maes = zip(*metrics)
    print(f"iTransformer ETTh1 (96->96) | MSE: {np.mean(mses):.4f} ± {np.std(mses):.4f} | MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")