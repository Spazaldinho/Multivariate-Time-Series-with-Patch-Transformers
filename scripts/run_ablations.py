import torch
import numpy as np
from torch.utils.data import DataLoader
from data_provider.data_loader import ETTDataset
from models.PatchTST import PatchTST
import warnings
warnings.filterwarnings('ignore')

def evaluate_ablation(dataset_path, lookback, patch_size, pred_len=96, seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_ds = ETTDataset(dataset_path, lookback_len=lookback, pred_len=pred_len, split='train')
    test_ds = ETTDataset(dataset_path, lookback_len=lookback, pred_len=pred_len, split='test', scaler=train_ds.scaler)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    torch.manual_seed(seed)
    # Instantiate PatchTST with specific ablation parameters
    model = PatchTST(num_variates=7, lookback_len=lookback, pred_len=pred_len, patch_len=patch_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    model.train()
    for epoch in range(3):
        for seq_x, seq_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(seq_x.to(device)), seq_y.to(device))
            loss.backward()
            optimizer.step()
            
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for seq_x, seq_y in test_loader:
            preds.append(model(seq_x.to(device)).cpu())
            trues.append(seq_y)
    
    preds, trues = torch.cat(preds, dim=0).numpy(), torch.cat(trues, dim=0).numpy()
    return np.mean((preds - trues) ** 2), np.mean(np.abs(preds - trues))

if __name__ == "__main__":
    dataset_path = 'data/ETTh1.csv'
    lookbacks = [96, 336, 720]
    patch_sizes = [8, 16, 32, 64]
    
    print("Running Week 5 PatchTST Ablation Studies...")
    for L in lookbacks:
        print(f"\n--- Lookback Window: {L} ---")
        for P in patch_sizes:
            try:
                mse, mae = evaluate_ablation(dataset_path, lookback=L, patch_size=P)
                print(f"Patch Size {P:2d} | MSE: {mse:.3f} | MAE: {mae:.3f}")
            except Exception as e:
                print(f"Patch Size {P:2d} | Failed (Likely matrix dimension mismatch for L/P ratio)")