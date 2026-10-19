import torch
import numpy as np
from torch.utils.data import DataLoader
from data_provider.data_loader import ETTDataset
from models.Autoformer import Autoformer
from models.PatchTST import PatchTST
from models.iTransformer import iTransformer
from models.FRP_PatchTST import FRPPatchTST
import warnings
warnings.filterwarnings('ignore')

def evaluate_domain(model, dataset_path, lookback=96, pred_len=96, variates=862, seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_ds = ETTDataset(dataset_path, lookback_len=lookback, pred_len=pred_len, split='train')
    test_ds = ETTDataset(dataset_path, lookback_len=lookback, pred_len=pred_len, split='test', scaler=train_ds.scaler)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    model.train()
    for epoch in range(3):
        for seq_x, seq_y in train_loader:
            optimizer.zero_grad()
            
            # Pass the full sequence directly to the model (no slicing/subsampling)
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
    dataset_path = 'data/traffic.csv'
    variates = 862 # Full Traffic dataset features[cite: 4]
    
    # Initialize all baseline and FRP control models
    models = {
        'Autoformer': Autoformer(num_variates=variates, lookback_len=96, pred_len=96),
        'PatchTST (P=16)': PatchTST(num_variates=variates, lookback_len=96, pred_len=96, patch_len=16),
        'iTransformer': iTransformer(num_variates=variates, lookback_len=96, pred_len=96),
        'FRP (Heuristic)': FRPPatchTST(num_variates=variates, lookback_len=96, pred_len=96, router_type='heuristic'),
        'FRP (Gumbel)': FRPPatchTST(num_variates=variates, lookback_len=96, pred_len=96, router_type='gumbel')
    }
    
    print("Running Week 6 Traffic Domain Experiment (Full 862 Variates)...")
    for name, model in models.items():
        try:
            mse, mae = evaluate_domain(model, dataset_path, variates=variates)
            print(f"{name:25} | MSE: {mse:.3f} | MAE: {mae:.3f}")
        except Exception as e:
            print(f"{name:25} | Evaluation Failed: {e}")