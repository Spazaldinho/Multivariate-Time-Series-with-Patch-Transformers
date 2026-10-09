import torch
import numpy as np
from torch.utils.data import DataLoader
from data_provider.data_loader import ETTDataset
from models.PatchTST import PatchTST
from models.iTransformer import iTransformer
from models.Autoformer import Autoformer
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model_class, dataset_path, lookback=96, pred_len=96, seeds=[42, 123, 999]):
    # Dynamically detect hardware accelerator
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    mses, maes = [], []
    
    train_ds = ETTDataset(dataset_path, lookback_len=lookback, pred_len=pred_len, split='train')
    test_ds = ETTDataset(dataset_path, lookback_len=lookback, pred_len=pred_len, split='test', scaler=train_ds.scaler)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    for seed in seeds:
        torch.manual_seed(seed)
        
        # Send model to the detected device
        model = model_class(num_variates=7, lookback_len=lookback, pred_len=pred_len).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.MSELoss()
        
        # Train for 3 epochs for rapid midterm baseline generation
        model.train()
        for epoch in range(3):
            for seq_x, seq_y in train_loader:
                optimizer.zero_grad()
                # Send data tensors to the detected device
                loss = criterion(model(seq_x.to(device)), seq_y.to(device))
                loss.backward()
                optimizer.step()
                
        # Evaluate
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for seq_x, seq_y in test_loader:
                # Send data to device, then pull predictions back to CPU for numpy math
                preds.append(model(seq_x.to(device)).cpu())
                trues.append(seq_y)
        
        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        
        mses.append(np.mean((preds - trues) ** 2))
        maes.append(np.mean(np.abs(preds - trues)))
        
    return np.mean(mses), np.std(mses), np.mean(maes), np.std(maes)

if __name__ == "__main__":
    datasets = [('ETTh1', 'data/ETTh1.csv'), ('ETTm1', 'data/ETTm1.csv')]
    models = [('Autoformer', Autoformer), ('PatchTST', PatchTST), ('iTransformer', iTransformer)]
    
    print("Generating Midterm Report Table 1...")
    for data_name, path in datasets:
        print(f"\nDataset: {data_name} (96 -> 96)")
        for model_name, model_cls in models:
            mse_m, mse_s, mae_m, mae_s = evaluate_model(model_cls, path)
            print(f"{model_name:15} | MSE: {mse_m:.3f} ± {mse_s:.3f} | MAE: {mae_m:.3f} ± {mae_s:.3f}")