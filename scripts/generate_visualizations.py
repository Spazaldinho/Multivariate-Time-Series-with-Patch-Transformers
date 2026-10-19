import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from data_provider.data_loader import ETTDataset
from models.PatchTST import PatchTST
import warnings
warnings.filterwarnings('ignore')

def generate_plot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data (ETTh1, Lookback 96, Predict 96)
    dataset_path = 'data/ETTh1.csv'
    train_ds = ETTDataset(dataset_path, lookback_len=96, pred_len=96, split='train')
    test_ds = ETTDataset(dataset_path, lookback_len=96, pred_len=96, split='test', scaler=train_ds.scaler)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # 2. Initialize and quickly train PatchTST
    torch.manual_seed(42)
    model = PatchTST(num_variates=7, lookback_len=96, pred_len=96, patch_len=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    print("Training model for visualization...")
    model.train()
    for epoch in range(3): 
        for seq_x, seq_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(seq_x.to(device)), seq_y.to(device))
            loss.backward()
            optimizer.step()
            
    # 3. Generate Prediction for a single test sequence
    model.eval()
    with torch.no_grad():
        # Get first batch of test data
        seq_x, seq_y = next(iter(test_loader))
        seq_x_dev = seq_x.to(device)
        pred_y = model(seq_x_dev).cpu()
    
    # Extract the first sample in the batch, and select the last variate channel (e.g., Oil Temperature)
    sample_idx = 0
    channel_idx = -1 
    
    history = seq_x[sample_idx, :, channel_idx].numpy()
    true_future = seq_y[sample_idx, :, channel_idx].numpy()
    pred_future = pred_y[sample_idx, :, channel_idx].numpy()
    
    # 4. Plotting
    plt.figure(figsize=(10, 4))
    
    # Create x-axis time steps
    time_history = np.arange(len(history))
    time_future = np.arange(len(history), len(history) + len(true_future))
    
    plt.plot(time_history, history, label="Look-back History (96 steps)", color="black", alpha=0.7)
    plt.plot(time_future, true_future, label="Ground Truth", color="blue", linewidth=2)
    plt.plot(time_future, pred_future, label="PatchTST Prediction", color="orange", linestyle="--", linewidth=2)
    
    plt.axvline(x=len(history), color="gray", linestyle=":", label="Forecast Horizon Start")
    plt.title("Qualitative Analysis: PatchTST 96-Step Forecast on ETTh1")
    plt.xlabel("Time Steps (Hours)")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("prediction_plot.png", dpi=300)
    print("Saved visualization to prediction_plot.png")

if __name__ == "__main__":
    # Force CPU fallback if running locally on Mac
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    generate_plot()