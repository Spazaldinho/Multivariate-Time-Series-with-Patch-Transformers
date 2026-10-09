import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class StandardScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std[self.std == 0.0] = 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class ETTDataset(Dataset):
    # ADDED: scaler=None to the parameters
    def __init__(self, csv_path, lookback_len=96, pred_len=96, split='train', scaler=None):
        df = pd.read_csv(csv_path)
        data = df.select_dtypes(include=[np.number]).values
        
        num_total = len(data)
        num_train = int(num_total * 0.7)
        num_val = int(num_total * 0.1)
        num_test = num_total - num_train - num_val

        borders = {
            'train': (0, num_train),
            'val': (num_train - lookback_len, num_train + num_val),
            'test': (num_total - num_test - lookback_len, num_total)
        }
        
        start_idx, end_idx = borders[split]
        
        # FIXED: Use the passed scaler for test/val, or create a new one for train
        if split == 'train':
            self.scaler = StandardScaler()
            self.scaler.fit(data[start_idx:end_idx])
        else:
            self.scaler = scaler
            
        self.data = self.scaler.transform(data[start_idx:end_idx])
        self.lookback_len = lookback_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.lookback_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.lookback_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        return (torch.tensor(self.data[s_begin:s_end], dtype=torch.float32), 
                torch.tensor(self.data[r_begin:r_end], dtype=torch.float32))