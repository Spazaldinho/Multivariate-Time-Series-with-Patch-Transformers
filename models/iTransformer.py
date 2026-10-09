import torch
import torch.nn as nn

class iTransformer(nn.Module):
    def __init__(self, num_variates, lookback_len, pred_len, d_model=512):
        super().__init__()
        self.pred_len = pred_len
        
        # Embed the entire historical sequence of a single variate into one token[cite: 91]
        self.embedding = nn.Linear(lookback_len, d_model)
        
        # Transformer blocks operate on the inverted [Batch, Variates, D] dimensions
        self.transformer = nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True)
        
        # Dense linear projection directly to future forecast horizon
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x):
        B, T, N = x.shape
        
        # Temporal instance normalization[cite: 91]
        means = x.mean(dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(x - means, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x - means) / stdev
        
        # Invert the time and variate axes[cite: 89, 91]
        x_inverted = x_norm.transpose(1, 2)
        
        # Variate attention and temporal Feed-Forward Networks[cite: 89]
        tokens = self.embedding(x_inverted)
        encoded = self.transformer(tokens)
        
        y_hat = self.projection(encoded).transpose(1, 2)
        return y_hat * stdev + means