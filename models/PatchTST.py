import torch
import torch.nn as nn

class RevIN(nn.Module):
    """Reversible Instance Normalization to address distribution shift[cite: 90]."""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            return x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            return x * self.stdev + self.mean

class PatchTST(nn.Module):
    def __init__(self, num_variates, lookback_len, pred_len, patch_len=16, stride=8, d_model=128):
        super().__init__()
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.revin = RevIN(num_variates)
        
        self.num_patches = (lookback_len - patch_len) // stride + 2
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        
        self.transformer = nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True)
        self.head = nn.Linear(self.num_patches * d_model, pred_len)

    def forward(self, x):
        B, L, M = x.shape
        x = self.revin(x, mode='norm')
        
        # Channel Independence: process each variate as an independent sequence[cite: 90]
        x_ci = x.permute(0, 2, 1).reshape(B * M, L)
        
        # Pad sequence and generate patches
        x_padded = torch.cat([x_ci, x_ci[:, -self.stride:]], dim=1)
        patches = x_padded.unfold(dimension=1, size=self.patch_len, step=self.stride)
        
        tokens = self.patch_embed(patches) + self.pos_embed
        out = self.transformer(tokens)
        
        # Decode and restore original multivariate shape
        pred_ci = self.head(out.reshape(B * M, -1))
        pred = pred_ci.reshape(B, M, self.pred_len).permute(0, 2, 1)
        return self.revin(pred, mode='denorm')