import torch
import torch.nn as nn
import math
import torch.fft

class SeriesDecomp(nn.Module):
    """Isolates long-term stationary trends from seasonal components[cite: 93]."""
    def __init__(self, kernel_size=25):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_trend = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_seasonal = x - x_trend
        return x_seasonal, x_trend

class AutoCorrelation(nn.Module):
    """Discovers period-based dependencies via Wiener-Khinchin theorem[cite: 93]."""
    def __init__(self, d_model, factor=3):
        super().__init__()
        self.factor = factor
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        # Simplified FFT-based cross-correlation for dimensional clarity
        q_fft = torch.fft.rfft(q, dim=1)
        k_fft = torch.fft.rfft(k, dim=1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=q.size(1), dim=1)
        
        # Roll and aggregate based on top autocorrelation delays
        top_k = int(self.factor * math.log(q.size(1)))
        weights, delays = torch.topk(corr.mean(dim=-1), k=top_k, dim=1)
        
        out = torch.zeros_like(v)
        for i in range(top_k):
            out += torch.roll(v, shifts=-int(delays[0, i]), dims=1) * torch.softmax(weights, dim=-1)[0, i]
        return self.proj(out)

class Autoformer(nn.Module):
    def __init__(self, num_variates, lookback_len, pred_len, d_model=128):
        super().__init__()
        self.decomp = SeriesDecomp()
        self.embedding = nn.Linear(num_variates, d_model)
        self.autocorr = AutoCorrelation(d_model)
        self.projection = nn.Linear(d_model, num_variates)
        self.pred_len = pred_len

    def forward(self, x):
        x_seasonal, x_trend = self.decomp(x)
        encoded = self.embedding(x_seasonal)
        # Simplified encoder/decoder integration for baseline testing
        correlated = self.autocorr(encoded, encoded, encoded)
        refined_seasonal = self.projection(correlated)[:, -self.pred_len:, :]
        
        # Extrapolate trend using global mean padding[cite: 93]
        mean_trend = x_trend.mean(dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        return refined_seasonal + mean_trend