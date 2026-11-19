import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PatchTST import RevIN

def compute_spectral_entropy(x, eps=1e-8):
    """
    Computes normalized Shannon Spectral Entropy per univariate window instance.
    x shape: [B * M, L]
    Returns: [B * M, 1] entropy values in [0, 1]
    """
    x_centered = x - torch.mean(x, dim=-1, keepdim=True)
    fft_vals = torch.fft.rfft(x_centered, dim=-1)[..., 1:]
    psd = torch.abs(fft_vals) ** 2
    # Normalize Power Spectral Density into a probability distribution
    psd_norm = psd / (torch.sum(psd, dim=-1, keepdim=True) + eps)
    entropy = -torch.sum(psd_norm * torch.log2(psd_norm.clamp(min=eps)), dim=-1, keepdim=True)
    # Normalize by log2(number of non-DC frequency bins)
    max_entropy = torch.log2(torch.tensor(psd.shape[-1], dtype=torch.float32, device=x.device))
    return torch.clamp(entropy / (max_entropy + eps), 0.0, 1.0)

class FRPPatchTST(nn.Module):
    def __init__(self, num_variates, lookback_len=96, pred_len=96, patch_sizes=(8, 16, 32, 64), 
                 d_model=128, router_type='gumbel', tau=1.0, quartiles=(0.45, 0.60, 0.75)):
        super().__init__()
        self.pred_len = pred_len
        self.patch_sizes = patch_sizes
        self.router_type = router_type # 'gumbel' or 'heuristic'
        self.tau = tau
        self.quartiles = quartiles
        self.revin = RevIN(num_variates)

        # Learned Gumbel-Softmax Router (1D spectral entropy -> logits over patch sizes)
        self.router_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, len(patch_sizes))
        )

        # Resolution-specific tokenizers and heads sharing a single Transformer backbone
        self.embed_layers = nn.ModuleList()
        self.pos_embeds = nn.ParameterList()
        self.heads = nn.ModuleList()

        for P in patch_sizes:
            stride = P // 2
            num_patches = (lookback_len - P) // stride + 2
            self.embed_layers.append(nn.Linear(P, d_model))
            self.pos_embeds.append(nn.Parameter(torch.zeros(1, num_patches, d_model)))
            self.heads.append(nn.Linear(num_patches * d_model, pred_len))

        self.encoder = nn.TransformerEncoderLayer(
            d_model, nhead=8, dim_feedforward=2 * d_model, batch_first=True
        )

    def set_temperature(self, tau):
        """Update Gumbel-Softmax temperature for annealing."""
        self.tau = float(tau)

    @classmethod
    def from_config(cls, config_path, dataset_name, lookback_len, **kwargs):
        """Instantiate FRPPatchTST with calibrated quartile thresholds from JSON config."""
        with open(config_path, "r") as f:
            config = json.load(f)
        stats = config[dataset_name][str(lookback_len)]
        quartiles = (float(stats["q25"]), float(stats["q50"]), float(stats["q75"]))
        kwargs.setdefault("quartiles", quartiles)
        if "num_variates" not in kwargs:
            default_variates = {"ETTh1": 7, "ETTm1": 7, "traffic": 862}.get(dataset_name)
            if default_variates is None:
                raise ValueError(f"num_variates must be provided for dataset '{dataset_name}'")
            kwargs["num_variates"] = default_variates
        return cls(lookback_len=lookback_len, **kwargs)

    def get_routing_weights(self, entropy):
        if self.router_type == 'gumbel':
            logits = self.router_mlp(entropy)
            if self.training:
                # Straight-Through Gumbel-Softmax produces one-hot weights in forward, differentiable in backward
                return F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
            return F.one_hot(logits.argmax(dim=-1), num_classes=len(self.patch_sizes)).float()
        elif self.router_type == 'heuristic':
            # Deterministic quartile thresholding: Higher entropy -> smaller patch size
            q25, q50, q75 = self.quartiles
            idx = torch.zeros_like(entropy, dtype=torch.long)
            idx = torch.where(entropy < q75, 1, idx) # P=16
            idx = torch.where(entropy < q50, 2, idx) # P=32
            idx = torch.where(entropy < q25, 3, idx) # P=64
            return F.one_hot(idx.squeeze(-1), num_classes=len(self.patch_sizes)).float()

    def _encode_branch(self, x_sub, branch_idx):
        """Patch, embed, encode, and project a batch of univariate windows for branch_idx."""
        P = self.patch_sizes[branch_idx]
        stride = P // 2
        x_pad = torch.cat([x_sub, x_sub[:, -1:].repeat(1, stride)], dim=1)
        patches = x_pad.unfold(dimension=1, size=P, step=stride)
        tokens = self.embed_layers[branch_idx](patches) + self.pos_embeds[branch_idx]
        encoded = self.encoder(tokens)
        return self.heads[branch_idx](encoded.reshape(x_sub.shape[0], -1))

    def forward(self, x, return_routing=False):
        B, L, M = x.shape
        x = self.revin(x, mode='norm')
        x_ci = x.permute(0, 2, 1).reshape(B * M, L)

        entropy = compute_spectral_entropy(x_ci.detach())
        weights = self.get_routing_weights(entropy) # [B * M, num_resolutions]

        # Sparse dispatch when routing weights are non-differentiable; full pass during Gumbel training
        out_total = torch.zeros(B * M, self.pred_len, device=x.device, dtype=x.dtype)
        if not weights.requires_grad:
            for i in range(len(self.patch_sizes)):
                mask = weights[:, i] > 0.5
                if mask.any():
                    out_total[mask] = self._encode_branch(x_ci[mask], i)
        else:
            for i in range(len(self.patch_sizes)):
                pred_i = self._encode_branch(x_ci, i)
                out_total += pred_i * weights[:, i:i+1]

        pred = out_total.reshape(B, M, self.pred_len).permute(0, 2, 1)
        denorm_pred = self.revin(pred, mode='denorm')
        if return_routing:
            return denorm_pred, weights
        return denorm_pred