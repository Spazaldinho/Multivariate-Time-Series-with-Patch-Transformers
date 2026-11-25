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
                 d_model=128, router_type='gumbel', tau=1.0, quartiles=(0.45, 0.60, 0.75),
                 balance_weight=0.5):
        super().__init__()
        if router_type not in ('gumbel', 'heuristic', 'random'):
            raise ValueError(f"Unsupported router_type '{router_type}'; expected 'gumbel', 'heuristic', or 'random'.")
        self.pred_len = pred_len
        self.patch_sizes = patch_sizes
        self.router_type = router_type # 'gumbel', 'heuristic', or 'random'
        self.tau = tau
        self.quartiles = quartiles
        self.balance_weight = balance_weight
        self.aux_loss = 0.0
        self.revin = RevIN(num_variates)

        # Learned Gumbel-Softmax Router (IQR-normalized entropy -> residual logits over patch sizes)
        self.router_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, len(patch_sizes))
        )
        nn.init.normal_(self.router_mlp[-1].weight, std=0.05)
        nn.init.zeros_(self.router_mlp[-1].bias)

        # Learnable quartile anchors in IQR-normalized entropy space [P=8, P=16, P=32, P=64]
        self.anchor_centers = nn.Parameter(
            torch.tensor([[0.85, 0.25, -0.25, -0.85]])[:, :len(patch_sizes)].clone()
        )
        self.anchor_scale = nn.Parameter(torch.ones(1, len(patch_sizes)) * 1.5)

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

    def _compute_gumbel_logits(self, entropy):
        """Compute IQR-normalized anchor-biased router logits."""
        q25, q50, q75 = self.quartiles
        iqr = max(float(q75 - q25), 1e-4)
        h_norm = (entropy - float(q50)) / iqr
        anchor_logits = -F.softplus(self.anchor_scale) * (h_norm - self.anchor_centers) ** 2
        return self.router_mlp(h_norm) + anchor_logits

    def _compute_heuristic_indices(self, entropy_1d):
        """Map 1D spectral entropy [B*M] to deterministic quartile branch indices."""
        q25, q50, q75 = self.quartiles
        idx = torch.zeros_like(entropy_1d, dtype=torch.long)
        idx = torch.where(entropy_1d < q75, 1, idx) # P=16
        idx = torch.where(entropy_1d < q50, 2, idx) # P=32
        idx = torch.where(entropy_1d < q25, 3, idx) # P=64
        return idx

    def get_routing_weights(self, entropy):
        K = len(self.patch_sizes)
        if self.router_type == 'gumbel':
            logits = self._compute_gumbel_logits(entropy)
            if self.training:
                probs = F.softmax(logits / max(self.tau, 1e-4), dim=-1)
                mean_prob = probs.mean(dim=0)
                target_prob = 1.0 / K
                balance_loss = K * torch.sum((mean_prob - target_prob) ** 2)
                cond_entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-8)), dim=-1).mean()
                weights = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
                frac = weights.detach().mean(dim=0)
                self.aux_loss = (
                    self.balance_weight * balance_loss
                    + 0.02 * cond_entropy
                    + self.balance_weight * K * torch.sum(frac * mean_prob)
                )
                return weights
            self.aux_loss = torch.tensor(0.0, device=entropy.device, dtype=entropy.dtype)
            return F.one_hot(logits.argmax(dim=-1), num_classes=K).float()
        elif self.router_type == 'heuristic':
            self.aux_loss = torch.tensor(0.0, device=entropy.device, dtype=entropy.dtype)
            idx = self._compute_heuristic_indices(entropy.squeeze(-1))
            return F.one_hot(idx, num_classes=K).float()
        elif self.router_type == 'random':
            self.aux_loss = torch.tensor(0.0, device=entropy.device, dtype=entropy.dtype)
            idx = torch.randint(0, K, (entropy.shape[0],), device=entropy.device)
            return F.one_hot(idx, num_classes=K).float()

    def _encode_branch(self, x_sub, branch_idx):
        """Patch, embed, encode, and project a batch of univariate windows for branch_idx."""
        P = self.patch_sizes[branch_idx]
        stride = P // 2
        x_pad = torch.cat([x_sub, x_sub[:, -1:].repeat(1, stride)], dim=1)
        if x_pad.is_mps:
            num_patches = (x_sub.shape[1] - P) // stride + 2
            idx = (
                torch.arange(P, device=x_sub.device)[None, :]
                + stride * torch.arange(num_patches, device=x_sub.device)[:, None]
            )
            patches = x_pad[:, idx]
        else:
            patches = x_pad.unfold(dimension=1, size=P, step=stride)
        tokens = self.embed_layers[branch_idx](patches) + self.pos_embeds[branch_idx]
        encoded = self.encoder(tokens)
        return self.heads[branch_idx](encoded.reshape(x_sub.shape[0], -1))

    def _dispatch_sparse(self, x_ci, branch_idx):
        """Execute only the active resolution branches for the assigned windows."""
        preds = torch.empty((x_ci.shape[0], self.pred_len), device=x_ci.device, dtype=x_ci.dtype)
        for i in range(len(self.patch_sizes)):
            mask = (branch_idx == i)
            if mask.any():
                preds[mask] = self._encode_branch(x_ci[mask], i)
        return preds

    def forward(self, x, return_routing=False, dense_eval=False):
        B, L, M = x.shape
        K = len(self.patch_sizes)
        x = self.revin(x, mode='norm')
        x_ci = x.permute(0, 2, 1).reshape(B * M, L)

        if self.training:
            if self.router_type == 'gumbel':
                entropy = compute_spectral_entropy(x_ci.detach())
                weights = self.get_routing_weights(entropy)
                preds = torch.zeros((B * M, self.pred_len), device=x.device, dtype=x.dtype)
                for i in range(K):
                    preds = preds + self._encode_branch(x_ci, i) * weights[:, i:i+1]
            elif self.router_type == 'heuristic':
                self.aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
                entropy = compute_spectral_entropy(x_ci.detach()).squeeze(-1)
                branch_idx = self._compute_heuristic_indices(entropy)
                preds = self._dispatch_sparse(x_ci, branch_idx)
                weights = F.one_hot(branch_idx, num_classes=K).to(dtype=x.dtype) if return_routing else None
            else: # 'random'
                self.aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
                branch_idx = torch.randint(0, K, (B * M,), device=x.device)
                preds = self._dispatch_sparse(x_ci, branch_idx)
                weights = F.one_hot(branch_idx, num_classes=K).to(dtype=x.dtype) if return_routing else None
        else:
            self.aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            if self.router_type == 'gumbel':
                entropy = compute_spectral_entropy(x_ci.detach())
                branch_idx = self._compute_gumbel_logits(entropy).argmax(dim=-1)
            elif self.router_type == 'heuristic':
                entropy = compute_spectral_entropy(x_ci.detach()).squeeze(-1)
                branch_idx = self._compute_heuristic_indices(entropy)
            else: # 'random'
                branch_idx = torch.randint(0, K, (B * M,), device=x.device)

            if not dense_eval:
                preds = self._dispatch_sparse(x_ci, branch_idx)
            else:
                one_hot = F.one_hot(branch_idx, num_classes=K).to(dtype=x.dtype)
                preds = torch.zeros((B * M, self.pred_len), device=x.device, dtype=x.dtype)
                for i in range(K):
                    preds = preds + self._encode_branch(x_ci, i) * one_hot[:, i:i+1]
            weights = F.one_hot(branch_idx, num_classes=K).to(dtype=x.dtype) if return_routing else None

        pred = preds.reshape(B, M, self.pred_len).permute(0, 2, 1)
        denorm_pred = self.revin(pred, mode='denorm')
        if return_routing:
            return denorm_pred, weights
        return denorm_pred