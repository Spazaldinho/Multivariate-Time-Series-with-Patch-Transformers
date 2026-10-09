import torch
import pytest
from models.Autoformer import SeriesDecomp, AutoCorrelation, Autoformer
from models.PatchTST import RevIN, PatchTST
from models.iTransformer import iTransformer

# Fixed dummy dimensions: Batch=4, Lookback=96, Prediction=96, Variates=7
B, L, S, N = 4, 96, 96, 7

def test_series_decomp_identity():
    """Verify that seasonal and trend components sum back exactly to the original input."""
    decomp = SeriesDecomp(kernel_size=25)
    x = torch.randn(B, L, N)
    x_s, x_t = decomp(x)
    assert x_s.shape == (B, L, N), f"Seasonal shape mismatch: {x_s.shape}"
    assert x_t.shape == (B, L, N), f"Trend shape mismatch: {x_t.shape}"
    torch.testing.assert_close(x_s + x_t, x, rtol=1e-5, atol=1e-5)

def test_revin_reversibility():
    """Verify RevIN normalizes to zero-mean/unit-variance and reverses losslessly[cite: 90, 91]."""
    revin = RevIN(num_features=N)
    x = torch.randn(B, L, N) * 5.0 + 12.0
    x_norm = revin(x, mode='norm')
    
    # Check zero mean and unit variance per sequence instance
    torch.testing.assert_close(x_norm.mean(dim=1), torch.zeros(B, N), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(x_norm.var(dim=1, unbiased=False), torch.ones(B, N), atol=1e-3, rtol=1e-3)
    
    x_denorm = revin(x_norm, mode='denorm')
    torch.testing.assert_close(x_denorm, x, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("model_class,model_name", [
    (Autoformer, "Autoformer"),
    (PatchTST, "PatchTST"),
    (iTransformer, "iTransformer")
])
def test_model_forward_and_backward(model_class, model_name):
    """Verify output shapes [B, S, N] and check that backpropagation updates all parameters."""
    x = torch.randn(B, L, N, requires_grad=True)
    target = torch.randn(B, S, N)
    
    if model_name == "PatchTST":
        model = model_class(num_variates=N, lookback_len=L, pred_len=S, patch_len=16, stride=8)
    else:
        model = model_class(num_variates=N, lookback_len=L, pred_len=S)
        
    out = model(x)
    assert out.shape == (B, S, N), f"{model_name} output shape mismatch: got {out.shape}, expected {(B, S, N)}"
    assert not torch.isnan(out).any(), f"{model_name} produced NaN outputs"

    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient not computed for parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN detected in gradients of {name}"