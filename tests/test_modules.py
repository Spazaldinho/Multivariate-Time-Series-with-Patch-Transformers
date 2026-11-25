import torch
import pytest
from models.Autoformer import SeriesDecomp, AutoCorrelation, Autoformer
from models.PatchTST import RevIN, PatchTST
from models.iTransformer import iTransformer
from models.FRP_PatchTST import FRPPatchTST, compute_spectral_entropy

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
    (iTransformer, "iTransformer"),
    (FRPPatchTST, "FRPPatchTST_gumbel"),
    (FRPPatchTST, "FRPPatchTST_heuristic"),
    (FRPPatchTST, "FRPPatchTST_random"),
])
def test_model_forward_and_backward(model_class, model_name):
    """Verify output shapes [B, S, N] and check that backpropagation updates all parameters."""
    x = torch.randn(B, L, N, requires_grad=True)
    target = torch.randn(B, S, N)
    
    if model_name == "PatchTST":
        model = model_class(num_variates=N, lookback_len=L, pred_len=S, patch_len=16, stride=8)
    elif model_name == "FRPPatchTST_gumbel":
        model = model_class(num_variates=N, lookback_len=L, pred_len=S, router_type="gumbel")
    elif model_name == "FRPPatchTST_heuristic":
        model = model_class(num_variates=N, lookback_len=L, pred_len=S, router_type="heuristic")
    elif model_name == "FRPPatchTST_random":
        model = model_class(num_variates=N, lookback_len=L, pred_len=S, router_type="random")
    else:
        model = model_class(num_variates=N, lookback_len=L, pred_len=S)
        
    if model_name in ("FRPPatchTST_heuristic", "FRPPatchTST_random"):
        out, weights = model(x, return_routing=True)
        active_branches = {i for i in range(len(model.patch_sizes)) if (weights[:, i] > 0.5).any()}
    else:
        out = model(x)
        active_branches = None

    assert out.shape == (B, S, N), f"{model_name} output shape mismatch: got {out.shape}, expected {(B, S, N)}"
    assert not torch.isnan(out).any(), f"{model_name} produced NaN outputs"

    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()

    for name, param in model.named_parameters():
        if model_name in ("FRPPatchTST_heuristic", "FRPPatchTST_random"):
            if name.startswith("router_mlp.") or name in ("anchor_centers", "anchor_scale"):
                continue
            # Sparse dispatch only runs active branches present in the batch
            if any(name.startswith(f"{prefix}.{i}") for prefix in ("embed_layers", "pos_embeds", "heads") for i in range(len(model.patch_sizes)) if i not in active_branches):
                continue
        if param.requires_grad:
            assert param.grad is not None, f"Gradient not computed for parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN detected in gradients of {name}"

@pytest.mark.parametrize("lookback_len", [96, 336, 720])
def test_spectral_entropy_bounds(lookback_len):
    """Verify compute_spectral_entropy stays within [0.0, 1.0] across lookbacks."""
    x_rand = torch.randn(B * N, lookback_len)
    x_const = torch.ones(B * N, lookback_len) * 5.0
    t = torch.linspace(0, 8 * torch.pi, lookback_len).unsqueeze(0).repeat(B * N, 1)
    x_sine = torch.sin(t) + 10.0

    for x in (x_rand, x_const, x_sine):
        entropy = compute_spectral_entropy(x)
        assert entropy.shape == (B * N, 1)
        assert not torch.isnan(entropy).any()
        assert (entropy >= 0.0).all() and (entropy <= 1.0).all()

    # DC shift invariance: adding a constant offset must not change spectral entropy
    torch.testing.assert_close(
        compute_spectral_entropy(x_rand),
        compute_spectral_entropy(x_rand + 100.0),
        rtol=1e-4,
        atol=1e-4,
    )

def test_frp_from_config():
    """Verify FRPPatchTST.from_config loads calibrated quartiles and runs forward pass."""
    import json
    config_path = "configs/entropy_thresholds.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)
    expected = tuple(float(cfg["ETTh1"]["96"][k]) for k in ("q25", "q50", "q75"))

    model = FRPPatchTST.from_config(
        config_path, "ETTh1", 96, num_variates=7, pred_len=96, router_type="heuristic"
    )
    assert model.quartiles == expected

    x = torch.randn(2, 96, 7)
    out = model(x)
    assert out.shape == (2, 96, 7)
    assert not torch.isnan(out).any()

def test_frp_temperature_and_return_routing():
    """Verify set_temperature, return_routing=True, aux_loss grad, and multi-branch routing diversity."""
    model = FRPPatchTST(num_variates=N, lookback_len=L, pred_len=S, router_type="gumbel", tau=1.0)
    assert model.tau == 1.0
    model.set_temperature(0.5)
    assert model.tau == 0.5

    x = torch.randn(B, L, N)
    model.train()
    out_train, w_train = model(x, return_routing=True)
    assert out_train.shape == (B, S, N)
    assert w_train.shape == (B * N, 4)
    torch.testing.assert_close(w_train.sum(dim=-1), torch.ones(B * N))
    assert isinstance(model.aux_loss, torch.Tensor) and model.aux_loss.ndim == 0
    assert model.aux_loss.requires_grad

    model.eval()
    out_eval1, w_eval1 = model(x, return_routing=True)
    out_eval2, w_eval2 = model(x, return_routing=True)
    assert w_eval1.shape == (B * N, 4)
    torch.testing.assert_close(w_eval1.sum(dim=-1), torch.ones(B * N))
    torch.testing.assert_close(w_eval1, w_eval2)
    torch.testing.assert_close(out_eval1, out_eval2)

    # Synthetic batch spanning distinct spectral regimes (constant, low-freq, mid-freq, white noise)
    t = torch.linspace(0, 2 * torch.pi, L)
    x_const = torch.ones(L)
    x_low = torch.sin(t)
    x_mid = torch.sin(8 * t) + 0.5 * torch.sin(13 * t)
    torch.manual_seed(0)
    x_noise = torch.randn(L)
    variates = torch.stack([x_const, x_low, x_mid, x_noise, x_const * 2, x_low * 3, x_noise * 0.5], dim=-1)
    x_diverse = variates.unsqueeze(0).repeat(2, 1, 1)
    _, w_diverse = model(x_diverse, return_routing=True)
    active_branches = (w_diverse.sum(dim=0) > 0).sum().item()
    assert active_branches >= 2, f"Expected multi-branch routing across diverse spectra, got {w_diverse.sum(dim=0)}"


def test_frp_sparse_vs_dense_eval_equivalence_and_skip():
    """Verify sparse eval() matches dense one-hot eval() within 1e-5 and skips unused branches."""
    torch.manual_seed(42)
    x = torch.randn(B, L, N)

    for rtype in ("gumbel", "heuristic"):
        model = FRPPatchTST(num_variates=N, lookback_len=L, pred_len=S, router_type=rtype)
        model.eval()
        with torch.no_grad():
            out_sparse, w_sparse = model(x, return_routing=True, dense_eval=False)
            out_dense, w_dense = model(x, return_routing=True, dense_eval=True)
        assert torch.equal(w_sparse, w_dense)
        torch.testing.assert_close(out_sparse, out_dense, rtol=1e-5, atol=1e-5)

    rand_model = FRPPatchTST(num_variates=N, lookback_len=L, pred_len=S, router_type="random")
    rand_model.eval()
    with torch.no_grad():
        torch.manual_seed(123)
        out_r_sparse, w_r_sparse = rand_model(x, return_routing=True, dense_eval=False)
        torch.manual_seed(123)
        out_r_dense, w_r_dense = rand_model(x, return_routing=True, dense_eval=True)
    assert torch.equal(w_r_sparse, w_r_dense)
    torch.testing.assert_close(out_r_sparse, out_r_dense, rtol=1e-5, atol=1e-5)

    # Pure low-frequency sine wave routes all windows to a single branch, skipping the other 3
    t = torch.linspace(0, 4 * torch.pi, L)
    x_sine = torch.sin(t).view(1, L, 1).repeat(B, 1, N)
    for rtype in ("gumbel", "heuristic"):
        model = FRPPatchTST(num_variates=N, lookback_len=L, pred_len=S, router_type=rtype)
        model.eval()
        embed_calls = [0, 0, 0, 0]
        head_calls = [0, 0, 0, 0]
        hooks = []
        for idx in range(4):
            hooks.append(model.embed_layers[idx].register_forward_hook(lambda m, inp, out, i=idx: embed_calls.__setitem__(i, embed_calls[i] + 1)))
            hooks.append(model.heads[idx].register_forward_hook(lambda m, inp, out, i=idx: head_calls.__setitem__(i, head_calls[i] + 1)))

        with torch.no_grad():
            _, w = model(x_sine, return_routing=True, dense_eval=False)
        active = (w.sum(dim=0) > 0).tolist()
        assert sum(active) == 1, f"Expected single active branch for pure sine, got {w.sum(dim=0)}"
        for i, is_active in enumerate(active):
            expected_calls = 1 if is_active else 0
            assert embed_calls[i] == expected_calls, f"embed_layers[{i}] called {embed_calls[i]} times"
            assert head_calls[i] == expected_calls, f"heads[{i}] called {head_calls[i]} times"

        embed_calls = [0, 0, 0, 0]
        head_calls = [0, 0, 0, 0]
        with torch.no_grad():
            _ = model(x_sine, dense_eval=True)
        assert embed_calls == [1, 1, 1, 1] and head_calls == [1, 1, 1, 1]
        for h in hooks:
            h.remove()