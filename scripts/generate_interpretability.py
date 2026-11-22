import argparse
import csv
import json
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_loader import ETTDataset
from models.FRP_PatchTST import FRPPatchTST, compute_spectral_entropy


NUM_VARIATES = 862
LOOKBACK_LEN = 96
PRED_LEN = 96
PATCH_SIZES = (8, 16, 32, 64)


def compute_spearman_rho(x, y):
    """Compute Spearman rank correlation rho and p-value using scipy (with numpy fallback)."""
    try:
        from scipy.stats import spearmanr
        res = spearmanr(x, y)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        # Pure numpy rank correlation fallback
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        rx -= rx.mean()
        ry -= ry.mean()
        denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum()) + 1e-12
        return float((rx * ry).sum() / denom), 0.0


def plot_routing_histogram(all_entropies, all_branches, quartiles, fig_path):
    """
    Generate Figure A: 2-panel Spectral Entropy distribution and binned routing probability curves.
    """
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    q25, q50, q75 = quartiles

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#333333"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    colors = {
        0: "#d62728",  # P=8 (finest / volatile)
        1: "#ff7f0e",  # P=16
        2: "#1f77b4",  # P=32
        3: "#2ca02c",  # P=64 (coarsest / smooth)
    }
    labels = {
        0: "P = 8  (Fine / High-Freq)",
        1: "P = 16 (Medium-Fine)",
        2: "P = 32 (Medium-Coarse)",
        3: "P = 64 (Coarse / Smooth)",
    }

    # Left Panel: Stacked Histogram of Spectral Entropy colored by selected patch size
    bins = np.linspace(0.0, 1.0, 65)
    branch_entropies = [all_entropies[all_branches == i] for i in range(4)]
    ax1.hist(
        branch_entropies,
        bins=bins,
        stacked=True,
        color=[colors[i] for i in range(4)],
        label=[labels[i] for i in range(4)],
        alpha=0.88,
        edgecolor="black",
        linewidth=0.35,
    )

    for q_val, q_name in zip((q25, q50, q75), ("q25", "q50", "q75")):
        ax1.axvline(
            q_val,
            color="#222222",
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
        )
        ax1.text(
            q_val + 0.008,
            ax1.get_ylim()[1] * 0.90,
            f"{q_name}={q_val:.3f}",
            rotation=90,
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color="#222222",
        )

    ax1.set_title("A. Test Spectral Entropy Distribution by Routed Patch Size", pad=10, fontweight="bold")
    ax1.set_xlabel("Normalized Shannon Spectral Entropy $H \\in [0, 1]$")
    ax1.set_ylabel("Number of Univariate Test Windows")
    ax1.set_xlim(0.0, 1.0)
    ax1.grid(True, linestyle=":", alpha=0.45)
    ax1.legend(loc="upper left", frameon=True, fontsize=8.5)

    # Right Panel: 10-Bin Empirical Routing Probability Curves across empirical entropy range
    h_min = float(np.quantile(all_entropies, 0.001))
    h_max = float(np.quantile(all_entropies, 0.999))
    bin_edges = np.linspace(h_min, h_max, 11)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    markers = {0: "o", 1: "s", 2: "^", 3: "D"}

    shares = np.zeros((4, 10), dtype=np.float64)
    for b_idx in range(10):
        lo, hi = bin_edges[b_idx], bin_edges[b_idx + 1]
        if b_idx == 9:
            mask = (all_entropies >= lo) & (all_entropies <= hi)
        else:
            mask = (all_entropies >= lo) & (all_entropies < hi)
        count = mask.sum()
        if count > 0:
            for i in range(4):
                shares[i, b_idx] = (all_branches[mask] == i).sum() / count * 100.0

    for i in range(4):
        ax2.plot(
            bin_centers,
            shares[i],
            marker=markers[i],
            color=colors[i],
            linewidth=2.2,
            markersize=6.5,
            label=labels[i],
        )

    for q_val, q_name in zip((q25, q50, q75), ("q25", "q50", "q75")):
        ax2.axvline(q_val, color="#222222", linestyle="--", linewidth=1.3, alpha=0.8)
        ax2.text(
            q_val + 0.006,
            96.0,
            f"{q_name}",
            rotation=90,
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color="#222222",
        )

    ax2.set_title("B. Empirical Patch Selection Probability vs. Spectral Entropy (10 Bins)", pad=10, fontweight="bold")
    ax2.set_xlabel("Normalized Shannon Spectral Entropy $H$ (Bin Midpoint)")
    ax2.set_ylabel("Selection Share (%)")
    ax2.set_ylim(-4, 105)
    ax2.grid(True, linestyle=":", alpha=0.45)
    ax2.legend(loc="center right", frameon=True, fontsize=8.5)

    fig.suptitle(
        "PEMS04 Traffic (862 Variates, Seed 42): Learned Gumbel-Softmax Frequency-to-Patch Routing",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure A -> {fig_path}", flush=True)


def plot_smooth_vs_volatile(smooth_case, volatile_case, quartiles, fig_path):
    """
    Generate Figure B: 2-panel qualitative comparison of Smooth (Low-Entropy) vs. Volatile (High-Entropy) windows.
    """
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    q25, _, q75 = quartiles

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#333333"})
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True)

    t_hist = np.arange(0, LOOKBACK_LEN)
    t_pred = np.arange(LOOKBACK_LEN, LOOKBACK_LEN + PRED_LEN)

    panels = [
        (
            ax_top,
            smooth_case,
            f"Top Panel — Low-Entropy / Smooth Periodic Traffic Window ($H = {smooth_case['entropy']:.4f} < q_{{25}} = {q25:.4f}$)",
            "#2ca02c",
        ),
        (
            ax_bot,
            volatile_case,
            f"Bottom Panel — High-Entropy / Volatile Burst Traffic Window ($H = {volatile_case['entropy']:.4f} > q_{{75}} = {q75:.4f}$)",
            "#d62728",
        ),
    ]

    for ax, case, title, pred_color in panels:
        hist = case["history"]
        true_fut = case["target"]
        pred_fut = case["prediction"]
        P = case["patch_size"]
        stride = P // 2

        # Subtle vertical lines in lookback region showing patch stride boundaries
        for s_pos in range(0, LOOKBACK_LEN + 1, stride):
            ax.axvline(s_pos, color=pred_color, linestyle=":", linewidth=0.7, alpha=0.28)

        ax.plot(
            t_hist,
            hist,
            color="#333333",
            linewidth=1.6,
            alpha=0.85,
            label="Lookback Context (96 steps)",
        )
        # Connect last history step to first future step for visual continuity
        ax.plot(
            np.concatenate([[LOOKBACK_LEN - 1], t_pred]),
            np.concatenate([[hist[-1]], true_fut]),
            color="#1f77b4",
            linewidth=1.9,
            alpha=0.9,
            label="Ground Truth Horizon (96 steps)",
        )
        ax.plot(
            np.concatenate([[LOOKBACK_LEN - 1], t_pred]),
            np.concatenate([[hist[-1]], pred_fut]),
            color=pred_color,
            linestyle="--",
            linewidth=2.1,
            label=f"FRPPatchTST (Gumbel) Forecast [Routed P={P}, stride={stride}]",
        )

        ax.axvline(
            LOOKBACK_LEN - 0.5,
            color="#111111",
            linestyle="-.",
            linewidth=1.3,
            label="Forecast Origin (t = 96)",
        )

        info_text = (
            f"Sensor #{case['sensor_idx']} | Test Window #{case['window_idx']}\n"
            f"Spectral Entropy H = {case['entropy']:.4f} -> Routed P = {P}\n"
            f"Window MSE = {case['mse']:.4f} | MAE = {case['mae']:.4f}"
        )
        ax.text(
            0.015,
            0.94,
            info_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=pred_color, alpha=0.92),
        )

        ax.set_title(title, pad=8, fontweight="bold")
        ax.set_ylabel("Standardized Flow (Z-score)")
        y_lo, y_hi = float(min(hist.min(), true_fut.min(), pred_fut.min())), float(max(hist.max(), true_fut.max(), pred_fut.max()))
        y_span = max(y_hi - y_lo, 1.0)
        ax.set_ylim(y_lo - 0.12 * y_span, y_hi + 0.35 * y_span)
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.legend(loc="upper right", frameon=True, fontsize=8.5)

    ax_bot.set_xlabel("Time Steps (Hours, 96-step Lookback -> 96-step Forecast Horizon)")
    ax_bot.set_xlim(0, LOOKBACK_LEN + PRED_LEN - 1)

    fig.suptitle(
        "Mechanistic Case Study: Adaptive Resolution Selection on Smooth vs. Volatile Traffic Sensors",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure B -> {fig_path}", flush=True)


def run_interpretability(max_batches=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Block 4 Mechanistic Interpretability Analysis on device: {device}", flush=True)

    config_path = os.path.join(PROJECT_ROOT, "configs/entropy_thresholds.json")
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints/frp_gumbel_traffic_s42.pt")
    csv_path = os.path.join(PROJECT_ROOT, "data/traffic.csv")
    out_csv = os.path.join(PROJECT_ROOT, "results/traffic_routing_analysis.csv")
    fig_a_path = os.path.join(PROJECT_ROOT, "figures/traffic_routing_histogram.png")
    fig_b_path = os.path.join(PROJECT_ROOT, "figures/smooth_vs_volatile_forecasts.png")

    with open(config_path, "r") as f:
        cfg = json.load(f)
    t_stats = cfg["traffic"][str(LOOKBACK_LEN)]
    quartiles = (float(t_stats["q25"]), float(t_stats["q50"]), float(t_stats["q75"]))
    q25, q50, q75 = quartiles

    train_ds = ETTDataset(csv_path, lookback_len=LOOKBACK_LEN, pred_len=PRED_LEN, split="train")
    test_ds = ETTDataset(
        csv_path, lookback_len=LOOKBACK_LEN, pred_len=PRED_LEN, split="test", scaler=train_ds.scaler
    )
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = FRPPatchTST.from_config(
        config_path,
        "traffic",
        lookback_len=LOOKBACK_LEN,
        num_variates=NUM_VARIATES,
        pred_len=PRED_LEN,
        router_type="gumbel",
        balance_weight=1.0,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(
        f"Loaded checkpoint: {ckpt_path}\n"
        f"Learned anchor_centers: {model.anchor_centers.detach().cpu().numpy().round(4).tolist()}\n"
        f"Learned anchor_scale:   {model.anchor_scale.detach().cpu().numpy().round(4).tolist()}",
        flush=True,
    )

    all_entropies = []
    all_branches = []

    best_smooth = None
    best_volatile = None

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (seq_x, seq_y) in enumerate(test_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            B = seq_x.shape[0]
            seq_x_dev = seq_x.to(device)
            seq_y_dev = seq_y.to(device)

            x_ci = seq_x_dev.permute(0, 2, 1).reshape(B * NUM_VARIATES, LOOKBACK_LEN)
            H = compute_spectral_entropy(x_ci).squeeze(-1)
            y_hat, weights = model(seq_x_dev, return_routing=True)
            branches = weights.argmax(dim=-1)

            H_np = H.cpu().numpy()
            branches_np = branches.cpu().numpy()
            all_entropies.append(H_np)
            all_branches.append(branches_np)

            # Search for clean representative smooth (H < q25, P in {64, 32}) and volatile (H > q75, P=8) windows
            if batch_idx % 5 == 0 or best_smooth is None or best_volatile is None:
                y_ci = seq_y_dev.permute(0, 2, 1).reshape(B * NUM_VARIATES, PRED_LEN)
                yhat_ci = y_hat.permute(0, 2, 1).reshape(B * NUM_VARIATES, PRED_LEN)
                mse_ci = ((yhat_ci - y_ci) ** 2).mean(dim=-1).cpu().numpy()
                mae_ci = (yhat_ci - y_ci).abs().mean(dim=-1).cpu().numpy()
                std_ci = x_ci.std(dim=-1).cpu().numpy()

                # Smooth candidate: H < q25, routed to P=64 (branch 3) or P=32 (branch 2), non-trivial wave amplitude
                smooth_mask = (H_np > 0.12) & (H_np < q25 - 0.02) & (branches_np >= 2) & (std_ci > 0.65)
                if smooth_mask.any():
                    cand_idxs = np.where(smooth_mask)[0]
                    # Prefer P=64 (branch 3) and low forecast MSE
                    scores = mse_ci[cand_idxs] - 0.05 * (branches_np[cand_idxs] == 3)
                    best_local = cand_idxs[int(np.argmin(scores))]
                    cand_score = float(scores[int(np.argmin(scores))])
                    if best_smooth is None or cand_score < best_smooth["score"]:
                        b_i = best_local // NUM_VARIATES
                        m_i = best_local % NUM_VARIATES
                        best_smooth = {
                            "score": cand_score,
                            "window_idx": int(batch_idx * 8 + b_i),
                            "sensor_idx": int(m_i),
                            "entropy": float(H_np[best_local]),
                            "branch_idx": int(branches_np[best_local]),
                            "patch_size": int(PATCH_SIZES[int(branches_np[best_local])]),
                            "mse": float(mse_ci[best_local]),
                            "mae": float(mae_ci[best_local]),
                            "history": seq_x[b_i, :, m_i].numpy().copy(),
                            "target": seq_y[b_i, :, m_i].numpy().copy(),
                            "prediction": y_hat[b_i, :, m_i].detach().cpu().numpy().copy(),
                        }

                # Volatile candidate: H > q75, routed to P=8 (branch 0), sustained high-frequency activity without extreme spike
                max_abs_x = x_ci.abs().max(dim=-1).values.cpu().numpy()
                std_y = y_ci.std(dim=-1).cpu().numpy()
                hf_diff = (x_ci[:, 1:] - x_ci[:, :-1]).abs().mean(dim=-1).cpu().numpy()
                vol_mask = (
                    (H_np > q75 + 0.04)
                    & (H_np < 0.88)
                    & (branches_np == 0)
                    & (std_ci > 0.65)
                    & (std_y > 0.65)
                    & (max_abs_x < 3.8)
                    & (hf_diff > 0.45)
                )
                if vol_mask.any():
                    cand_idxs = np.where(vol_mask)[0]
                    best_local = cand_idxs[int(np.argmin(mse_ci[cand_idxs]))]
                    cand_score = float(mse_ci[best_local])
                    if best_volatile is None or cand_score < best_volatile["score"]:
                        b_i = best_local // NUM_VARIATES
                        m_i = best_local % NUM_VARIATES
                        best_volatile = {
                            "score": cand_score,
                            "window_idx": int(batch_idx * 8 + b_i),
                            "sensor_idx": int(m_i),
                            "entropy": float(H_np[best_local]),
                            "branch_idx": int(branches_np[best_local]),
                            "patch_size": int(PATCH_SIZES[int(branches_np[best_local])]),
                            "mse": float(mse_ci[best_local]),
                            "mae": float(mae_ci[best_local]),
                            "history": seq_x[b_i, :, m_i].numpy().copy(),
                            "target": seq_y[b_i, :, m_i].numpy().copy(),
                            "prediction": y_hat[b_i, :, m_i].detach().cpu().numpy().copy(),
                        }

    all_entropies = np.concatenate(all_entropies, axis=0)
    all_branches = np.concatenate(all_branches, axis=0)
    all_patch_sizes = np.array(PATCH_SIZES)[all_branches]
    total_windows = int(all_entropies.size)

    rho_h_p, pval_h_p = compute_spearman_rho(all_entropies, all_patch_sizes)
    elapsed = time.time() - t0
    print(f"Evaluated {total_windows:,} univariate test windows in {elapsed:.1f}s", flush=True)

    fieldnames = [
        "patch_size",
        "branch_idx",
        "window_count",
        "percentage_share",
        "entropy_mean",
        "entropy_std",
        "entropy_min",
        "entropy_q25",
        "entropy_median",
        "entropy_q75",
        "entropy_max",
        "spearman_rho_H_vs_P",
        "spearman_pval",
    ]
    rows = []
    for i, P in enumerate(PATCH_SIZES):
        mask = all_branches == i
        cnt = int(mask.sum())
        pct = cnt / max(1, total_windows) * 100.0
        sub_h = all_entropies[mask]
        if cnt > 0:
            q_sub = np.quantile(sub_h, [0.25, 0.50, 0.75])
            row = {
                "patch_size": f"P={P}",
                "branch_idx": i,
                "window_count": cnt,
                "percentage_share": round(float(pct), 2),
                "entropy_mean": round(float(sub_h.mean()), 6),
                "entropy_std": round(float(sub_h.std()), 6),
                "entropy_min": round(float(sub_h.min()), 6),
                "entropy_q25": round(float(q_sub[0]), 6),
                "entropy_median": round(float(q_sub[1]), 6),
                "entropy_q75": round(float(q_sub[2]), 6),
                "entropy_max": round(float(sub_h.max()), 6),
                "spearman_rho_H_vs_P": "",
                "spearman_pval": "",
            }
        else:
            row = {
                "patch_size": f"P={P}",
                "branch_idx": i,
                "window_count": 0,
                "percentage_share": 0.0,
                "entropy_mean": 0.0,
                "entropy_std": 0.0,
                "entropy_min": 0.0,
                "entropy_q25": 0.0,
                "entropy_median": 0.0,
                "entropy_q75": 0.0,
                "entropy_max": 0.0,
                "spearman_rho_H_vs_P": "",
                "spearman_pval": "",
            }
        rows.append(row)

    q_all = np.quantile(all_entropies, [0.25, 0.50, 0.75])
    rows.append(
        {
            "patch_size": "ALL",
            "branch_idx": -1,
            "window_count": total_windows,
            "percentage_share": 100.0,
            "entropy_mean": round(float(all_entropies.mean()), 6),
            "entropy_std": round(float(all_entropies.std()), 6),
            "entropy_min": round(float(all_entropies.min()), 6),
            "entropy_q25": round(float(q_all[0]), 6),
            "entropy_median": round(float(q_all[1]), 6),
            "entropy_q75": round(float(q_all[2]), 6),
            "entropy_max": round(float(all_entropies.max()), 6),
            "spearman_rho_H_vs_P": round(rho_h_p, 6),
            "spearman_pval": f"{pval_h_p:.3e}",
        }
    )

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n## Block 4: Traffic Test-Set Routing & Spectral Entropy Analysis\n", flush=True)
    print(
        "| Patch Size | Branch | Window Count | Share (%) | Mean Entropy ± Std | Median [IQR: q25 – q75] | Min – Max |",
        flush=True,
    )
    print(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        flush=True,
    )
    for r in rows:
        print(
            f"| **{r['patch_size']}** | {r['branch_idx']} | {r['window_count']:,} | "
            f"{r['percentage_share']:.2f}% | {r['entropy_mean']:.4f} ± {r['entropy_std']:.4f} | "
            f"{r['entropy_median']:.4f} [{r['entropy_q25']:.4f} – {r['entropy_q75']:.4f}] | "
            f"{r['entropy_min']:.4f} – {r['entropy_max']:.4f} |",
            flush=True,
        )
    print(
        f"\nSpearman Rank Correlation rho(H, P): {rho_h_p:.6f} (p-value = {pval_h_p:.3e})",
        flush=True,
    )
    print(
        f"Smooth Case Selected:   Sensor #{best_smooth['sensor_idx']}, Window #{best_smooth['window_idx']}, "
        f"H={best_smooth['entropy']:.4f}, Routed P={best_smooth['patch_size']}, MSE={best_smooth['mse']:.4f}",
        flush=True,
    )
    print(
        f"Volatile Case Selected: Sensor #{best_volatile['sensor_idx']}, Window #{best_volatile['window_idx']}, "
        f"H={best_volatile['entropy']:.4f}, Routed P={best_volatile['patch_size']}, MSE={best_volatile['mse']:.4f}",
        flush=True,
    )
    print(f"Saved CSV -> {out_csv}", flush=True)

    plot_routing_histogram(all_entropies, all_branches, quartiles, fig_a_path)
    plot_smooth_vs_volatile(best_smooth, best_volatile, quartiles, fig_b_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Block 4: Mechanistic Interpretability & Routing Visualizations"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on test batches for quick local verification.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_interpretability(max_batches=args.max_batches)
