import argparse
import copy
import csv
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_loader import ETTDataset
from models.FRP_PatchTST import FRPPatchTST
from models.PatchTST import PatchTST
from models.iTransformer import iTransformer


CSV_FIELDNAMES = [
    "row_type",
    "dataset",
    "lookback",
    "pred_len",
    "model",
    "seed",
    "num_params",
    "best_epoch",
    "best_val_mse",
    "mse",
    "mse_std",
    "mae",
    "mae_std",
    "latency_ms",
    "latency_std",
    "peak_mem_mb",
    "pct_p8",
    "pct_p16",
    "pct_p32",
    "pct_p64",
]

NUM_VARIATES = 862
LOOKBACK_LEN = 96
PRED_LEN = 96


def build_model(model_label, config_path):
    """Instantiate one of the five Block 3 benchmark models for full 862-variate traffic."""
    if model_label == "PatchTST (P=16)":
        return PatchTST(
            num_variates=NUM_VARIATES,
            lookback_len=LOOKBACK_LEN,
            pred_len=PRED_LEN,
            patch_len=16,
            stride=8,
            d_model=128,
        )
    if model_label == "PatchTST (P=32)":
        return PatchTST(
            num_variates=NUM_VARIATES,
            lookback_len=LOOKBACK_LEN,
            pred_len=PRED_LEN,
            patch_len=32,
            stride=16,
            d_model=128,
        )
    if model_label == "iTransformer":
        return iTransformer(
            num_variates=NUM_VARIATES,
            lookback_len=LOOKBACK_LEN,
            pred_len=PRED_LEN,
            d_model=512,
        )
    if model_label == "FRPPatchTST (Heuristic)":
        return FRPPatchTST.from_config(
            config_path,
            "traffic",
            lookback_len=LOOKBACK_LEN,
            num_variates=NUM_VARIATES,
            pred_len=PRED_LEN,
            router_type="heuristic",
        )
    if model_label == "FRPPatchTST (Gumbel)":
        return FRPPatchTST.from_config(
            config_path,
            "traffic",
            lookback_len=LOOKBACK_LEN,
            num_variates=NUM_VARIATES,
            pred_len=PRED_LEN,
            router_type="gumbel",
            tau=1.0,
            balance_weight=1.0,
        )
    raise ValueError(f"Unknown model label: {model_label}")


def evaluate_mse(model, loader, device, max_batches=None):
    """Compute exact dataset-level MSE on a DataLoader."""
    model.eval()
    sq_err = 0.0
    total_elems = 0
    with torch.no_grad():
        for idx, (seq_x, seq_y) in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            pred = model(seq_x.to(device))
            target = seq_y.to(device)
            sq_err += ((pred - target) ** 2).sum().item()
            total_elems += target.numel()
    return sq_err / max(1, total_elems)


def profile_and_evaluate_test(model, loader, device, warmup_batches=10, max_batches=None):
    """Evaluate test MSE/MAE, routing distribution, mean inference latency, and peak VRAM."""
    model.eval()
    is_frp = isinstance(model, FRPPatchTST)
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        # Warm-up passes before resetting memory/timing stats
        for idx, (seq_x, _) in enumerate(loader):
            if idx >= warmup_batches:
                break
            _ = model(seq_x.to(device))

        if use_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        sq_err = 0.0
        abs_err = 0.0
        total_elems = 0
        latencies_ms = []
        routing_counts = (
            torch.zeros(len(model.patch_sizes), dtype=torch.float64) if is_frp else None
        )
        total_windows = 0

        for idx, (seq_x, seq_y) in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            seq_x_dev = seq_x.to(device)
            seq_y_dev = seq_y.to(device)

            if use_cuda:
                start_event.record()
                if is_frp:
                    pred, weights = model(seq_x_dev, return_routing=True)
                else:
                    pred = model(seq_x_dev)
                end_event.record()
                torch.cuda.synchronize(device)
                latencies_ms.append(start_event.elapsed_time(end_event))
            else:
                t0 = time.perf_counter()
                if is_frp:
                    pred, weights = model(seq_x_dev, return_routing=True)
                else:
                    pred = model(seq_x_dev)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            diff = pred - seq_y_dev
            sq_err += (diff ** 2).sum().item()
            abs_err += diff.abs().sum().item()
            total_elems += seq_y_dev.numel()

            if is_frp:
                routing_counts += weights.detach().cpu().double().sum(dim=0)
                total_windows += weights.shape[0]

        peak_mem_mb = (
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2) if use_cuda else 0.0
        )

    mse = sq_err / max(1, total_elems)
    mae = abs_err / max(1, total_elems)
    mean_latency_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    pcts = (
        (routing_counts / max(1, total_windows) * 100.0).tolist()
        if is_frp
        else None
    )
    return {
        "mse": mse,
        "mae": mae,
        "latency_ms": mean_latency_ms,
        "peak_mem_mb": peak_mem_mb,
        "pcts": pcts,
    }


def train_and_eval_seed(
    model_label,
    seed,
    train_loader,
    val_loader,
    test_loader,
    device,
    config_path,
    ckpt_path=None,
    max_epochs=10,
    patience=3,
    grad_accum_steps=4,
    lr=1e-4,
    aux_loss_weight=0.01,
    warmup_batches=10,
    max_train_batches=None,
    max_val_batches=None,
    max_test_batches=None,
):
    """Train one seed with gradient accumulation and validation early stopping, then profile test."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    model = build_model(model_label, config_path).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    is_frp = isinstance(model, FRPPatchTST)
    is_gumbel = is_frp and model.router_type == "gumbel"

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_mse = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0

    for ep in range(max_epochs):
        if is_frp:
            tau = 1.0 - 0.5 * (ep / max(1, max_epochs - 1))
            model.set_temperature(tau)

        model.train()
        optimizer.zero_grad()
        step_count = 0

        for step_idx, (seq_x, seq_y) in enumerate(train_loader):
            if max_train_batches is not None and step_idx >= max_train_batches:
                break
            pred = model(seq_x.to(device))
            loss = criterion(pred, seq_y.to(device))
            if is_gumbel:
                loss = loss + aux_loss_weight * model.aux_loss
            (loss / grad_accum_steps).backward()
            step_count += 1
            if step_count % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        if step_count % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        val_mse = evaluate_mse(model, val_loader, device, max_batches=max_val_batches)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    if ckpt_path is not None:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(best_state, ckpt_path)
        print(f"  Saved checkpoint -> {ckpt_path}", flush=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    test_stats = profile_and_evaluate_test(
        model,
        test_loader,
        device,
        warmup_batches=warmup_batches,
        max_batches=max_test_batches,
    )

    row = {
        "row_type": "seed",
        "dataset": "traffic",
        "lookback": LOOKBACK_LEN,
        "pred_len": PRED_LEN,
        "model": model_label,
        "seed": str(seed),
        "num_params": num_params,
        "best_epoch": best_epoch,
        "best_val_mse": round(best_val_mse, 6),
        "mse": round(test_stats["mse"], 6),
        "mse_std": "",
        "mae": round(test_stats["mae"], 6),
        "mae_std": "",
        "latency_ms": round(test_stats["latency_ms"], 4),
        "latency_std": "",
        "peak_mem_mb": round(test_stats["peak_mem_mb"], 2),
        "pct_p8": round(test_stats["pcts"][0], 2) if test_stats["pcts"] is not None else "",
        "pct_p16": round(test_stats["pcts"][1], 2) if test_stats["pcts"] is not None else "",
        "pct_p32": round(test_stats["pcts"][2], 2) if test_stats["pcts"] is not None else "",
        "pct_p64": round(test_stats["pcts"][3], 2) if test_stats["pcts"] is not None else "",
    }
    return row


def load_existing_seed_rows(output_csv):
    """Load previously completed seed rows for resume support."""
    if not os.path.exists(output_csv):
        return {}
    cached = {}
    with open(output_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("row_type") == "seed":
                cached[(row["model"], int(row["seed"]))] = row
    return cached


def compute_summary_row(model_label, seed_rows):
    """Aggregate Mean ± Std summary row from completed seed rows for a model."""
    mses = [float(r["mse"]) for r in seed_rows]
    maes = [float(r["mae"]) for r in seed_rows]
    vals = [float(r["best_val_mse"]) for r in seed_rows]
    lats = [float(r["latency_ms"]) for r in seed_rows]
    mems = [float(r["peak_mem_mb"]) for r in seed_rows]
    epochs = [float(r["best_epoch"]) for r in seed_rows]
    num_params = int(seed_rows[0]["num_params"])

    has_routing = seed_rows[0]["pct_p8"] != ""
    if has_routing:
        p8 = round(float(np.mean([float(r["pct_p8"]) for r in seed_rows])), 2)
        p16 = round(float(np.mean([float(r["pct_p16"]) for r in seed_rows])), 2)
        p32 = round(float(np.mean([float(r["pct_p32"]) for r in seed_rows])), 2)
        p64 = round(float(np.mean([float(r["pct_p64"]) for r in seed_rows])), 2)
    else:
        p8 = p16 = p32 = p64 = ""

    return {
        "row_type": "summary",
        "dataset": "traffic",
        "lookback": LOOKBACK_LEN,
        "pred_len": PRED_LEN,
        "model": model_label,
        "seed": "mean±std",
        "num_params": num_params,
        "best_epoch": round(float(np.mean(epochs)), 2),
        "best_val_mse": round(float(np.mean(vals)), 6),
        "mse": round(float(np.mean(mses)), 6),
        "mse_std": round(float(np.std(mses)), 6),
        "mae": round(float(np.mean(maes)), 6),
        "mae_std": round(float(np.std(maes)), 6),
        "latency_ms": round(float(np.mean(lats)), 4),
        "latency_std": round(float(np.std(lats)), 4),
        "peak_mem_mb": round(float(np.mean(mems)), 2),
        "pct_p8": p8,
        "pct_p16": p16,
        "pct_p32": p32,
        "pct_p64": p64,
    }


def write_csv(rows, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()


def compute_pareto_frontier(points):
    """Return sorted subset of (x, y, label) points not dominated in both lower x and lower y."""
    sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
    frontier = []
    best_y = float("inf")
    for x, y, label in sorted_pts:
        if y < best_y:
            frontier.append((x, y, label))
            best_y = y
    return frontier


def plot_pareto(summary_rows, fig_path):
    """Generate publication-ready Test MSE vs. Inference Latency Pareto frontier plot."""
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#333333"})
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    styles = {
        "PatchTST (P=16)": {"color": "#1f77b4", "marker": "o", "offset": (0, -26), "ha": "center"},
        "PatchTST (P=32)": {"color": "#2ca02c", "marker": "s", "offset": (10, -12), "ha": "left"},
        "iTransformer": {"color": "#9467bd", "marker": "^", "offset": (10, 6), "ha": "left"},
        "FRPPatchTST (Heuristic)": {"color": "#ff7f0e", "marker": "D", "offset": (-12, -8), "ha": "right"},
        "FRPPatchTST (Gumbel)": {"color": "#d62728", "marker": "*", "offset": (-12, 10), "ha": "right"},
    }

    pareto_candidates = []
    for r in summary_rows:
        label = r["model"]
        x = float(r["latency_ms"])
        x_err = float(r["latency_std"])
        y = float(r["mse"])
        y_err = float(r["mse_std"])
        mem = float(r["peak_mem_mb"])
        st = styles.get(label, {"color": "#333333", "marker": "o", "offset": (8, 8)})
        ms = 14 if st["marker"] == "*" else 9

        ax.errorbar(
            x,
            y,
            xerr=x_err,
            yerr=y_err,
            fmt=st["marker"],
            color=st["color"],
            markersize=ms,
            capsize=4,
            elinewidth=1.3,
            markeredgecolor="black",
            markeredgewidth=0.7,
            label=f"{label} ({mem:.0f} MB)" if mem > 0 else label,
            zorder=4,
        )

        dx, dy = st["offset"]
        ax.annotate(
            f"{label}\nMSE={y:.4f}",
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=st.get("ha", "left"),
            fontsize=8.5,
            fontweight="bold" if "Gumbel" in label else "normal",
            color=st["color"],
        )
        pareto_candidates.append((x, y, label))

    frontier = compute_pareto_frontier(pareto_candidates)
    if len(frontier) >= 2:
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        ax.plot(
            fx,
            fy,
            linestyle="--",
            linewidth=1.8,
            color="#444444",
            alpha=0.85,
            label="Pareto Frontier",
            zorder=2,
        )

    ax.set_title("PEMS04 Traffic (862 Variates, 96→96): Accuracy vs. Inference Latency Pareto", pad=12)
    ax.set_xlabel("Mean Inference Latency (ms / batch, batch_size=8)")
    ax.set_ylabel("Test MSE (lower is better)")
    ax.margins(x=0.10, y=0.12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", frameon=True, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved Pareto plot -> {fig_path}", flush=True)


def print_markdown_summary(summary_rows):
    print("\n## Block 3: Traffic (862 Variates, 96 -> 96) Benchmark & Pareto Summary\n", flush=True)
    print(
        "| Model | Params | Test MSE | Test MAE | Latency (ms/batch) | Peak VRAM (MB) | Routing [8/16/32/64] |",
        flush=True,
    )
    print(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        flush=True,
    )
    for r in summary_rows:
        mse_str = f"{float(r['mse']):.4f} ± {float(r['mse_std']):.4f}"
        mae_str = f"{float(r['mae']):.4f} ± {float(r['mae_std']):.4f}"
        lat_str = f"{float(r['latency_ms']):.2f} ± {float(r['latency_std']):.2f}"
        route_str = (
            f"{r['pct_p8']}/{r['pct_p16']}/{r['pct_p32']}/{r['pct_p64']}%"
            if r["pct_p8"] != ""
            else "Fixed / N/A"
        )
        print(
            f"| {r['model']} | {int(r['num_params']):,} | {mse_str} | {mae_str} | "
            f"{lat_str} | {float(r['peak_mem_mb']):.1f} | {route_str} |",
            flush=True,
        )


def run_traffic_pareto(dry_run=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Running Block 3 Traffic (862 Variates) Pareto Benchmark on device: {device} "
        f"(dry_run={dry_run})",
        flush=True,
    )

    config_path = os.path.join(PROJECT_ROOT, "configs/entropy_thresholds.json")
    csv_path = os.path.join(PROJECT_ROOT, "data/traffic.csv")

    if dry_run:
        output_csv = os.path.join(PROJECT_ROOT, "results/traffic_pareto_dryrun.csv")
        fig_path = os.path.join(PROJECT_ROOT, "figures/pareto_latency_mse_dryrun.png")
        ckpt_target = os.path.join(PROJECT_ROOT, "checkpoints/frp_gumbel_traffic_s42_dryrun.pt")
        seeds = [42]
        max_epochs = 1
        patience = 1
        warmup_batches = 1
        max_train_batches = 1
        max_val_batches = 1
        max_test_batches = 1
    else:
        output_csv = os.path.join(PROJECT_ROOT, "results/traffic_pareto.csv")
        fig_path = os.path.join(PROJECT_ROOT, "figures/pareto_latency_mse.png")
        ckpt_target = os.path.join(PROJECT_ROOT, "checkpoints/frp_gumbel_traffic_s42.pt")
        seeds = [42, 123, 999]
        max_epochs = 10
        patience = 3
        warmup_batches = 10
        max_train_batches = None
        max_val_batches = None
        max_test_batches = None

    train_ds = ETTDataset(csv_path, lookback_len=LOOKBACK_LEN, pred_len=PRED_LEN, split="train")
    val_ds = ETTDataset(
        csv_path, lookback_len=LOOKBACK_LEN, pred_len=PRED_LEN, split="val", scaler=train_ds.scaler
    )
    test_ds = ETTDataset(
        csv_path, lookback_len=LOOKBACK_LEN, pred_len=PRED_LEN, split="test", scaler=train_ds.scaler
    )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    models_order = [
        "PatchTST (P=16)",
        "PatchTST (P=32)",
        "iTransformer",
        "FRPPatchTST (Heuristic)",
        "FRPPatchTST (Gumbel)",
    ]

    cached_seeds = {} if dry_run else load_existing_seed_rows(output_csv)
    rows_by_model = {m: [] for m in models_order}

    def rebuild_and_save():
        all_rows = []
        summary_rows = []
        for m in models_order:
            m_seeds = rows_by_model[m]
            all_rows.extend(m_seeds)
            if len(m_seeds) == len(seeds):
                s_row = compute_summary_row(m, m_seeds)
                all_rows.append(s_row)
                summary_rows.append(s_row)
        write_csv(all_rows, output_csv)
        return summary_rows

    for model_label in models_order:
        for seed in seeds:
            ckpt_path = (
                ckpt_target
                if (model_label == "FRPPatchTST (Gumbel)" and seed == 42)
                else None
            )
            need_ckpt = ckpt_path is not None and not os.path.exists(ckpt_path)

            if (model_label, seed) in cached_seeds and not need_ckpt:
                row = cached_seeds[(model_label, seed)]
                rows_by_model[model_label].append(row)
                print(
                    f"[RESUME] {model_label} | seed={seed} | MSE={row['mse']} | Latency={row['latency_ms']} ms",
                    flush=True,
                )
                continue

            t0 = time.time()
            row = train_and_eval_seed(
                model_label=model_label,
                seed=seed,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                config_path=config_path,
                ckpt_path=ckpt_path,
                max_epochs=max_epochs,
                patience=patience,
                grad_accum_steps=4,
                lr=1e-4,
                aux_loss_weight=0.01,
                warmup_batches=warmup_batches,
                max_train_batches=max_train_batches,
                max_val_batches=max_val_batches,
                max_test_batches=max_test_batches,
            )
            rows_by_model[model_label].append(row)
            summary_rows = rebuild_and_save()
            elapsed = time.time() - t0
            print(
                f"[{model_label} | seed={seed}] ep={row['best_epoch']} | val={row['best_val_mse']} | "
                f"MSE={row['mse']} | MAE={row['mae']} | lat={row['latency_ms']} ms | "
                f"VRAM={row['peak_mem_mb']} MB ({elapsed:.1f}s)",
                flush=True,
            )

    summary_rows = rebuild_and_save()
    print_markdown_summary(summary_rows)
    plot_pareto(summary_rows, fig_path)

    if dry_run:
        assert os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
        assert os.path.exists(fig_path) and os.path.getsize(fig_path) > 0
        assert os.path.exists(ckpt_target) and os.path.getsize(ckpt_target) > 0
        for p in (output_csv, fig_path, ckpt_target):
            os.remove(p)
        print("Dry-run assertions passed and temporary dry-run artifacts cleaned up.", flush=True)
    else:
        print(f"Saved full benchmark results to {output_csv}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Block 3: Full 862-Variate Traffic Benchmark & Pareto Curves"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a fast 1-epoch, multi-batch verification check across all 5 models without modifying real results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_traffic_pareto(dry_run=args.dry_run)
