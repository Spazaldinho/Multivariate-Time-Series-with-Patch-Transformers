import copy
import csv
import os
import sys
import time
import warnings

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


CSV_FIELDNAMES = [
    "row_type",
    "dataset",
    "lookback",
    "pred_len",
    "model",
    "seed",
    "best_epoch",
    "best_val_mse",
    "mse",
    "mse_std",
    "mae",
    "mae_std",
    "pct_p8",
    "pct_p16",
    "pct_p32",
    "pct_p64",
]


def evaluate_mse(model, loader, device):
    """Compute exact dataset-level MSE on a DataLoader."""
    model.eval()
    sq_err = 0.0
    total_elems = 0
    with torch.no_grad():
        for seq_x, seq_y in loader:
            pred = model(seq_x.to(device))
            target = seq_y.to(device)
            sq_err += ((pred - target) ** 2).sum().item()
            total_elems += target.numel()
    return sq_err / total_elems


def evaluate_test(model, loader, device):
    """Compute exact test MSE, MAE, and routing distribution [P8, P16, P32, P64]."""
    model.eval()
    sq_err = 0.0
    abs_err = 0.0
    total_elems = 0
    routing_counts = torch.zeros(len(model.patch_sizes), dtype=torch.float64)
    total_windows = 0

    with torch.no_grad():
        for seq_x, seq_y in loader:
            pred, weights = model(seq_x.to(device), return_routing=True)
            target = seq_y.to(device)
            diff = pred - target
            sq_err += (diff ** 2).sum().item()
            abs_err += diff.abs().sum().item()
            total_elems += target.numel()
            routing_counts += weights.detach().cpu().double().sum(dim=0)
            total_windows += weights.shape[0]

    mse = sq_err / total_elems
    mae = abs_err / total_elems
    pcts = (routing_counts / max(1, total_windows) * 100.0).tolist()
    return mse, mae, pcts


def train_and_eval_seed(
    dataset_name,
    lookback,
    pred_len,
    router_type,
    seed,
    train_loader,
    val_loader,
    test_loader,
    device,
    config_path,
    max_epochs=10,
    patience=3,
    lr=1e-4,
):
    """Train one seed with early stopping and evaluate best validation checkpoint on test."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = FRPPatchTST.from_config(
        config_path,
        dataset_name,
        lookback_len=lookback,
        num_variates=7,
        pred_len=pred_len,
        router_type=router_type,
        tau=1.0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_mse = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for ep in range(max_epochs):
        tau = 1.0 - 0.5 * (ep / max(1, max_epochs - 1))
        model.set_temperature(tau)

        model.train()
        for seq_x, seq_y in train_loader:
            optimizer.zero_grad()
            pred = model(seq_x.to(device))
            loss = criterion(pred, seq_y.to(device))
            loss.backward()
            optimizer.step()

        val_mse = evaluate_mse(model, val_loader, device)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = ep + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    test_mse, test_mae, routing_pcts = evaluate_test(model, test_loader, device)
    return {
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "mse": test_mse,
        "mae": test_mae,
        "pcts": routing_pcts,
    }


def write_csv(rows, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()


def print_markdown_summary(summary_rows):
    print("\n## Block 2: FRPPatchTST ETT Benchmark Summary (Mean ± Std over Seeds 42, 123, 999)\n", flush=True)
    print(
        "| Dataset | L -> S | Model | MSE | MAE | %P8 | %P16 | %P32 | %P64 |",
        flush=True,
    )
    print(
        "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        flush=True,
    )
    for r in summary_rows:
        mse_str = f"{r['mse']:.4f} ± {r['mse_std']:.4f}"
        mae_str = f"{r['mae']:.4f} ± {r['mae_std']:.4f}"
        print(
            f"| {r['dataset']} | {r['lookback']} -> {r['pred_len']} | {r['model']} | "
            f"{mse_str} | {mae_str} | {r['pct_p8']:.1f}% | {r['pct_p16']:.1f}% | "
            f"{r['pct_p32']:.1f}% | {r['pct_p64']:.1f}% |",
            flush=True,
        )


def run_benchmarks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Block 2 FRPPatchTST ETT Benchmarks on device: {device}", flush=True)

    config_path = os.path.join(PROJECT_ROOT, "configs/entropy_thresholds.json")
    output_csv = os.path.join(PROJECT_ROOT, "results/frp_ett_benchmarks.csv")

    datasets = [
        ("ETTh1", os.path.join(PROJECT_ROOT, "data/ETTh1.csv")),
        ("ETTm1", os.path.join(PROJECT_ROOT, "data/ETTm1.csv")),
    ]
    tasks = [(96, 96), (96, 192), (96, 336), (96, 720), (336, 96)]
    variants = [
        ("FRPPatchTST (Heuristic)", "heuristic"),
        ("FRPPatchTST (Gumbel)", "gumbel"),
    ]
    seeds = [42, 123, 999]

    all_rows = []
    summary_rows = []

    for dataset_name, csv_path in datasets:
        for L, S in tasks:
            train_ds = ETTDataset(csv_path, lookback_len=L, pred_len=S, split="train")
            val_ds = ETTDataset(
                csv_path, lookback_len=L, pred_len=S, split="val", scaler=train_ds.scaler
            )
            test_ds = ETTDataset(
                csv_path, lookback_len=L, pred_len=S, split="test", scaler=train_ds.scaler
            )

            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

            for model_label, router_type in variants:
                t0 = time.time()
                seed_metrics = []
                for seed in seeds:
                    res = train_and_eval_seed(
                        dataset_name=dataset_name,
                        lookback=L,
                        pred_len=S,
                        router_type=router_type,
                        seed=seed,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        device=device,
                        config_path=config_path,
                    )
                    seed_metrics.append(res)
                    all_rows.append(
                        {
                            "row_type": "seed",
                            "dataset": dataset_name,
                            "lookback": L,
                            "pred_len": S,
                            "model": model_label,
                            "seed": seed,
                            "best_epoch": res["best_epoch"],
                            "best_val_mse": round(res["best_val_mse"], 6),
                            "mse": round(res["mse"], 6),
                            "mse_std": "",
                            "mae": round(res["mae"], 6),
                            "mae_std": "",
                            "pct_p8": round(res["pcts"][0], 2),
                            "pct_p16": round(res["pcts"][1], 2),
                            "pct_p32": round(res["pcts"][2], 2),
                            "pct_p64": round(res["pcts"][3], 2),
                        }
                    )

                mses = [m["mse"] for m in seed_metrics]
                maes = [m["mae"] for m in seed_metrics]
                val_mses = [m["best_val_mse"] for m in seed_metrics]
                pct_matrix = np.array([m["pcts"] for m in seed_metrics])
                pct_means = pct_matrix.mean(axis=0).tolist()

                summary_row = {
                    "row_type": "summary",
                    "dataset": dataset_name,
                    "lookback": L,
                    "pred_len": S,
                    "model": model_label,
                    "seed": "mean±std",
                    "best_epoch": round(float(np.mean([m["best_epoch"] for m in seed_metrics])), 2),
                    "best_val_mse": round(float(np.mean(val_mses)), 6),
                    "mse": round(float(np.mean(mses)), 6),
                    "mse_std": round(float(np.std(mses)), 6),
                    "mae": round(float(np.mean(maes)), 6),
                    "mae_std": round(float(np.std(maes)), 6),
                    "pct_p8": round(pct_means[0], 2),
                    "pct_p16": round(pct_means[1], 2),
                    "pct_p32": round(pct_means[2], 2),
                    "pct_p64": round(pct_means[3], 2),
                }
                all_rows.append(summary_row)
                summary_rows.append(summary_row)
                write_csv(all_rows, output_csv)

                elapsed = time.time() - t0
                print(
                    f"[{dataset_name} {L}->{S}] {model_label:24s} | "
                    f"MSE: {summary_row['mse']:.4f} ± {summary_row['mse_std']:.4f} | "
                    f"MAE: {summary_row['mae']:.4f} ± {summary_row['mae_std']:.4f} | "
                    f"Routing [8/16/32/64]: {pct_means[0]:.1f}/{pct_means[1]:.1f}/{pct_means[2]:.1f}/{pct_means[3]:.1f}% "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )

    print_markdown_summary(summary_rows)
    print(f"\nSaved full benchmark results to {output_csv}", flush=True)


if __name__ == "__main__":
    run_benchmarks()
