import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import copy
import csv
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
from models.PatchTST import PatchTST


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

LATENCY_FIELDNAMES = [
    "mode",
    "device",
    "num_variates",
    "batch_size",
    "timed_batches",
    "latency_ms_mean",
    "latency_ms_std",
    "peak_mem_mb",
    "speedup_vs_dense",
    "latency_reduction_pct",
    "max_abs_diff_vs_dense",
    "test_mse",
    "test_mae",
]

SEEDS = [42, 123, 999]
TASKS = [
    ("ETTh1", 96, 96),
    ("ETTh1", 96, 192),
    ("ETTh1", 336, 96),
    ("ETTm1", 96, 96),
    ("ETTm1", 336, 96),
]
PATCH_SIZES = [8, 16, 32, 64]


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
    return sq_err / max(1, total_elems)


def evaluate_test(model, loader, device, fixed_patch_size=None):
    """Compute exact test MSE, MAE, and routing distribution [P8, P16, P32, P64]."""
    model.eval()
    sq_err = 0.0
    abs_err = 0.0
    total_elems = 0
    is_frp = isinstance(model, FRPPatchTST)
    routing_counts = torch.zeros(len(PATCH_SIZES), dtype=torch.float64)
    total_windows = 0

    with torch.no_grad():
        for seq_x, seq_y in loader:
            if is_frp:
                pred, weights = model(seq_x.to(device), return_routing=True)
                routing_counts += weights.detach().cpu().double().sum(dim=0)
                total_windows += weights.shape[0]
            else:
                pred = model(seq_x.to(device))
            target = seq_y.to(device)
            diff = pred - target
            sq_err += (diff ** 2).sum().item()
            abs_err += diff.abs().sum().item()
            total_elems += target.numel()

    mse = sq_err / max(1, total_elems)
    mae = abs_err / max(1, total_elems)
    if is_frp:
        pcts = (routing_counts / max(1, total_windows) * 100.0).tolist()
    else:
        pcts = [100.0 if p == fixed_patch_size else 0.0 for p in PATCH_SIZES]
    return mse, mae, pcts


def train_and_eval_seed(
    model_builder,
    seed,
    train_loader,
    val_loader,
    test_loader,
    device,
    fixed_patch_size=None,
    max_epochs=10,
    patience=3,
    lr=1e-4,
):
    """Train one seed with early stopping and evaluate the best validation checkpoint on test."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = model_builder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_mse = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0

    for ep in range(max_epochs):
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_mse, test_mae, routing_pcts = evaluate_test(
        model, test_loader, device, fixed_patch_size=fixed_patch_size
    )
    return {
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "mse": test_mse,
        "mae": test_mae,
        "pcts": routing_pcts,
    }


def make_seed_and_summary_rows(dataset_name, lookback, pred_len, model_label, seed_results):
    """Format 3 seed rows and 1 summary (mean±std) row matching frp_ett_benchmarks.csv."""
    rows = []
    for seed, res in zip(SEEDS, seed_results):
        rows.append(
            {
                "row_type": "seed",
                "dataset": dataset_name,
                "lookback": str(lookback),
                "pred_len": str(pred_len),
                "model": model_label,
                "seed": str(seed),
                "best_epoch": str(res["best_epoch"]),
                "best_val_mse": f"{round(float(res['best_val_mse']), 6)}",
                "mse": f"{round(float(res['mse']), 6)}",
                "mse_std": "",
                "mae": f"{round(float(res['mae']), 6)}",
                "mae_std": "",
                "pct_p8": f"{round(float(res['pcts'][0]), 2)}",
                "pct_p16": f"{round(float(res['pcts'][1]), 2)}",
                "pct_p32": f"{round(float(res['pcts'][2]), 2)}",
                "pct_p64": f"{round(float(res['pcts'][3]), 2)}",
            }
        )

    mses = [float(r["mse"]) for r in seed_results]
    maes = [float(r["mae"]) for r in seed_results]
    val_mses = [float(r["best_val_mse"]) for r in seed_results]
    pct_means = np.array([r["pcts"] for r in seed_results], dtype=np.float64).mean(axis=0).tolist()

    summary_row = {
        "row_type": "summary",
        "dataset": dataset_name,
        "lookback": str(lookback),
        "pred_len": str(pred_len),
        "model": model_label,
        "seed": "mean±std",
        "best_epoch": f"{round(float(np.mean([float(r['best_epoch']) for r in seed_results])), 2)}",
        "best_val_mse": f"{round(float(np.mean(val_mses)), 6)}",
        "mse": f"{round(float(np.mean(mses)), 6)}",
        "mse_std": f"{round(float(np.std(mses)), 6)}",
        "mae": f"{round(float(np.mean(maes)), 6)}",
        "mae_std": f"{round(float(np.std(maes)), 6)}",
        "pct_p8": f"{round(float(pct_means[0]), 2)}",
        "pct_p16": f"{round(float(pct_means[1]), 2)}",
        "pct_p32": f"{round(float(pct_means[2]), 2)}",
        "pct_p64": f"{round(float(pct_means[3]), 2)}",
    }
    rows.append(summary_row)
    return rows, summary_row


def write_csv(rows, fieldnames, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()


def load_existing_ablation_cache(csv_path):
    """Load completed (dataset, lookback, pred_len, model) blocks from routing_ablation.csv."""
    if not os.path.exists(csv_path):
        return {}
    grouped = {}
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], int(row["lookback"]), int(row["pred_len"]), row["model"])
            grouped.setdefault(key, []).append(row)

    valid = {}
    for key, rows in grouped.items():
        seeds_present = {r["seed"] for r in rows if r["row_type"] == "seed"}
        has_summary = any(r["row_type"] == "summary" for r in rows)
        if seeds_present == {"42", "123", "999"} and has_summary and len(rows) == 4:
            valid[key] = rows
    return valid


def load_reference_rows(csv_path, source_model_label, target_model_label):
    """Load existing ETT benchmark blocks for reuse in the routing ablation."""
    grouped = {}
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            if row["model"] != source_model_label:
                continue
            row_copy = dict(row)
            row_copy["model"] = target_model_label
            key = (row_copy["dataset"], int(row_copy["lookback"]), int(row_copy["pred_len"]))
            grouped.setdefault(key, []).append(row_copy)
    return grouped


def run_ett_ablation(fresh=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running ETT Routing & Fixed-P* Ablation on device: {device}", flush=True)

    config_path = os.path.join(PROJECT_ROOT, "configs/entropy_thresholds.json")
    output_csv = os.path.join(PROJECT_ROOT, "results/routing_ablation.csv")
    bal_csv = os.path.join(PROJECT_ROOT, "results/frp_ett_benchmarks.csv")
    unreg_csv = os.path.join(PROJECT_ROOT, "results/frp_ett_benchmarks_unregularized.csv")

    cached = {} if fresh else load_existing_ablation_cache(output_csv)
    heur_ref = load_reference_rows(bal_csv, "FRPPatchTST (Heuristic)", "FRPPatchTST (Heuristic)")
    bal_ref = load_reference_rows(bal_csv, "FRPPatchTST (Gumbel)", "FRPPatchTST (Gumbel)")
    unreg_ref = load_reference_rows(unreg_csv, "FRPPatchTST (Gumbel)", "FRPPatchTST (Unreg. Gumbel)")

    all_rows = []
    summary_rows = []

    for dataset_name, L, S in TASKS:
        csv_path = os.path.join(PROJECT_ROOT, f"data/{dataset_name}.csv")
        train_ds = ETTDataset(csv_path, lookback_len=L, pred_len=S, split="train")
        val_ds = ETTDataset(csv_path, lookback_len=L, pred_len=S, split="val", scaler=train_ds.scaler)
        test_ds = ETTDataset(csv_path, lookback_len=L, pred_len=S, split="test", scaler=train_ds.scaler)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        fixed_seed_results = {p: [] for p in PATCH_SIZES}

        for P in PATCH_SIZES:
            model_label = f"PatchTST (P={P})"
            key = (dataset_name, L, S, model_label)
            if key in cached:
                block = cached[key]
                all_rows.extend(block)
                s_row = [r for r in block if r["row_type"] == "summary"][0]
                summary_rows.append(s_row)
                for r in block:
                    if r["row_type"] == "seed":
                        fixed_seed_results[P].append(
                            {
                                "best_epoch": int(r["best_epoch"]),
                                "best_val_mse": float(r["best_val_mse"]),
                                "mse": float(r["mse"]),
                                "mae": float(r["mae"]),
                                "pcts": [float(r["pct_p8"]), float(r["pct_p16"]), float(r["pct_p32"]), float(r["pct_p64"])],
                            }
                        )
                print(f"[{dataset_name} {L}->{S}] {model_label:26s} | CACHED MSE: {float(s_row['mse']):.4f} ± {float(s_row['mse_std']):.4f}", flush=True)
                continue

            t0 = time.time()
            seed_res = []
            for seed in SEEDS:
                res = train_and_eval_seed(
                    model_builder=lambda p=P: PatchTST(
                        num_variates=7, lookback_len=L, pred_len=S, patch_len=p, stride=p // 2, d_model=128
                    ),
                    seed=seed,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    fixed_patch_size=P,
                )
                seed_res.append(res)
            fixed_seed_results[P] = seed_res
            block, s_row = make_seed_and_summary_rows(dataset_name, L, S, model_label, seed_res)
            all_rows.extend(block)
            summary_rows.append(s_row)
            write_csv(all_rows, CSV_FIELDNAMES, output_csv)
            print(
                f"[{dataset_name} {L}->{S}] {model_label:26s} | "
                f"MSE: {float(s_row['mse']):.4f} ± {float(s_row['mse_std']):.4f} | "
                f"MAE: {float(s_row['mae']):.4f} ± {float(s_row['mae_std']):.4f} ({time.time() - t0:.1f}s)",
                flush=True,
            )

        # Val-Selected Fixed P* per seed
        pstar_label = "Val-Selected Fixed P*"
        pstar_key = (dataset_name, L, S, pstar_label)
        if pstar_key in cached:
            block = cached[pstar_key]
            all_rows.extend(block)
            s_row = [r for r in block if r["row_type"] == "summary"][0]
            summary_rows.append(s_row)
            print(f"[{dataset_name} {L}->{S}] {pstar_label:26s} | CACHED MSE: {float(s_row['mse']):.4f} ± {float(s_row['mse_std']):.4f}", flush=True)
        else:
            pstar_seed_res = []
            for idx in range(len(SEEDS)):
                best_p = min(PATCH_SIZES, key=lambda p: fixed_seed_results[p][idx]["best_val_mse"])
                pstar_seed_res.append(copy.deepcopy(fixed_seed_results[best_p][idx]))
            block, s_row = make_seed_and_summary_rows(dataset_name, L, S, pstar_label, pstar_seed_res)
            all_rows.extend(block)
            summary_rows.append(s_row)
            write_csv(all_rows, CSV_FIELDNAMES, output_csv)
            print(
                f"[{dataset_name} {L}->{S}] {pstar_label:26s} | "
                f"MSE: {float(s_row['mse']):.4f} ± {float(s_row['mse_std']):.4f} | "
                f"MAE: {float(s_row['mae']):.4f} ± {float(s_row['mae_std']):.4f} | "
                f"P* Dist [8/16/32/64]: {s_row['pct_p8']}/{s_row['pct_p16']}/{s_row['pct_p32']}/{s_row['pct_p64']}%",
                flush=True,
            )

        # FRPPatchTST (Random)
        rand_label = "FRPPatchTST (Random)"
        rand_key = (dataset_name, L, S, rand_label)
        if rand_key in cached:
            block = cached[rand_key]
            all_rows.extend(block)
            s_row = [r for r in block if r["row_type"] == "summary"][0]
            summary_rows.append(s_row)
            print(f"[{dataset_name} {L}->{S}] {rand_label:26s} | CACHED MSE: {float(s_row['mse']):.4f} ± {float(s_row['mse_std']):.4f}", flush=True)
        else:
            t0 = time.time()
            rand_seed_res = []
            for seed in SEEDS:
                res = train_and_eval_seed(
                    model_builder=lambda: FRPPatchTST.from_config(
                        config_path,
                        dataset_name,
                        lookback_len=L,
                        num_variates=7,
                        pred_len=S,
                        router_type="random",
                    ),
                    seed=seed,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                )
                rand_seed_res.append(res)
            block, s_row = make_seed_and_summary_rows(dataset_name, L, S, rand_label, rand_seed_res)
            all_rows.extend(block)
            summary_rows.append(s_row)
            write_csv(all_rows, CSV_FIELDNAMES, output_csv)
            print(
                f"[{dataset_name} {L}->{S}] {rand_label:26s} | "
                f"MSE: {float(s_row['mse']):.4f} ± {float(s_row['mse_std']):.4f} | "
                f"MAE: {float(s_row['mae']):.4f} ± {float(s_row['mae_std']):.4f} ({time.time() - t0:.1f}s)",
                flush=True,
            )

        # Reuse Heuristic, Unreg. Gumbel, and Balanced Gumbel from Block 2 CSVs
        for ref_dict, ref_label in [
            (heur_ref, "FRPPatchTST (Heuristic)"),
            (unreg_ref, "FRPPatchTST (Unreg. Gumbel)"),
            (bal_ref, "FRPPatchTST (Gumbel)"),
        ]:
            block = ref_dict[(dataset_name, L, S)]
            all_rows.extend(block)
            s_row = [r for r in block if r["row_type"] == "summary"][0]
            summary_rows.append(s_row)
            write_csv(all_rows, CSV_FIELDNAMES, output_csv)
            print(
                f"[{dataset_name} {L}->{S}] {ref_label:26s} | "
                f"LOADED MSE: {float(s_row['mse']):.4f} ± {float(s_row['mse_std']):.4f}",
                flush=True,
            )

    print(f"\nSaved complete routing ablation to {output_csv}", flush=True)
    return summary_rows


def _time_model_batches(model, batches, device, dense_eval=False, warmup=10):
    use_cuda = device.type == "cuda"
    model.eval()
    with torch.no_grad():
        for i in range(min(warmup, len(batches))):
            x = batches[i].to(device)
            if isinstance(model, FRPPatchTST):
                _ = model(x, dense_eval=dense_eval)
            else:
                _ = model(x)

        if use_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)

        latencies = []
        for x_cpu in batches:
            x = x_cpu.to(device)
            if use_cuda:
                start_ev.record()
                if isinstance(model, FRPPatchTST):
                    _ = model(x, dense_eval=dense_eval)
                else:
                    _ = model(x)
                end_ev.record()
                torch.cuda.synchronize(device)
                latencies.append(start_ev.elapsed_time(end_ev))
            else:
                t0 = time.perf_counter()
                if isinstance(model, FRPPatchTST):
                    _ = model(x, dense_eval=dense_eval)
                else:
                    _ = model(x)
                latencies.append((time.perf_counter() - t0) * 1000.0)

        peak_mem_mb = (
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2) if use_cuda else 0.0
        )

    return float(np.mean(latencies)), float(np.std(latencies)), float(peak_mem_mb)


def benchmark_traffic_sparse_latency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(PROJECT_ROOT, "configs/entropy_thresholds.json")
    traffic_csv = os.path.join(PROJECT_ROOT, "data/traffic.csv")
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints/frp_gumbel_traffic_s42.pt")
    out_csv = os.path.join(PROJECT_ROOT, "results/traffic_sparse_latency.csv")

    train_ds = ETTDataset(traffic_csv, lookback_len=96, pred_len=96, split="train")
    test_ds = ETTDataset(traffic_csv, lookback_len=96, pred_len=96, split="test", scaler=train_ds.scaler)
    assert test_ds.data.shape[1] == 862, f"Expected 862 variates, got {test_ds.data.shape[1]}"

    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = FRPPatchTST.from_config(
        config_path,
        "traffic",
        lookback_len=96,
        num_variates=862,
        pred_len=96,
        router_type="gumbel",
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Check equivalence on first 10 batches and full test MSE/MAE in sparse eval mode
    max_diff = 0.0
    sq_err = 0.0
    abs_err = 0.0
    total_elems = 0
    timed_batches = []
    with torch.no_grad():
        for idx, (seq_x, seq_y) in enumerate(test_loader):
            if idx < 50:
                timed_batches.append(seq_x)
            x_dev = seq_x.to(device)
            y_dev = seq_y.to(device)
            pred_sparse = model(x_dev, dense_eval=False)
            if idx < 10:
                pred_dense = model(x_dev, dense_eval=True)
                diff_val = (pred_sparse - pred_dense).abs().max().item()
                if diff_val > max_diff:
                    max_diff = diff_val
            diff = pred_sparse - y_dev
            sq_err += (diff ** 2).sum().item()
            abs_err += diff.abs().sum().item()
            total_elems += y_dev.numel()

    test_mse = sq_err / max(1, total_elems)
    test_mae = abs_err / max(1, total_elems)
    assert max_diff < 1e-4, f"Sparse vs dense mismatch on Traffic: {max_diff}"

    dense_mean, dense_std, dense_mem = _time_model_batches(model, timed_batches, device, dense_eval=True, warmup=10)
    sparse_mean, sparse_std, sparse_mem = _time_model_batches(model, timed_batches, device, dense_eval=False, warmup=10)

    p16_model = PatchTST(num_variates=862, lookback_len=96, pred_len=96, patch_len=16, stride=8, d_model=128).to(device)
    p32_model = PatchTST(num_variates=862, lookback_len=96, pred_len=96, patch_len=32, stride=16, d_model=128).to(device)
    p16_mean, p16_std, p16_mem = _time_model_batches(p16_model, timed_batches, device, warmup=10)
    p32_mean, p32_std, p32_mem = _time_model_batches(p32_model, timed_batches, device, warmup=10)

    speedup = dense_mean / max(sparse_mean, 1e-9)
    reduction_pct = (1.0 - sparse_mean / max(dense_mean, 1e-9)) * 100.0

    rows = [
        {
            "mode": "FRPPatchTST (Dense 4-Branch Eval)",
            "device": str(device),
            "num_variates": 862,
            "batch_size": 8,
            "timed_batches": len(timed_batches),
            "latency_ms_mean": round(dense_mean, 3),
            "latency_ms_std": round(dense_std, 3),
            "peak_mem_mb": round(dense_mem, 2),
            "speedup_vs_dense": 1.0,
            "latency_reduction_pct": 0.0,
            "max_abs_diff_vs_dense": 0.0,
            "test_mse": round(test_mse, 6),
            "test_mae": round(test_mae, 6),
        },
        {
            "mode": "FRPPatchTST (Sparse Single-Branch Eval)",
            "device": str(device),
            "num_variates": 862,
            "batch_size": 8,
            "timed_batches": len(timed_batches),
            "latency_ms_mean": round(sparse_mean, 3),
            "latency_ms_std": round(sparse_std, 3),
            "peak_mem_mb": round(sparse_mem, 2),
            "speedup_vs_dense": round(speedup, 3),
            "latency_reduction_pct": round(reduction_pct, 2),
            "max_abs_diff_vs_dense": f"{max_diff:.2e}",
            "test_mse": round(test_mse, 6),
            "test_mae": round(test_mae, 6),
        },
        {
            "mode": "PatchTST (P=16)",
            "device": str(device),
            "num_variates": 862,
            "batch_size": 8,
            "timed_batches": len(timed_batches),
            "latency_ms_mean": round(p16_mean, 3),
            "latency_ms_std": round(p16_std, 3),
            "peak_mem_mb": round(p16_mem, 2),
            "speedup_vs_dense": round(dense_mean / max(p16_mean, 1e-9), 3),
            "latency_reduction_pct": round((1.0 - p16_mean / max(dense_mean, 1e-9)) * 100.0, 2),
            "max_abs_diff_vs_dense": "",
            "test_mse": "",
            "test_mae": "",
        },
        {
            "mode": "PatchTST (P=32)",
            "device": str(device),
            "num_variates": 862,
            "batch_size": 8,
            "timed_batches": len(timed_batches),
            "latency_ms_mean": round(p32_mean, 3),
            "latency_ms_std": round(p32_std, 3),
            "peak_mem_mb": round(p32_mem, 2),
            "speedup_vs_dense": round(dense_mean / max(p32_mean, 1e-9), 3),
            "latency_reduction_pct": round((1.0 - p32_mean / max(dense_mean, 1e-9)) * 100.0, 2),
            "max_abs_diff_vs_dense": "",
            "test_mse": "",
            "test_mae": "",
        },
    ]
    write_csv(rows, LATENCY_FIELDNAMES, out_csv)

    print("\n## Traffic 862-Variate Sparse vs. Dense Inference Latency\n", flush=True)
    print(f"Checkpoint Test MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f} | Max Abs Diff: {max_diff:.2e}", flush=True)
    for r in rows:
        print(
            f"  {r['mode']:42s} | {r['latency_ms_mean']:8.3f} ± {r['latency_ms_std']:6.3f} ms/batch "
            f"| Peak VRAM: {r['peak_mem_mb']:6.2f} MB | Speedup: {r['speedup_vs_dense']}x ({r['latency_reduction_pct']}% reduction)",
            flush=True,
        )
    print(f"Saved Traffic sparse latency benchmark to {out_csv}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ETT routing ablation and Traffic sparse latency benchmark.")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing results/routing_ablation.csv and rerun all ablation models.")
    parser.add_argument("--latency-only", action="store_true", help="Run only the Traffic sparse vs. dense latency benchmark.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.latency_only:
        run_ett_ablation(fresh=args.fresh)
    benchmark_traffic_sparse_latency()
