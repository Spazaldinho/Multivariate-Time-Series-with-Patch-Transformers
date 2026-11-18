import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_loader import ETTDataset
from models.FRP_PatchTST import compute_spectral_entropy


def calibrate_entropy(output_path="configs/entropy_thresholds.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Spectral Entropy Calibration on device: {device}")

    datasets = [
        ("ETTh1", os.path.join(PROJECT_ROOT, "data/ETTh1.csv"), 256),
        ("ETTm1", os.path.join(PROJECT_ROOT, "data/ETTm1.csv"), 256),
        ("traffic", os.path.join(PROJECT_ROOT, "data/traffic.csv"), 128),
    ]
    lookbacks = [96, 336, 720]

    results = {}
    header = (
        f"{'Dataset':<10} | {'L':>4} | {'q25':>8} | {'q50':>8} | {'q75':>8} | "
        f"{'Mean ± Std':>19} | {'Total Windows':>13}"
    )
    print("\n" + header)
    print("-" * len(header))

    with torch.no_grad():
        for dataset_name, csv_path, batch_size in datasets:
            results[dataset_name] = {}
            for L in lookbacks:
                train_ds = ETTDataset(
                    csv_path, lookback_len=L, pred_len=96, split="train"
                )
                train_loader = DataLoader(
                    train_ds, batch_size=batch_size, shuffle=False
                )

                entropies = []
                for seq_x, _ in train_loader:
                    seq_x = seq_x.to(device)
                    x_ci = seq_x.permute(0, 2, 1).reshape(-1, L)
                    ent = compute_spectral_entropy(x_ci).squeeze(-1)
                    entropies.append(ent.cpu().numpy())

                all_entropies = np.concatenate(entropies, axis=0)
                if np.isnan(all_entropies).any():
                    raise ValueError(
                        f"NaN detected in spectral entropy for {dataset_name} (L={L})"
                    )
                if not ((all_entropies >= 0.0).all() and (all_entropies <= 1.0).all()):
                    raise ValueError(
                        f"Out-of-range spectral entropy for {dataset_name} (L={L}): "
                        f"[{all_entropies.min()}, {all_entropies.max()}]"
                    )

                q25, q50, q75 = np.quantile(all_entropies, [0.25, 0.50, 0.75])
                mean = float(np.mean(all_entropies))
                std = float(np.std(all_entropies))
                num_windows = int(all_entropies.size)

                results[dataset_name][str(L)] = {
                    "q25": round(float(q25), 6),
                    "q50": round(float(q50), 6),
                    "q75": round(float(q75), 6),
                    "mean": round(mean, 6),
                    "std": round(std, 6),
                    "num_windows": num_windows,
                }

                mean_std_str = f"{mean:.4f} ± {std:.4f}"
                print(
                    f"{dataset_name:<10} | {L:>4} | {q25:>8.4f} | {q50:>8.4f} | "
                    f"{q75:>8.4f} | {mean_std_str:>19} | {num_windows:>13,}"
                )

    out_abs = (
        output_path
        if os.path.isabs(output_path)
        else os.path.join(PROJECT_ROOT, output_path)
    )
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    with open(out_abs, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved calibrated entropy thresholds to {output_path}")
    return results


if __name__ == "__main__":
    calibrate_entropy()
