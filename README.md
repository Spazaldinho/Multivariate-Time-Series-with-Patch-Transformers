# Multivariate Long-Sequence Time Series Forecasting with Frequency-Routed Patch Transformers (`FRP-PatchTST`)

**Course:** CISC 473 — Deep Learning (Fall 2026)  
**Compute Target:** CASLab Lobot GPU Cluster (NVIDIA A16, 16GB VRAM)

---

## 1. Project Overview & Architecture Summary

Standard patch-based Time Series Transformers such as `PatchTST` tokenize every univariate lookback window using a single fixed patch length $P$ (typically $P=16$). However, real-world multivariate time series exhibit pronounced **temporal heterogeneity**:
* **Low-entropy, smooth periodic regimes** (e.g., stationary 24-hour diurnal cycles) are most concisely and accurately represented by **coarse patches** ($P \in \{32, 64\}$), which capture long-range phase structure without overfitting local noise.
* **High-entropy, volatile regimes** (e.g., sharp rush-hour traffic spikes or sudden load transients) require **fine patches** ($P \in \{8, 16\}$) to preserve high-frequency local dynamics.

### Proposed Solution: `FRPPatchTST`
We introduce **Frequency-Routed Patching PatchTST (`FRPPatchTST`)** in [`models/FRP_PatchTST.py`](models/FRP_PatchTST.py), which dynamically routes each univariate channel window to an appropriate temporal patch resolution $P \in \{8, 16, 32, 64\}$ ($\text{stride} = P / 2$):

1. **Reversible Instance Normalization (`RevIN`) & Strict Channel Independence:**  
   Input batches $[B, L, M]$ are instance-normalized along the temporal axis $L$ prior to patching, reshaped to $[B \cdot M, L]$ to process each variate independently without spurious cross-channel mixing, and denormalized after the projection head.
2. **DC-Invariant Shannon Spectral Entropy ($H \in [0, 1]$):**  
   Each window is demeaned ($x - \bar{x}$), transformed via real FFT (`torch.fft.rfft(..., dim=-1)[..., 1:]`) excluding the $f=0$ DC offset bin, converted into a normalized Power Spectral Density (PSD) distribution, and summarized via normalized Shannon Spectral Entropy.
3. **Multi-Resolution Shared-Backbone Architecture:**  
   Four resolution branches ($P \in \{8, 16, 32, 64\}$) maintain resolution-specific linear patch tokenizers, learnable positional embeddings, and linear prediction heads while sharing a single Transformer Encoder backbone ($d_{\text{model}}=128$, $d_{\text{ff}}=256$, 8 heads). At evaluation time, sparse branch dispatch executes only the selected resolution branch per window.
4. **Calibrated Heuristic & Load-Balanced Gumbel-Softmax Routers:**
   * **`FRPPatchTST (Heuristic)`:** Routes windows deterministically using empirical training-split spectral entropy quartiles $(q_{25}, q_{50}, q_{75})$ from [`configs/entropy_thresholds.json`](configs/entropy_thresholds.json) ($H \ge q_{75} \to P=8$, $q_{50} \le H < q_{75} \to P=16$, $q_{25} \le H < q_{50} \to P=32$, $H < q_{25} \to P=64$).
   * **`FRPPatchTST (Gumbel)`:** Standardizes window entropy by the calibrated interquartile range $\tilde{H} = (H - q_{50}) / (q_{75} - q_{25})$, combines learnable quartile anchors with a 2-layer MLP (`1 -> 32 -> 4`), and trains end-to-end via Straight-Through Gumbel-Softmax (`hard=True`) with linear temperature annealing ($\tau: 1.0 \to 0.5$) and a Switch-Transformer load-balancing auxiliary loss to prevent single-expert mode collapse.

---

## 2. Repository Structure

```text
Multivariate-Time-Series-with-Patch-Transformers/
├── README.md                                        # Project documentation & benchmark summary
├── requirements.txt                                 # Python dependencies
├── checkpoints/
│   └── frp_gumbel_traffic_s42.pt                    # Best validation checkpoint (Traffic, 862 variates, seed 42)
├── configs/
│   └── entropy_thresholds.json                      # Calibrated training-split spectral entropy quartiles (Block 1)
├── data/                                            # Raw CSV datasets (ETTh1.csv, ETTm1.csv, traffic.csv)
├── data_provider/
│   └── data_loader.py                               # Chronological 70/10/20 ETTDataset & leak-free StandardScaler
├── figures/
│   ├── pareto_latency_mse.png                       # Block 3: 862-variate Traffic MSE vs. Latency Pareto frontier
│   ├── traffic_routing_histogram.png                # Block 4: Test entropy histogram & 10-bin selection probability
│   └── smooth_vs_volatile_forecasts.png             # Block 4: Smooth (P=64) vs. volatile (P=8) sensor forecasts
├── models/
│   ├── Autoformer.py                                # Series decomposition & autocorrelation baseline
│   ├── PatchTST.py                                  # RevIN & fixed-resolution Channel-Independent PatchTST
│   ├── iTransformer.py                              # Inverted variate-token attention baseline
│   └── FRP_PatchTST.py                              # Frequency-Routed Patching PatchTST (Heuristic & Gumbel)
├── results/
│   ├── frp_ett_benchmarks.csv                       # Block 2: Load-balanced Gumbel vs. Heuristic on ETTh1 & ETTm1
│   ├── frp_ett_benchmarks_unregularized.csv         # Block 2: Unregularized Gumbel ablation (expert collapse study)
│   ├── traffic_pareto.csv                           # Block 3: 862-variate Traffic accuracy, latency & VRAM metrics
│   └── traffic_routing_analysis.csv                 # Block 4: 2.94M-window Traffic routing statistics & Spearman rho
├── scripts/
│   ├── calibrate_entropy.py                         # Block 1: Training-split spectral entropy quartile calibration
│   ├── run_frp_ett.py                               # Block 2: Multi-horizon ETTh1/ETTm1 benchmark & ablation runner
│   ├── run_traffic_pareto.py                        # Block 3: Full 862-variate Traffic benchmark & Pareto profiler
│   ├── generate_interpretability.py                 # Block 4: Mechanistic routing analysis & case-study visualizer
│   ├── generate_midterm_results.py                  # Midterm baseline evaluation script (Autoformer/PatchTST/iTransformer)
│   ├── generate_visualizations.py                   # Qualitative ETTh1 96->96 forecast plotter
│   ├── run_ablations.py                             # Fixed patch-size (P in {8,16,32,64}) ablation study
│   └── run_new_domain.py                            # Initial Traffic domain baseline evaluation script
└── tests/
    ├── test_data.py                                 # DataLoader shape, continuity & train-only scaling tests
    └── test_modules.py                              # RevIN, SeriesDecomp, spectral entropy & model gradient tests
```

---

## 3. End-to-End Reproduction Commands

### Environment Setup
```bash
pip install -r requirements.txt
# If raw CSVs are not yet extracted into data/:
unzip -n data/datasets.zip -d data/
```

### Sequential Execution Pipeline
```bash
# 1. Verify all 13 unit tests (DataLoader integrity, RevIN reversibility, spectral entropy bounds, model gradients)
python3 -m pytest -v

# 2. Block 1 — Empirical Spectral Entropy Calibration (writes configs/entropy_thresholds.json)
python3 scripts/calibrate_entropy.py

# 3. Block 2 — ETT Multi-Horizon & Long-Lookback Benchmarks (writes results/frp_ett_benchmarks.csv)
python3 scripts/run_frp_ett.py

# 4. Block 3 — Full 862-Variate Traffic Benchmark & Pareto Frontier (writes results/traffic_pareto.csv,
#    figures/pareto_latency_mse.png, and checkpoints/frp_gumbel_traffic_s42.pt)
python3 scripts/run_traffic_pareto.py

# 5. Block 4 — Mechanistic Interpretability & Forecast Case Studies (writes results/traffic_routing_analysis.csv,
#    figures/traffic_routing_histogram.png, and figures/smooth_vs_volatile_forecasts.png)
python3 scripts/generate_interpretability.py
```

---

## 4. Summary of Key Experimental Findings

### A. Block 1 — Empirical Spectral Entropy Calibration ([`configs/entropy_thresholds.json`](configs/entropy_thresholds.json))
Evaluated strictly on the chronological `70%` training split across all 9 `(dataset, lookback_len)` pairs:

| Dataset | Variates ($M$) | Lookback ($L$) | $q_{25}$ | $q_{50}$ (Median) | $q_{75}$ | Mean $\pm$ Std | Training Windows |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ETTh1` | 7 | 96 | 0.4588 | 0.5588 | 0.6437 | $0.5455 \pm 0.1395$ | 84,021 |
| `ETTh1` | 7 | 336 | 0.4492 | 0.5513 | 0.6355 | $0.5437 \pm 0.1289$ | 82,341 |
| `ETTh1` | 7 | 720 | 0.4500 | 0.5343 | 0.6255 | $0.5360 \pm 0.1224$ | 79,653 |
| `ETTm1` | 7 | 96 | 0.3072 | 0.4138 | 0.5238 | $0.4177 \pm 0.1596$ | 340,095 |
| `ETTm1` | 7 | 336 | 0.4176 | 0.4878 | 0.5503 | $0.4830 \pm 0.1040$ | 338,415 |
| `ETTm1` | 7 | 720 | 0.4202 | 0.4943 | 0.5562 | $0.4859 \pm 0.1026$ | 335,727 |
| `traffic` | 862 | 96 | 0.3336 | 0.4632 | 0.5780 | $0.4578 \pm 0.1583$ | 10,420,718 |
| `traffic` | 862 | 336 | 0.3585 | 0.4638 | 0.5510 | $0.4524 \pm 0.1378$ | 10,213,838 |
| `traffic` | 862 | 720 | 0.3614 | 0.4599 | 0.5429 | $0.4496 \pm 0.1322$ | 9,882,830 |

---

### B. Block 2 — ETT Benchmarks & Load-Balancing Ablation ([`results/frp_ett_benchmarks.csv`](results/frp_ett_benchmarks.csv))
* **Unregularized Gumbel Collapse ([`results/frp_ett_benchmarks_unregularized.csv`](results/frp_ett_benchmarks_unregularized.csv)):** Without IQR standardization and load balancing, Straight-Through Gumbel-Softmax collapsed 100% of test windows to a single patch size in `28/30` seeds (`>=99.9%` in `30/30` seeds), acting as a global dataset-level scale selector (`P=16/32` on `ETTh1`, `P=8` on `ETTm1 L=96`, `P=32/64` on `ETTm1 L=336`).
* **Load-Balanced Gumbel (`10/10` Win Over Heuristic):** Adding IQR standardization and Switch-Transformer load balancing completely eliminated mode collapse (`16.4% – 33.9%` per branch across all `30/30` seeds), beat `FRPPatchTST (Heuristic)` on **10 out of 10 tasks** (`-0.67%` to `-3.84%` Test MSE), and reduced `ETTm1 (96->96)` Test MSE by **`-6.99%`** over unregularized Gumbel with a **`27×` reduction in seed standard deviation** (`0.0189 -> 0.0007`):

| Dataset | `L -> S` | Heuristic Test MSE | Heuristic Test MAE | Unreg Gumbel MSE | **Balanced Gumbel MSE** | **Balanced Gumbel MAE** | Balanced Routing `[8/16/32/64]` | Bal vs. Heur |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ETTh1` | `96 -> 96` | $0.4486 \pm 0.0017$ | $0.4568 \pm 0.0023$ | $0.4339 \pm 0.0029$ | **$0.4372 \pm 0.0009$** | **$0.4493 \pm 0.0006$** | `23.1 / 17.8 / 26.8 / 32.2%` | **`-2.53%`** |
| `ETTh1` | `96 -> 192` | $0.5058 \pm 0.0025$ | $0.4953 \pm 0.0012$ | $0.4898 \pm 0.0032$ | **$0.4920 \pm 0.0029$** | **$0.4888 \pm 0.0012$** | `23.0 / 17.8 / 27.4 / 31.9%` | **`-2.74%`** |
| `ETTh1` | `96 -> 336` | $0.5588 \pm 0.0014$ | $0.5295 \pm 0.0001$ | $0.5560 \pm 0.0041$ | **$0.5510 \pm 0.0012$** | **$0.5259 \pm 0.0023$** | `23.1 / 17.8 / 27.2 / 31.9%` | **`-1.40%`** |
| `ETTh1` | `96 -> 720` | $0.7093 \pm 0.0012$ | $0.6211 \pm 0.0004$ | $0.7150 \pm 0.0066$ | **$0.7046 \pm 0.0033$** | **$0.6174 \pm 0.0019$** | `23.2 / 16.9 / 26.9 / 32.9%` | **`-0.67%`** |
| `ETTh1` | `336 -> 96` | $0.4423 \pm 0.0014$ | $0.4668 \pm 0.0004$ | $0.4297 \pm 0.0058$ | **$0.4287 \pm 0.0021$** | **$0.4554 \pm 0.0010$** | `21.0 / 25.4 / 19.9 / 33.7%` | **`-3.08%`** |
| `ETTm1` | `96 -> 96` | $0.3893 \pm 0.0001$ | $0.4059 \pm 0.0013$ | $0.4068 \pm 0.0189$ | **$0.3783 \pm 0.0007$** | **$0.4000 \pm 0.0008$** | `19.4 / 19.8 / 28.3 / 32.5%` | **`-2.81%`** |
| `ETTm1` | `96 -> 192` | $0.4300 \pm 0.0022$ | $0.4348 \pm 0.0012$ | $0.4400 \pm 0.0015$ | **$0.4249 \pm 0.0010$** | **$0.4317 \pm 0.0004$** | `19.6 / 19.6 / 28.4 / 32.4%` | **`-1.19%`** |
| `ETTm1` | `96 -> 336` | $0.4929 \pm 0.0008$ | $0.4726 \pm 0.0005$ | $0.4858 \pm 0.0037$ | **$0.4872 \pm 0.0011$** | **$0.4683 \pm 0.0012$** | `20.1 / 19.4 / 27.8 / 32.7%` | **`-1.15%`** |
| `ETTm1` | `96 -> 720` | $0.5576 \pm 0.0025$ | $0.5175 \pm 0.0015$ | $0.5570 \pm 0.0099$ | **$0.5492 \pm 0.0036$** | **$0.5124 \pm 0.0012$** | `20.3 / 18.9 / 28.4 / 32.4%` | **`-1.50%`** |
| `ETTm1` | `336 -> 96` | $0.3375 \pm 0.0037$ | $0.3870 \pm 0.0022$ | $0.3163 \pm 0.0010$ | **$0.3245 \pm 0.0035$** | **$0.3783 \pm 0.0021$** | `24.8 / 24.3 / 20.1 / 30.9%` | **`-3.84%`** |

---

### C. Block 3 — Full 862-Variate Traffic Benchmark & Pareto Frontier ([`results/traffic_pareto.csv`](results/traffic_pareto.csv), [`figures/pareto_latency_mse.png`](figures/pareto_latency_mse.png))
Evaluated on the full 862-variate PEMS04 Traffic dataset (`96 -> 96`, `batch_size=8`, `grad_accum_steps=4`, 3 seeds `42, 123, 999`) on an NVIDIA A16 GPU:

| Model | Params | Best Val MSE | Test MSE | Test MAE | Latency (`ms/batch`) | Peak VRAM | Routing `[8 / 16 / 32 / 64]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`iTransformer`** | 2,201,696 | **0.4009** | **$0.4372 \pm 0.0003$** | **$0.2975 \pm 0.0007$** | **$22.06 \pm 0.03$** | **140.2 MB** | Fixed / N/A |
| **`PatchTST (P=32)`** | 213,020 | 0.4444 | $0.5148 \pm 0.0005$ | $0.3359 \pm 0.0007$ | $37.33 \pm 0.01$ | 159.7 MB | Fixed / N/A |
| **`PatchTST (P=16)`** | 285,468 | 0.4428 | $0.5182 \pm 0.0010$ | $0.3348 \pm 0.0009$ | $45.67 \pm 0.01$ | 282.3 MB | Fixed / N/A |
| **`FRPPatchTST (Heuristic)`** | 709,384 | 0.4617 | $0.5477 \pm 0.0014$ | $0.3455 \pm 0.0010$ | $48.85 \pm 0.07$ | 277.2 MB | `27.8 / 24.9 / 24.1 / 23.2%` |
| **`FRPPatchTST (Gumbel)`** | 709,384 | 0.4630 | $0.5514 \pm 0.0024$ | $0.3441 \pm 0.0009$ | $50.08 \pm 0.01$ | 276.7 MB | `27.5 / 31.2 / 24.2 / 17.1%` |

* **Inductive-Bias Takeaway:** On 862 spatially coupled highway sensors, `iTransformer` dominates both accuracy (`0.4372` MSE) and latency (`22.06 ms/batch`) by attending directly across the 862 variate tokens (`8` sequences of `862` tokens), whereas Channel-Independent temporal-patch models process `8 × 862 = 6,896` isolated sensor sequences without cross-sensor propagation visibility and require more than 10 epochs for 4 separate resolution heads (`709K` params) to fully converge.

---

### D. Block 4 — Mechanistic Interpretability & Case Studies ([`results/traffic_routing_analysis.csv`](results/traffic_routing_analysis.csv))
Evaluated across all **`2,943,730` univariate Traffic test windows** using [`checkpoints/frp_gumbel_traffic_s42.pt`](checkpoints/frp_gumbel_traffic_s42.pt):

| Patch Size | Branch | Window Count | Share (%) | Mean Entropy $H \pm \text{Std}$ | Median $[q_{25} \text{ – } q_{75}]$ | Learned $H$ Decision Interval |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`P=8`** (Fine) | `0` | 800,313 | **27.19%** | $0.6643 \pm 0.0649$ | $0.6502\ [0.6135 \text{ – } 0.7003]$ | **$[0.5807,\ 1.0000]$** |
| **`P=16`** (Med-Fine) | `1` | 924,905 | **31.42%** | $0.5081 \pm 0.0418$ | $0.5080\ [0.4721 \text{ – } 0.5442]$ | **$[0.4350,\ 0.5807)$** |
| **`P=32`** (Med-Coarse) | `2` | 706,261 | **23.99%** | $0.3692 \pm 0.0399$ | $0.3710\ [0.3348 \text{ – } 0.4041]$ | **$[0.2968,\ 0.4350)$** |
| **`P=64`** (Coarse) | `3` | 512,251 | **17.40%** | $0.2228 \pm 0.0511$ | $0.2312\ [0.1873 \text{ – } 0.2643]$ | **$[0.0000,\ 0.2968)$** |
| **ALL** | `-1` | **2,943,730** | **100.00%** | $0.4676 \pm 0.1623$ | $0.4757\ [0.3446 \text{ – } 0.5912]$ | **Spearman $\rho = -0.9643$ ($p = 0.000$)** |

* **Adaptive Boundary Shift ([`figures/traffic_routing_histogram.png`](figures/traffic_routing_histogram.png)):** The Gumbel router achieved a **Spearman rank correlation of $\rho(H, P) = -0.9643$ ($p = 0.000$)** and shifted the two lower thresholds leftward from the fixed training quartiles $(0.3336,\ 0.4632,\ 0.5780)$ to $(0.2968,\ 0.4350,\ 0.5807)$, reducing coarse 2-patch `P=64` allocation from `25%` to `17.40%` and expanding `P=16` to `31.42%`.
* **Qualitative Case Study ([`figures/smooth_vs_volatile_forecasts.png`](figures/smooth_vs_volatile_forecasts.png)):**
  * **Smooth Periodic Sensor (`Sensor #309 | Test Window #1047`):** $H = 0.1532 < q_{25} \to$ Routed to **`P = 64`** (`stride = 32`), achieving **Window MSE = `0.0066`**, **MAE = `0.0622`**.
  * **Volatile Rush-Hour Sensor (`Sensor #729 | Test Window #881`):** $H = 0.6181 > q_{75} \to$ Routed to **`P = 8`** (`stride = 4`), resolving sharp primary rush-hour spikes and secondary mid-day plateaus with **Window MSE = `0.0977`**, **MAE = `0.2425`**.
