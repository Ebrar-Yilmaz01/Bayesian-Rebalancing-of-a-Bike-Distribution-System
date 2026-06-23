# Code Summary — Project Overview

## Project: Bayesian Rebalancing of Citi Bike NYC

**Research Question:** For each station in Citi Bike NYC, what is the probability distribution of daily net flow (arrivals − departures), and how many bikes must be relocated at end-of-day?

**Dataset:** Citi Bike NYC, full year 2023, ~35 million trips, 2,145 stations after filtering.

---

## Architecture

```
Raw_Data/ (40 CSVs, 12 months)
    ↓
build_dataset.py  [ETL Pipeline]
    ↓
station_day_counts.csv  (2,145 stations × 365 days)
stations.csv  (lat/lng coordinates)
    ↓
├─→ analysis.py           [Exact Conjugate Model]
│   ├─ Gamma-Poisson conjugate posterior (closed-form)
│   ├─ Skellam net-flow predictive
│   └─ Output: 5 figures + operational summary
│
└─→ model_pymc.py         [Hierarchical GLM + MCMC]
    ├─ Poisson GLM with log link
    ├─ Hierarchical prior on station intercepts (partial pooling)
    ├─ Weekend/weekday covariate effect
    ├─ NUTS sampler, 4 chains
    └─ Output: 2 figures + MCMC diagnostics
```

---

## File-by-File Breakdown

### 1. **build_dataset.py** — Data Preparation (ETL)

**Purpose:** Read 40 raw Citi Bike CSVs and produce clean station-day aggregates.

**Key Features:**

- Recursive directory scan (handles monthly subfolders)
- Auto-detects column schema (modern vs. legacy format)
- **Station ID canonicalization:** "6948.1" and "6948.10" → same station (Citi Bike stores numeric IDs inconsistently)
- Handles split monthly files (2-4 parts per month)
- De-duplicates if both full file and split parts exist (by ride_id)
- Aggregates to: departures (trip start), arrivals (trip end) per station-day

**Outputs:**

- `station_day_counts.csv` — one row per station × day:
  - station, day, departures, arrivals, lon, lat, is_weekend
- `stations.csv` — unique stations with coordinates

**Data Integrity:**

- Total departures = Total arrivals = 35,001,868 (global consistency ✓)
- No split station IDs after canonicalization

**Run Time:** ~1-2 min (depends on disk I/O)

**Key Functions:**

- `_norm(col)` — normalize column names
- `_read_one(path)` — read CSV with schema flexibility
- `_canon_station(s)` — canonicalize numeric IDs to 2 decimals
- `_to_datetime(s)` — robust timestamp parsing across Citi Bike's historical formats
- `_aggregate(trips)` — group trips by station × day

---

### 2. **analysis.py** — Exact Conjugate Bayesian Model

**Purpose:** Fast, analytically exact inference using conjugate Gamma-Poisson priors. Produces headline figures and operational insights for all 2,145 stations.

**Mathematical Model:**
$$D_s \sim \text{Poisson}(\lambda_{\text{dep}}), \quad A_s \sim \text{Poisson}(\lambda_{\text{arr}})$$
$$\lambda \sim \text{Gamma}(a_0, b_0) \quad \text{[conjugate prior]}$$
$$\lambda | \text{data} \sim \text{Gamma}(a_0 + \sum c, b_0 + n_{\text{days}}) \quad \text{[posterior, closed-form]}$$
$$N = A - D \sim \text{Skellam}(\lambda_{\text{arr}}, \lambda_{\text{dep}})$$

**Hyperparameters:**

- `A0 = B0 = 0.001` — weakly informative prior (allows data to dominate)
- `NDRAW = 4000` — posterior draws per station (Monte Carlo sampling of posterior predictive)

**Workflow:**

1. Load `station_day_counts.csv`
2. Filter: year ≥ 2023, min 30 observation days per station → ~2,145 stations remain
3. For each station:
   - Compute posterior $\Gamma(a_0 + \sum c, b_0 + n)$ for arrivals and departures
   - Draw 4000 samples of $\lambda_{\text{arr}}, \lambda_{\text{dep}}$
   - Sample Poisson predictive: $A \sim \text{Poisson}(\lambda_{\text{arr}}), D \sim \text{Poisson}(\lambda_{\text{dep}})$
   - Compute net flow: $N = A - D$ (Skellam distribution)
4. Aggregate results: mean, 5th/95th percentiles, P(source)
5. Generate 5 visualizations (maps, distributions, top sinks/sources)

**Outputs:**

- `plots/1_net_flow_map.png` — geographic heatmap (red = sink, blue = source)
- `plots/2_net_flow_distribution.png` — histogram of mean net flows
- `plots/3_top_sources_sinks.png` — bar chart of extreme stations
- `plots/4_credible_intervals.png` — error bars for all stations
- `plots/5_example_predictive.png` — Skellam distribution for one station
- Console: statistics (top 10 sources, top 10 sinks, total rebalancing volume)

**Run Time:** ~5 seconds for all 2,145 stations

**Key Functions:**

- `repair_split_ids(df)` — canonicalize station IDs (same as in model_pymc.py)
- Posterior computation: Gamma-Poisson conjugacy (vectorized)
- Posterior-predictive sampling: Skellam construction

---

### 3. **model_pymc.py** — Hierarchical Poisson GLM (NUTS/MCMC)

**Purpose:** Full Bayesian hierarchical model with covariates and MCMC inference. Demonstrates advanced seminar material: GLMs, hierarchical priors (partial pooling), and MCMC diagnostics. Fit on top 150 busiest stations (can set `TOP_STATIONS=None` for all, but slow).

**Mathematical Model:**
$$D_{s,t} \sim \text{Poisson}(\lambda_{\text{dep}}), \quad A_{s,t} \sim \text{Poisson}(\lambda_{\text{arr}})$$
$$\log \lambda_{\text{dep}} = \alpha_s^{\text{dep}} + \beta_{\text{dep}} \cdot \mathbb{1}_{\text{weekend}}(t)$$
$$\log \lambda_{\text{arr}} = \alpha_s^{\text{arr}} + \beta_{\text{arr}} \cdot \mathbb{1}_{\text{weekend}}(t)$$
$$\alpha_s^{\text{dep}} \sim \text{Normal}(\mu_{\text{dep}}, \sigma_{\text{dep}}) \quad \text{[hierarchical prior]}$$
$$\mu_{\text{dep}}, \sigma_{\text{dep}}, \beta_{\text{dep}} \sim \text{weakly informative priors}$$

**Key Feature: Partial Pooling**

- Each station's intercept ($\alpha_s$) is drawn from a global distribution $\text{Normal}(\mu, \sigma)$
- Low-count stations borrow strength from the global mean
- Global parameters ($\mu, \sigma$) estimated hierarchically

**Priors:**

- $\mu \sim \text{Normal}(0, 2)$ — global mean intercept
- $\sigma \sim \text{HalfNormal}(1)$ — global scale
- $\beta \sim \text{Normal}(0, 1)$ — weekend effect

**Inference:**

- Sampler: NUTS (No-U-Turn Sampler, part of HMC)
- Chains: 4
- Draws: 1000 (after 1000 tuning steps per chain)
- Target acceptance: 0.9 (default ~0.65; higher = more careful exploration)
- Total posterior samples: 4,000 (4 chains × 1000 draws)
- Posterior predictive: 4,000 samples for A_obs and D_obs

**Outputs:**

- `plots/hierarchical_weekend_effect.png` — posterior of weekend coefficients
- `plots/hierarchical_net_flow.png` — posterior-predictive net flow for one example station
- Console: MCMC diagnostics (R-hat, ESS, divergences, summary stats)
- `idata.nc` — cached posterior (ArviZ InferenceData format, ~100 MB)

**MCMC Diagnostics (Signs of Convergence):**

- $\hat{R} < 1.01$ per parameter (≈1.0 = converged)
- ESS (effective sample size) > 400 per parameter (rule of thumb: > 10% of total)
- Divergences = 0 (sampler health; divergences = numerical problems)

**Run Time:**

- Subset (150 stations): ~10 minutes
- All stations: 2+ hours (not default)

**Key Functions:**

- `repair_split_ids(df)` — canonicalize station IDs
- `fit(df, draws, tune, chains, seed)` — build PyMC model and sample
- `net_flow_predictive(idata, df)` — compute posterior-predictive net flow (Skellam)

**Configuration:**

```python
TOP_STATIONS = 150  # set to None for all stations
SEED = 42
DRAWS = 1000
TUNE = 1000
CHAINS = 4
```

---

## Data Formats

### station_day_counts.csv

```
station   | day        | departures | arrivals | lon    | lat    | is_weekend
--------- | ---------- | ---------- | -------- | ------ | ------ | ----------
6948.10   | 2023-01-01 | 42         | 38       | -73.95 | 40.75  | 1
6948.10   | 2023-01-02 | 56         | 49       | -73.95 | 40.75  | 0
...
```

### stations.csv

```
station  | lon    | lat
-------- | ------ | ------
6948.10  | -73.95 | 40.75
...
```

---

## Key Results & Insights

### From analysis.py (Conjugate Model)

1. **Net Flow Map:** Geographic clustering of sources (blue, net outflow) and sinks (red, net inflow)
2. **Top Sources:** Stations with mean net flow < -20 (e.g., commuter rail hubs, airports)
3. **Top Sinks:** Stations with mean net flow > +20 (e.g., residential areas, tourists)
4. **System Total:** Sum of |net flow| across all stations = total daily rebalancing volume
5. **Credible Intervals:** 90% credible intervals quantify per-station uncertainty

### From model_pymc.py (Hierarchical GLM)

1. **Weekend Effect:** $\exp(\beta_{\text{weekend}})$ factor on rates
   - If $\beta > 0$: more activity on weekends
   - If $\beta < 0$: less activity on weekends
2. **Partial Pooling:** Station intercepts shrink towards global mean ($\mu$)
   - Stations with few observations benefit most from pooling
3. **MCMC Convergence:** R-hat ≈ 1, ESS > 400 → trustworthy samples
4. **Agreement:** Both models identify same source/sink stations (robustness check)

---

## Conceptual Connections to Seminar

| Seminar Topic               | Where It Appears                           | Why It Matters                                              |
| --------------------------- | ------------------------------------------ | ----------------------------------------------------------- |
| **Poisson Process**         | Departures and arrivals per station-day    | Counts of independent events in fixed time interval         |
| **Conjugate Priors**        | Gamma–Poisson in analysis.py               | Enables closed-form posterior (no MCMC needed)              |
| **Skellam Distribution**    | Net flow = A − D (difference of Poissons)  | Proper statistical model for imbalance; admits negatives    |
| **GLM / Log Link**          | `log λ = α + β·x` in model_pymc.py         | Keeps rate > 0; effects are multiplicative                  |
| **Hierarchical Model**      | Station intercepts drawn from Normal(μ, σ) | Partial pooling, information sharing across stations        |
| **Partial Pooling**         | Global mean shrinks individual estimates   | Balance between data and prior; improve low-count estimates |
| **MCMC / NUTS**             | Sampler in model_pymc.py                   | Approximate posterior for complex models without conjugacy  |
| **Convergence Diagnostics** | R-hat, ESS, divergences                    | Validate that MCMC samples are trustworthy                  |

---

## How to Run

### 1. Prepare Data (One-time)

```bash
python build_dataset.py /path/to/Raw_Data --out .
```

Produces: `station_day_counts.csv`, `stations.csv`

### 2. Quick Analysis (Fast, ~5 sec)

```bash
python analysis.py
```

Produces: 5 plots in `plots/` directory + console output

### 3. Full Hierarchical Model (Slower, ~10 min for subset or 2+ hrs for all)

```bash
python model_pymc.py
```

Produces: 2 plots + MCMC diagnostics + `idata.nc` cache

---

## Dependencies

**Required Packages:**

- `pandas` — data manipulation
- `numpy` — numerical computing
- `matplotlib` — plotting
- `scipy` — statistical functions (Gamma, Poisson sampling)
- `pymc` — Bayesian modeling (for model_pymc.py only)
- `arviz` — diagnostics and visualization (for model_pymc.py only)

**Install:**

```bash
pip install pandas numpy matplotlib scipy pymc arviz
```

---

## Code Quality & Reproducibility

✅ **No manual steps:** Fully automated ETL pipeline  
✅ **Schema flexibility:** Handles multiple CSV formats  
✅ **Data integrity:** Global consistency check (departures = arrivals)  
✅ **Deterministic:** Fixed random seed (42) for reproducibility  
✅ **Modular:** Each script is independent (build → analyze → model)  
✅ **Comments & docstrings:** Clear intent and assumptions  
✅ **Robustness check:** Two independent models agree on headline results

---

## Quick Reference: What Each Script Outputs

| Script             | Input                  | Output                               | Time             | Main Purpose                          |
| ------------------ | ---------------------- | ------------------------------------ | ---------------- | ------------------------------------- |
| `build_dataset.py` | 40 CSVs                | station_day_counts.csv, stations.csv | 1-2 min          | ETL & data cleaning                   |
| `analysis.py`      | station_day_counts.csv | 5 plots + summary stats              | 5 sec            | Fast exact inference for all stations |
| `model_pymc.py`    | station_day_counts.csv | 2 plots + diagnostics + idata.nc     | 10 min (150 stn) | Rich model with covariates + MCMC     |

---

## Troubleshooting

**Q: Why does build_dataset.py mention station ID canonicalization?**  
A: Citi Bike stored numeric station IDs as floats, so "6948.10" might appear as "6948.1" in another file. We normalize to 2 decimals to detect duplicates.

**Q: Which model should I use for the final results?**  
A: Use `analysis.py` (all 2,145 stations, headline figures). Use `model_pymc.py` for covariate effects and uncertainty validation.

**Q: How do I interpret "P(source) = 0.92"?**  
A: 92% posterior probability that the station's mean net flow is negative (net outflow = source station).

**Q: What if MCMC shows "divergences"?**  
A: Divergences = numerical instability. Increase tune steps, adjust priors, or use adaptive metric (see PyMC docs). For this dataset, should be ~0.

**Q: Can I fit model_pymc.py on all 2,145 stations?**  
A: Yes, set `TOP_STATIONS = None` in `__main__` block. Expect 2+ hours. Consider parallelization or limiting chains.
