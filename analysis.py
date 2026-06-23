"""
analysis.py  --  Bayesian end-of-day rebalancing (exact conjugate model).

Reads station_day_counts.csv (from build_dataset.py) and, per station s:
    departures D ~ Poisson(lambda_dep),   arrivals A ~ Poisson(lambda_arr)
    prior  lambda ~ Gamma(a0, b0)         (conjugate to the Poisson)
    => posterior  lambda | data ~ Gamma(a0 + sum(counts), b0 + n_days)
The daily net flow  N = A - D  is the difference of two Poisson counts,
i.e. a Skellam variable.  Its full predictive distribution comes from drawing
lambda from the two Gamma posteriors and then A, D from the Poissons.

This is the analytic special case of the hierarchical Poisson GLM in
model_pymc.py (no covariates, no pooling, no MCMC); it runs instantly.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

plt.rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120,
})
PREPARED_CSV = "station_day_counts.csv"
OUT   = "plots"
RNG   = np.random.default_rng(42)
NDRAW = 4000                       # posterior draws per station
A0, B0 = 0.001, 0.001             # weakly-informative Gamma prior
os.makedirs(OUT, exist_ok=True)    # make sure plots/ exists


def repair_split_ids(df):
    """Normalise numeric station ids to 2 decimals so '6948.1' and '6948.10'
    (the same station, stored inconsistently by Citi Bike) become one key.
    Alphanumeric ids (e.g. 'HB101') are left untouched.  No-op on a clean file."""
    def canon(s):
        try:
            return f"{float(s):.2f}"
        except (ValueError, TypeError):
            return str(s).strip()
    before = df["station"].nunique()
    df = df.copy()
    df["station"] = df["station"].astype(str).map(canon)
    after = df["station"].nunique()
    if after < before:
        print(f"NOTE: merged {before - after} split station id(s) "
              f"(e.g. 6948.1 == 6948.10). Rebuild the CSV to make this permanent.")
    return df

# --------------------------------------------------------------------------
# 1. data
# --------------------------------------------------------------------------
df = pd.read_csv(PREPARED_CSV, dtype={"station": str}, parse_dates=["day"])
if "is_weekend" not in df.columns:
    df["is_weekend"] = (df["day"].dt.weekday >= 5).astype(int)
df = repair_split_ids(df)

# ---- filters on the real data ----
YEAR_ONLY = 2023     # set to None to keep everything; drops the late-2022 bleed
MIN_DAYS  = 30       # ignore stations observed on fewer days (too noisy to map)
if YEAR_ONLY is not None:
    df = df[df["day"].dt.year == YEAR_ONLY]
if MIN_DAYS:
    keep = df.groupby("station")["day"].transform("size") >= MIN_DAYS
    df = df[keep]
print(f"stations after filtering: {df['station'].nunique()} | rows: {len(df):,}")

agg = (df.groupby("station")
         .agg(n_days=("day", "size"),
              S_arr=("arrivals", "sum"),
              S_dep=("departures", "sum"),
              lon=("lon", "first"), lat=("lat", "first"),
              obs_net_mean=("arrivals", "mean"))
         .reset_index())
agg["obs_net_mean"] = (df.groupby("station")
                         .apply(lambda g: (g.arrivals - g.departures).mean(),
                                include_groups=False).values)

# --------------------------------------------------------------------------
# 2. conjugate posterior for each station, and Skellam net-flow predictive
# --------------------------------------------------------------------------
post_arr = np.zeros((len(agg), NDRAW))   # lambda_arr draws
post_dep = np.zeros((len(agg), NDRAW))   # lambda_dep draws
pred_net = np.zeros((len(agg), NDRAW))   # predictive daily net flow draws

for i, r in agg.iterrows():
    la = RNG.gamma(A0 + r.S_arr, 1.0 / (B0 + r.n_days), NDRAW)
    ld = RNG.gamma(A0 + r.S_dep, 1.0 / (B0 + r.n_days), NDRAW)
    post_arr[i], post_dep[i] = la, ld
    pred_net[i] = RNG.poisson(la) - RNG.poisson(ld)      # Skellam predictive

net_mean = post_arr - post_dep                          # posterior of E[N]
agg["net_mean"]  = net_mean.mean(1)
agg["net_lo"]    = np.percentile(net_mean, 5, axis=1)
agg["net_hi"]    = np.percentile(net_mean, 95, axis=1)
agg["p_source"]  = (net_mean < 0).mean(1)               # P(station drains)

# --------------------------------------------------------------------------
# 3. a simple operational read-out
# --------------------------------------------------------------------------
sinks   = agg[agg.net_mean > 0].sort_values("net_mean", ascending=False)
sources = agg[agg.net_mean < 0].sort_values("net_mean")
move = agg["net_mean"].clip(lower=0).sum()              # bikes to remove from sinks
print(f"\nStations: {len(agg)} | est. bikes to relocate per day ~ {move:,.0f}")
print("\nTop 5 net SINKS (remove bikes):")
print(sinks[["station", "net_mean", "net_lo", "net_hi"]].head().to_string(index=False))
print("\nTop 5 net SOURCES (deliver bikes):")
print(sources[["station", "net_mean", "net_lo", "net_hi"]].head().to_string(index=False))

# pick example stations for the plots as the most extreme by net flow.
# (using the sorted extremes is robust even if every station were one sign).
# NOTE: post_*/pred_net are indexed by row position in `agg`; the station id is
# a string like "6948.10" -> keep both, never cast the id to int.
_by_net = agg.sort_values("net_mean")
source_pos, source_id = _by_net.index[0],  _by_net.iloc[0].station    # most negative
sink_pos,   sink_id   = _by_net.index[-1], _by_net.iloc[-1].station   # most positive

# ==========================================================================
# PLOTS
# ==========================================================================

# ---- Fig 1: net-flow map -------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 5.6))
# robust symmetric colour scale so a few huge hubs don't wash out the rest
vmax = np.percentile(np.abs(agg.net_mean), 98)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
sizes = 6 + 1.6 * np.sqrt(agg.net_mean.abs())
sc = ax.scatter(agg.lon, agg.lat, c=agg.net_mean, cmap="RdBu_r", norm=norm,
                s=sizes, alpha=0.8, edgecolor="k", linewidth=0.15)
cb = fig.colorbar(sc, ax=ax, extend="both")
cb.set_label("posterior mean daily net flow  (arrivals - departures)")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title("Estimated daily net flow per station\n(red = net sink, fills up;  blue = net source, drains)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_netflow_map.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 2: conjugate posterior of the rates (one station) ---------------
fig, ax = plt.subplots(figsize=(6.6, 4.2))
s = sink_pos
xa = np.linspace(min(post_dep[s].min(), post_arr[s].min()),
                 max(post_dep[s].max(), post_arr[s].max()), 400)
for draws, col, lab in [(post_dep[s], "#1f4e79", "departures  $\\lambda^{dep}$"),
                        (post_arr[s], "#c0504d", "arrivals  $\\lambda^{arr}$")]:
    a_post = A0 + (agg.loc[s, "S_dep"] if "dep" in lab else agg.loc[s, "S_arr"])
    b_post = B0 + agg.loc[s, "n_days"]
    ax.plot(xa, stats.gamma.pdf(xa, a_post, scale=1/b_post), color=col, lw=2, label=lab)
    lo, hi = np.percentile(draws, [5, 95])
    ax.axvspan(lo, hi, color=col, alpha=0.12)
ax.set_xlabel("daily rate  $\\lambda$  (bikes per day)"); ax.set_ylabel("posterior density")
ax.set_title(f"Conjugate Gamma posteriors for station {sink_id} (a net sink)\nshaded = 90% credible interval")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_rate_posteriors.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 3: Skellam net-flow predictive, source vs sink ------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, pos, sid, title in [(axes[0], source_pos, source_id, "source"),
                            (axes[1], sink_pos,   sink_id,   "sink")]:
    obs = (df[df.station == sid].arrivals - df[df.station == sid].departures).values
    lo, hi = np.percentile(pred_net[pos], [5, 95])
    bins = np.arange(pred_net[pos].min() - 1, pred_net[pos].max() + 2) - 0.5
    ax.hist(pred_net[pos], bins=bins, density=True, color="#7aa6c2",
            edgecolor="white", linewidth=0.3, label="predictive (Skellam)")
    ax.hist(obs, bins=bins, density=True, histtype="step", color="k",
            linewidth=1.4, label="observed days")
    ax.axvline(0, color="0.4", ls="--", lw=1)
    ax.axvspan(lo, hi, color="#7aa6c2", alpha=0.18)
    ax.set_title(f"Station {sid}  ({title})\n90% interval [{lo:.0f}, {hi:.0f}]")
    ax.set_xlabel("daily net flow  $N = A - D$")
axes[0].set_ylabel("probability"); axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Net flow is a Skellam (Poisson minus Poisson): it can be negative", y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_skellam_predictive.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 4: concentration of the rebalancing need ------------------------
order = agg.reindex(agg.net_mean.abs().sort_values(ascending=False).index)
cum = np.cumsum(order.net_mean.abs().values) / order.net_mean.abs().sum()
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.bar(np.arange(len(order)), order.net_mean.abs().values, color="#6b8fb0")
ax.set_xlabel("station rank (by absolute net flow)")
ax.set_ylabel("|posterior mean net flow|  (bikes/day)")
ax2 = ax.twinx()
ax2.plot(np.arange(len(order)), cum, color="#b5651d", lw=2, marker="o", ms=3)
ax2.set_ylabel("cumulative share of total imbalance", color="#b5651d")
ax2.set_ylim(0, 1.02); ax2.tick_params(axis="y", colors="#b5651d")
ax.set_title("A minority of stations drives most of the rebalancing")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_concentration.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 5: recovery / uncertainty (caterpillar) -------------------------
sub = agg.sort_values("net_mean").reset_index(drop=True)
take = np.linspace(0, len(sub) - 1, 25).astype(int)      # 25 stations across range
sub = sub.iloc[take].reset_index(drop=True)
y = np.arange(len(sub))
fig, ax = plt.subplots(figsize=(6.8, 6.2))
ax.hlines(y, sub.net_lo, sub.net_hi, color="#6b8fb0", lw=3, alpha=0.7,
          label="90% credible interval")
ax.plot(sub.net_mean, y, "o", color="#1f4e79", ms=5, label="posterior mean")
ax.plot(sub.obs_net_mean, y, "x", color="k", ms=6, label="observed mean")
ax.axvline(0, color="0.5", ls="--", lw=1)
ax.set_yticks(y); ax.set_yticklabels(sub.station.astype(str), fontsize=7)
ax.set_xlabel("daily net flow  (arrivals - departures)")
ax.set_ylabel("station id")
ax.set_title("Per-station net flow: estimate, uncertainty, and fit")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_recovery.png", bbox_inches="tight"); plt.close(fig)

print("\nsaved 5 figures to", OUT + "/")