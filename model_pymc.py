"""
model_pymc.py  --  the full Bayesian model for the report / real data.

Hierarchical Poisson GLM with a log link, fit with NUTS in PyMC.  It
generalises the conjugate model in analysis.py by (i) adding covariates
through the log link and (ii) sharing information across stations via a
hierarchical prior on the station intercepts (partial pooling).

    D_{s,t} ~ Poisson(lambda_dep),  A_{s,t} ~ Poisson(lambda_arr)
    log lambda^.  = alpha^._s + beta^._we * weekend_t
    alpha^._s     ~ Normal(mu^., sigma^.)        (hierarchical / partial pooling)

The daily net flow N = A - D is the difference of two Poisson counts and is
therefore Skellam-distributed; we obtain its posterior predictive directly
from the two fitted Poisson rates.

Requires:  pymc, arviz, numpy, pandas, matplotlib  (pip install pymc arviz)
Reads station_day_counts.csv from build_dataset.py.  Fitting NUTS over all
~2300 stations is heavy, so the __main__ block fits a subset of the busiest
stations by default; raise TOP_STATIONS (or set it to None) to use them all.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import re


def repair_split_ids(df):
    """Normalise numeric station ids to 2 decimals so '6948.1' == '6948.10'
    (same station, stored inconsistently by Citi Bike). No-op on a clean file."""
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


def fit(df, draws=1000, tune=1000, chains=4, seed=42):
    df = df.copy()
    df["sid"] = df["station"].astype("category").cat.codes
    stations = df["station"].astype("category").cat.categories
    s = df["sid"].values
    we = df["is_weekend"].values.astype(float)
    A = df["arrivals"].values
    D = df["departures"].values

    coords = {"station": stations.astype(str)}
    with pm.Model(coords=coords) as model:
        we_ = pm.Data("weekend", we)

        # ---- arrivals ----
        mu_a   = pm.Normal("mu_a", 0.0, 2.0)
        sig_a  = pm.HalfNormal("sig_a", 1.0)
        a_arr  = pm.Normal("a_arr", mu_a, sig_a, dims="station")
        b_arr  = pm.Normal("b_arr_weekend", 0.0, 1.0)
        pm.Poisson("A_obs", mu=pm.math.exp(a_arr[s] + b_arr * we_), observed=A)

        # ---- departures ----
        mu_d   = pm.Normal("mu_d", 0.0, 2.0)
        sig_d  = pm.HalfNormal("sig_d", 1.0)
        a_dep  = pm.Normal("a_dep", mu_d, sig_d, dims="station")
        b_dep  = pm.Normal("b_dep_weekend", 0.0, 1.0)
        pm.Poisson("D_obs", mu=pm.math.exp(a_dep[s] + b_dep * we_), observed=D)

        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=0.9, random_seed=seed)
        # extend_inferencedata=True adds the posterior_predictive group in place
        # (works with both classic InferenceData and the new DataTree backend)
        pm.sample_posterior_predictive(idata, var_names=["A_obs", "D_obs"],
                                       extend_inferencedata=True)
    return model, idata, df


def net_flow_predictive(idata, df):
    """Posterior-predictive daily net flow N = A - D per station (Skellam)."""
    ppc = idata.posterior_predictive
    A = ppc["A_obs"].stack(sample=("chain", "draw")).values   # (obs, samples)
    D = ppc["D_obs"].stack(sample=("chain", "draw")).values
    N = A - D
    df = df.reset_index(drop=True)
    out = {}
    for _, idx in df.groupby("sid").groups.items():
        idx = np.asarray(idx)
        station = df.loc[idx[0], "station"]          # real id, e.g. "6948.10"
        out[station] = N[idx].ravel()
    return out


if __name__ == "__main__":
    import os
    PREPARED_CSV  = "station_day_counts.csv"
    YEAR_ONLY     = 2023
    MIN_DAYS      = 30
    TOP_STATIONS  = 150     # fit the busiest N stations (None = all; much slower)
    IDATA_FILE    = "idata.nc"

    # rebuild the exact modelling frame (deterministic, same order as sampling)
    df = pd.read_csv(PREPARED_CSV, dtype={"station": str}, parse_dates=["day"])
    df = repair_split_ids(df)
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["day"].dt.weekday >= 5).astype(int)
    if YEAR_ONLY is not None:
        df = df[df["day"].dt.year == YEAR_ONLY]
    if MIN_DAYS:
        df = df[df.groupby("station")["day"].transform("size") >= MIN_DAYS]
    if TOP_STATIONS:
        busiest = (df.groupby("station")[["departures", "arrivals"]].sum().sum(axis=1)
                     .nlargest(TOP_STATIONS).index)
        df = df[df.station.isin(busiest)]
    df = df.reset_index(drop=True)
    df["sid"] = df["station"].astype("category").cat.codes
    print(f"{df.station.nunique()} stations, {len(df):,} station-days")

    os.makedirs("plots", exist_ok=True)

    # reuse a saved posterior if present (no 6-minute re-sample); delete
    # idata.nc to force a fresh fit.
    if os.path.exists(IDATA_FILE):
        idata = az.from_netcdf(IDATA_FILE)
        print(f"loaded existing {IDATA_FILE} (delete it to re-sample)")
    else:
        _, idata, df = fit(df)
        try:
            idata.to_netcdf(IDATA_FILE)
            print(f"saved posterior to {IDATA_FILE}")
        except Exception as e:
            print("could not save idata.nc:", e)

    # ---- convergence diagnostics (rubric: report these) ----
    summ = az.summary(idata, var_names=["mu_a", "sig_a", "b_arr_weekend",
                                        "mu_d", "sig_d", "b_dep_weekend"])
    print(summ.to_string())          # full table; HDI column names vary by version
    full = az.summary(idata, var_names=["mu_a", "sig_a", "a_arr", "b_arr_weekend",
                                        "mu_d", "sig_d", "a_dep", "b_dep_weekend"])
    if "r_hat" in full and "ess_bulk" in full:
        # some arviz versions return these as strings -> coerce to numbers
        rhat = pd.to_numeric(full["r_hat"], errors="coerce").max()
        ess  = pd.to_numeric(full["ess_bulk"], errors="coerce").min()
        print(f"max R-hat: {rhat:.3f} | min ESS-bulk: {ess:.0f}")

    # ---- posterior of the weekend effect (manual, no arviz plotting dep) ----
    try:
        post = idata.posterior
        ba = np.asarray(post["b_arr_weekend"]).ravel()
        bd = np.asarray(post["b_dep_weekend"]).ravel()
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.hist(ba, bins=45, density=True, alpha=0.6, color="#c0504d",
                label="arrivals  $\\beta_{weekend}$")
        ax.hist(bd, bins=45, density=True, alpha=0.6, color="#1f4e79",
                label="departures  $\\beta_{weekend}$")
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_xlabel("weekend effect on log-rate (negative = fewer trips)")
        ax.set_ylabel("posterior density")
        pct = (1 - np.exp(np.mean([ba.mean(), bd.mean()]))) * 100
        ax.set_title(f"Weekend effect: about {pct:.0f}% fewer trips")
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig("plots/pymc_weekend_effect.png", bbox_inches="tight")
        plt.close(fig)
        we = az.summary(idata, var_names=["b_arr_weekend", "b_dep_weekend"])
        print("\nweekend effect (log-scale coefficients):")
        print(we.to_string())
    except Exception as e:
        print("weekend-effect plot failed:", e)

    # ---- net-flow posterior predictive for one station ----
    try:
        net = net_flow_predictive(idata, df)
        s0 = max(net, key=lambda k: net[k].mean())     # the strongest sink
        plt.figure(figsize=(6, 4))
        plt.hist(net[s0], bins=40, density=True, color="#7aa6c2", edgecolor="white")
        lo, hi = np.percentile(net[s0], [5, 95])
        plt.axvspan(lo, hi, color="#7aa6c2", alpha=0.2)
        plt.axvline(0, color="0.4", ls="--")
        plt.xlabel("daily net flow  N = A - D"); plt.ylabel("posterior predictive density")
        plt.title(f"Station {s0}: net-flow predictive (Skellam), 90% [{lo:.0f}, {hi:.0f}]")
        plt.tight_layout(); plt.savefig("plots/pymc_netflow_station.png", bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("net-flow plot failed:", e)
    print("done.")