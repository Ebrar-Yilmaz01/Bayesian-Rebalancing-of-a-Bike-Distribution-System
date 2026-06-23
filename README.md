# Bayesian End-of-Day Rebalancing for a Bike Sharing System

This repository contains a Bayesian workflow for end-of-day rebalancing in a bike sharing system. The project builds a modeling dataset, explores the demand and inventory behavior, fits a probabilistic model in PyMC, and uses the results to support station-level rebalancing decisions.

## Project Goal

The objective is to estimate station-level bike shortages and surpluses at the end of the day and to support better rebalancing actions under uncertainty. Instead of relying on a single point estimate, the approach uses Bayesian inference to represent uncertainty explicitly and make decisions from posterior distributions.

### Main files

- `build_dataset.py` builds the modeling dataset from 40 CSVs
- `analysis.py` performs exploratory analysis, summary statistics, and plot generation.
- `model_pymc.py` defines and fits the Bayesian model used for end-of-day rebalancing.
- `code_summary.md` provides a compact description of the codebase and implementation choices.
- `plots/` stores generated figures used for analysis and reporting.

## Workflow

1. Build the dataset.
2. Run exploratory analysis and generate plots.
3. Fit the Bayesian model.
4. Inspect posterior behavior and decision-relevant outputs.
5. Use the inferred shortage or surplus signals for rebalancing planning.


## Plots

The `plots/` folder is intended for generated figures such as:

- Demand distributions.
- Station usage comparisons.
- Temporal usage patterns.
- Posterior distributions.
- Model diagnostics.
- Rebalancing priority visualizations.

## Possible Extensions

- Add spatial structure across nearby stations.
- Include weather, weekday, holiday, or event features.
- Compare the Bayesian model with baseline heuristics or frequentist models.
- Optimize actual truck routes after shortage and surplus prediction.
- Turn the posterior outputs into a cost-aware rebalancing policy.

