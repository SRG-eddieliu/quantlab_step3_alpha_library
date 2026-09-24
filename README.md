# QuantLab Alpha Library (Legacy Component)

> **Consolidated implementation:** start with [Systematic Alpha Lab](https://github.com/SRG-eddieliu/systematic-alpha-lab) and its [`alpha` package](https://github.com/SRG-eddieliu/systematic-alpha-lab/tree/main/src/systematic_alpha_lab/alpha).
>
> This repository is retained as an earlier standalone implementation. The notes below describe that version, including its original paths and assumptions. They are not evidence of a validated investment strategy. See the consolidated repository for current scope, evaluation limitations, and planned work.

## Original Component Documentation

A lightweight alpha-construction toolkit that consumes factor/composite outputs, purifies them against common risks, applies multiple weighting engines, and evaluates IC/IR/decay/turnover. Outputs per-method alpha series are saved under `outputs/`.

## Core specs / assumptions
- Target: 1-day forward return, open-to-open, shifted 1 day to avoid look-ahead.
- Universe filters: price > 5; optional ADV gate (disabled in the default config).
- Coverage gates: drop dates with <80% availability on primary FF factors; per-feature drop if <70% coverage on a date; median-impute remaining NaNs.
- Controls: sector one-hot (merge sectors with <5 names into `MISC`), beta (252d window suggested), size proxy (e.g., log mkt cap), plus FF style factors if provided.
- Purification: ridge regression to orthogonalize returns and factors vs. sector/beta/size + FF factors (if available/passing coverage); ridge alpha configurable.
- Purification on returns: per-date ridge regression on sector dummies + beta/size + FF factors (if available); residual is the FF/sector-neutral return we trade.
- IC calc: rank IC on purified returns (post-purging); IC window 252d for weighting; IC weights are walk-forward (use prior IC only per date).
- GMV variance-min weights (Σ^-1 * 1; abs for long-only) using a rolling covariance window (default 252d); MLR/MVO use rolling windows (default 252d) to avoid look-ahead; MLR supports ridge (`mlr_ridge_alpha`) for stability and skips pathological dates if the design matrix is ill-conditioned.
- Returns are simple open-to-open arithmetic; for daily horizons rank IC is insensitive to log vs. simple (we keep it simple for now).
- XGB pod (global_ml): walk-forward scheme with per-date train/val split (252d train, 30d val), rank-IC eval metric, early stopping; trees kept small (depth 3, lr 0.01, subsample/colsample 0.8, gamma 0.2, reg_lambda 1.0, up to 1000 estimators).
- ML variants (global_ml only): XGB plus RF/GBM/MLP using purified composites + selected purified raw factors; traditional weighting remains for other pods.


## Alpha pods (config-driven)
Pods are defined in `config/config.json` under `pods`. Each pod is a set of theta composites to weight and evaluate:
- `core_systematic`: `theta_pure_value`, `theta_pure_quality`, `theta_momentum_riskadj`, `theta_systematic_risk`
- `event_info`: `theta_info_forensic_drift`, `theta_shortterm_reversal`, `theta_structural_liquidity`, `theta_pure_quality`
- `growth_cyclical`: `theta_growth_acceleration`, `theta_momentum_riskadj`
- `global`: all major themes (`theta_pure_value`, `theta_pure_quality`, `theta_momentum_riskadj`, `theta_systematic_risk`, `theta_info_forensic_drift`, `theta_shortterm_reversal`, `theta_structural_liquidity`, `theta_growth_acceleration`, `theta_size_smallcap`)
- `global_ml`: same theme set, but **ML-only**: XGBoost (rank-IC eval) on purified returns/features; no traditional weighting.
Controls (beta/size/sector) are applied in purification and are not directly weighted into alpha.

## Setup
- Python >=3.10, pandas, numpy.
- Data locations are set in `config/config.json`:
  - `price_file`: daily price parquet with `date`, `ticker`, `open`, `volume`
  - `ff_file`: Fama-French factors (optional)
  - `factor_dir`: where `factor_<name>.parquet` live (from factor library)
  - `output_dir`: where alpha series are written
  - Gates/params: `beta_window`, `ic_window`, `mlr_window`, `mlr_ridge_alpha`, `cov_window`, coverage thresholds, liquidity gates (price/ADV), `purge_ridge_alpha`, `risk_parity_abs`, `mvo_risk_aversion`, `mvo_long_only`, `bayes_lambda`

## Workflow (example)
```python
from alpha_compositor import AlphaDataLoader, run_alpha_pipeline, load_config
import pandas as pd

cfg = load_config("config/config.json")
loader = AlphaDataLoader(cfg)

# Load base composites/factors (already cleaned by factor library)
theta = {
    "theta_pure_value": loader.load_composite("theta_pure_value"),
    "theta_pure_quality": loader.load_composite("theta_pure_quality"),
    "theta_momentum_riskadj": loader.load_composite("theta_momentum_riskadj"),
    "theta_systematic_risk": loader.load_composite("theta_systematic_risk"),
}

# Controls
sector_map = pd.read_parquet("../../data/data-processed/company_overview.parquet").set_index("ticker")["sector"]
beta_series = pd.read_parquet("../../data/factors/factor_beta_252d.parquet").pivot(index="Date", columns="Ticker", values="Value")
size_series = pd.read_parquet("../../data/factors/factor_size_log_mktcap.parquet").pivot(index="Date", columns="Ticker", values="Value")

results = run_alpha_pipeline(
    cfg_path="config/config.json",
    theta=theta,
    sector_map=sector_map,
    beta_series=beta_series,
    size_series=size_series,
    factors_for_ic=theta,  # use the same set for IC weighting
)

for method, res in results.items():
    print(method, res["metrics"]["ic"], res["alpha_path"])
```

## Alpha creation steps (input → output)
- Load config and parquet artifacts: prices/volume, Fama-French factors (optional), sector map, betas/sizes, composites/raw factors from `factor_dir`.
- Liquidity/universe: apply price > `price_min` (and ADV gate if configured).
- Forward returns: compute 1-day open-to-open forward returns.
- Purge controls via ridge (α = `purge_ridge_alpha`): per-date regress returns on sector dummies + beta + size + FF factors (when available/passing coverage); take residuals. Same for each composite/raw factor to orthogonalize them to controls. Sparse forensic composite is ffilled then zero-filled before purge to keep cross-sections alive.
- Align timelines: reindex purged composites/factors to the purged return dates and forward-fill to maximize overlap.
- Weighting:
  - Equal
  - IC (walk-forward: for each date, use prior `ic_window` ICs per factor to form weights; no look-ahead)
  - MLR pseudo-inverse: rolling window (`mlr_window`, default 252d) over past dates/tickers; fit ridge-stabilized pinv on clipped X/y, skip ill-conditioned slices, normalize betas (sum |beta| = 1) for that date, apply those date-specific weights to the same date’s purified theta (no look-ahead). `mlr_ridge_alpha` controls the ridge term.
  - Bayesian shrink: per-date convex blend of MLR and equal weights, `w = λ·w_mlr + (1-λ)·w_equal`, with λ = `bayes_lambda` (default 0.5); uses the same rolling-window MLR weights, so no look-ahead.
  - GMV: rolling covariance (`cov_window`, default 252d) of purified theta; weights ∝ Σ^-1·1 (abs for long-only) computed per date, no look-ahead.
  - MVO: rolling covariance (`cov_window`) plus walk-forward expected returns from prior IC means (`ic_window`); weights ∝ (Σ + ridge I)^-1 * μ / λ, with `mvo_risk_aversion` and `mvo_long_only`.
  - ML pod (global_ml only): trains XGB/RF/GBM/MLP on purified features vs purified returns with walk-forward splits (see ML section below).
  - Save outputs: per-method alpha parquet to `output_dir` and per-pod diagnostics to `diagnostics/alpha_metrics_all.csv`.

## Weighting engines implemented
- Equal weight
- IC weight (walk-forward, window = `ic_window`; uses only past ICs per date)
- MLR weight (rolling window `mlr_window`, default 252d; per-date ridge-stabilized pinv fit on prior window of purified returns vs purified theta; weights normalized by sum |beta| and applied date-by-date; ridge via `mlr_ridge_alpha`)
- Bayesian shrink (blend MLR with equal, λ configurable via `bayes_lambda`, default 0.5)
- GMV (rolling covariance over `cov_window`, pseudo-inverse covariance, abs exposures if enabled)
- MVO (rolling covariance over `cov_window`; mean-variance: weights ∝ (Σ + ridge I)^-1 * μ / λ where μ = prior-IC means per factor, λ = `mvo_risk_aversion`; optional long-only via `mvo_long_only`)

## ML pod (global_ml)
- **Features**: purified theta composites + selected purified raw factors (per config), aligned to purged return dates.
- **Target**: purified forward returns.
- **Walk-forward scheme**: per-date training using only past data; outputs daily alpha predictions.
- **XGB**: `train_window` 126, `val_window` 10; trees depth 3, lr 0.01, subsample/colsample 0.8, gamma 0.2, reg_lambda 1.0, `n_estimators` 200 (hist). Good for non-linear interactions; tune windows/regularization if overfitting.
- **RF**: bootstrap ensembles (`n_estimators` 100, depth 5, min_samples_leaf 5). Stable, handles non-linearities; less sensitive to scaling.
- **GBM**: gradient boosting (`learning_rate` 0.03, depth 3, `n_estimators` 100, subsample 0.8). More sensitive to overfit; watch shrinkage/trees.
- **MLP**: feedforward net (`hidden_layer_sizes` [64,32], relu, alpha 5e-4, lr 1e-3, `max_iter` 150, `train_window` 126). Needs normalized features; may need patience/early stopping for stability.
- **Considerations**: ensure no look-ahead by keeping train/val windows strictly before the prediction date; shrink/regularize aggressively on sparse factors; check coverage/NaNs; xgboost requires `xgboost` + `libomp` installed.

Each method saves alpha series to `outputs/alpha_<method>.parquet`. Metrics include IC, IR, decay (5/10/21d), turnover.

## Evaluation metrics
- Rank IC and IR on purified returns
- Decay at 5/10/21 days
- Turnover: average |Δ rank| pct day-over-day. This is a lightweight “signal churn” proxy; production turnover is usually measured on portfolio weights after trading/constraints, but rank-churn is a standard diagnostic for raw alphas.

## Notes / considerations
- Small sectors merged into `MISC` to keep regressions stable.
- Ridge penalty is configurable; consider CV to tune.
- IC weighting uses 252d window; cap weights if a single theta dominates.
- Missing data: drop features below coverage threshold per date; median-impute the rest, then renormalize weights.
- FF controls: included in the purge when present and passing `min_ff_coverage`; otherwise the purge falls back to sector/beta/size only.
- Forensic composite (`theta_info_forensic_drift`) mixes SUE/earnings-surprise, analyst revisions, and Benford chi-square (sign-flipped). Positive = beats/upgrades/clean accounting; negative = misses/downgrades/Benford flags. When no new info is present we fill with `0` (neutral); switch to ffill if you want the last known signal to persist.
- Cross-sectional regression (one big regression per date) is the standard way to estimate factor payoffs/weights; per-ticker time-series fits are noisy and not used here.
- GMV uses absolute exposures; adjust if you need sign-aware RP or full ERC.
- XGBoost/ML weighting is not yet wired; add a guarded train/predict step with strong regularization if needed.

## Notebook demo
See [notebooks/alpha_demo.ipynb](notebooks/alpha_demo.ipynb) for a runnable example that:
1) Loads composites and controls
2) Runs the pipeline
3) Summarizes IC/IR/decay/turnover and shows per-method weights
