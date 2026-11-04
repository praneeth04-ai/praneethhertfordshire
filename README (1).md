# Forecasting German Overnight Base Load

**Repository:** `<paste the GitHub URL here after pushing>`

This repository contains a reproducible time-series forecasting workflow for German electricity demand, built around one question: how well can the weekly **overnight base load** be forecast two years ahead?

Hourly load from Open Power System Data is reduced to the weekly mean of the overnight hours (00:00 to 05:59, Berlin local time, weeks ending on Tuesday). That floor is what must-run plant has to cover and the window in which storage is charged. A range of models is compared on the final 104 weeks: simple benchmarks, SARIMA chosen by AIC over all 147 (p, d, q) orders, SARIMAX with temperature and holiday covariates, random forest and gradient boosting, and an hourly LSTM.

## Project aim

The aim is to forecast weekly overnight German electricity demand two years ahead and to compare the accuracy, uncertainty, interpretability and complexity of the forecasting approaches.

The research questions are:

1. How well do simple benchmark methods forecast weekly overnight demand?
2. Does SARIMA or SARIMAX improve on the seasonal naive benchmark?
3. Do temperature and holiday covariates improve accuracy, and are they known at the forecast origin?
4. Do feature-based or neural models justify their extra complexity?
5. Which model is most appropriate for operational use?

## Data

The target is German load (`DE_load_actual_entsoe_transparency`, MW) from the OPSD 60-minute file, 1 January 2015 to the end of the German series (2020-10-01 01:00:00 local time).

Processing:

* the UTC index is sliced from one hour before local midnight on 1 January 2015 and converted to Berlin wall-clock time (the first value is empty in the file, so the series starts at 01:00 on 1 January 2015); the repeated autumn 02:00 hour is averaged and the skipped spring 02:00 hour is interpolated (6 hours filled in total);
* load is converted from MW to GW;
* **weekly target `overnight_gw`**: mean of the hours 00:00 to 05:59 in each `W-TUE` week, kept only if at least 40 of the 42 overnight hours were observed (299 weeks). Its mean is 44.90 GW against 55.51 GW for the plain all-hours weekly mean, which is kept beside it for comparison;
* **daily series**: overnight mean and all-hours mean per day, used for EDA and the stationarity battery;
* **hourly series**: used by the LSTM.

Covariates:

```text
temp_c      weekly mean of the population-weighted daily mean 2 m temperature of
            Berlin, Hamburg, Munich, Cologne and Frankfurt (Open-Meteo archive API);
            weights are approximate city populations (3.6, 1.8, 1.5, 1.1, 0.75 million)
hdd         weekly heating degree days, base 16.0 °C
cdd         weekly cooling degree days, base 20.0 °C
temp_lag1   previous week's temperature (tree models, built with shift(1))
hdd_clim    mean HDD for the week of the year over TRAINING weeks only (known at origin)
holidays    number of German national public holidays in the week (holidays package)
```

The brief specifies Berlin as the representative location for temperature. Berlin is kept as that baseline: a SARIMAX run with Berlin-only degree days is reported next to the main one. The five-city, population-weighted index is a deliberate extension, used by the main models because German demand is spread well beyond Berlin.

Temperature is an external covariate: two years ahead the realised temperature is not known. Every model fed realised test-period temperature is labelled a **conditional (explanatory) forecast** and flagged `known_at_origin = no` in the results. Holiday counts and the training climatology are known in advance.

The raw OPSD file (about 130 MB) is not stored in git; `scripts/download_data.py` (or the first notebook cell that loads data) fetches it into `data/`. The five-city temperature cache (`data/temperature_five_cities_daily.csv`) is small and is kept in the repository, so the notebook also runs offline.

## Repository structure

The layout follows the course example README in spirit but is deliberately simpler: one flat `data/` folder instead of raw/interim/processed, model code in flat modules rather than a `models/` subpackage, and `scripts/run_pipeline.py` executing the narrated notebook instead of a separate `pipeline.py`, so the command line and the notebook can never disagree.

```text
electricity-demand-forecast/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── time_series_60min_singleindex.csv      # downloaded, git-ignored
│   └── temperature_five_cities_daily.csv      # Open-Meteo cache
│
├── src/
│   └── nightload/
│       ├── __init__.py
│       ├── config.py         # paths, lens, split, grids
│       ├── data.py           # OPSD loading, Berlin clock, bins, weather, holidays
│       ├── stationarity.py   # STL / MSTL, ADF + KPSS battery
│       ├── features.py       # weekly frame, leakage-safe supervised table
│       ├── benchmarks.py     # mean, naive, seasonal naive (+ interval), drift
│       ├── sarima.py         # 147-order AIC grid, SARIMA / SARIMAX, diagnostics
│       ├── tree_models.py    # random forest, gradient boosting, recursive rollout
│       ├── lstm.py           # PyTorch LSTM, sweep, day-by-day rollout
│       ├── evaluation.py     # metrics, Diebold-Mariano, regime errors
│       ├── narrative.py      # renders the findings text from computed values
│       └── plotting.py       # FigureRole style engine and all figures
│
├── notebooks/
│   └── overnight_base_load_forecast.ipynb    # the full, executed workflow
│
├── scripts/
│   ├── download_data.py
│   └── run_pipeline.py       # command-line entry point: runs the whole notebook
│
├── tests/
│   └── test_core.py
│
├── figures/                  # PNGs written by the notebook
└── outputs/                  # forecasts, metrics, grid, sweep, run facts
```

## Pipeline overview

The main workflow is controlled by:

```text
scripts/run_pipeline.py
```

which executes `notebooks/overnight_base_load_forecast.ipynb` top to bottom. The notebook imports the reusable code in `src/nightload` and runs nine checkpoints:

1. **Data and EDA**: load, Berlin clock, weekly lens and plain mean, daily bins, temperature and holidays, plots, STL (weekly, period 52) and MSTL (daily, periods 7 and 365), ADF + KPSS on level, first, seasonal and combined differences, ACF and PACF.
2. **Benchmarks**: mean, naive, seasonal naive (with a 95% interval) and drift over 104 weeks.
3. **SARIMA**: every order p in 0..6, d in 0..2, q in 0..6 fitted in parallel with seasonal (0,1,1)[52] and written to `outputs/sarima_aic_grid.csv` with a metadata JSON (all 147 are refitted on every run unless `REFIT_SARIMA_GRID` in `config.py` is set to False); a seasonal-part check; a check of whether the MA terms cancel the differences; residual ACF, histogram, QQ plot, Ljung-Box and Jarque-Bera; forecast with a 95% interval.
4. **SARIMAX**: the chosen order with temperature, holidays, and both, as standardised `exog`, plus a Berlin-only temperature sensitivity run.
5. **Feature models**: random forest (primary) and gradient boosting (secondary), lags 1, 3, 13 and 52, shifted rolling means over 4 and 8 weeks, tuned with `TimeSeriesSplit` on training weeks, recursive 104-week forecasts, validation-rollout intervals, four covariate sets.
6. **Hourly LSTM**: anomaly from a training-only seasonal profile, 24-hour output, sweep over window, hidden size, dropout and learning rate plus a layer-depth stage (18 settings), recursive day-by-day rollout over the test window, binned to the weekly lens.
7. **Comparison**: one table, Diebold-Mariano tests against the seasonal naive, master plot, regime errors.
8. **Extra checks**: the improvement rule, a leakage test, model size and run time, and the operational shortlist.
9. **Outputs** written to `outputs/` and `figures/`.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the analysis

From a fresh clone:

```bash
python scripts/download_data.py      # once: OPSD file (about 130 MB) and temperature cache
python scripts/run_pipeline.py       # runs the whole notebook and writes outputs/ and figures/
```

or open `notebooks/overnight_base_load_forecast.ipynb` and run all cells. On Google Colab, paste the repository URL into the marked `REPO_URL` line of the first code cell and run all cells; missing packages are installed automatically.

The LSTM runs on CPU by default, with every seed fixed, so its numbers repeat exactly. Setting `NIGHTLOAD_DEVICE=mps` or `NIGHTLOAD_DEVICE=cuda` opts in to an accelerator; the later decimals may then move. On an Apple-silicon laptop the modelling steps took about 8 minutes in total (SARIMA grid 5.0 min, LSTM sweep 2.4 min).

## Pushing this repository to GitHub

The repository is not yet under version control. To publish it:

```bash
cd electricity-demand-forecast
git init
git add .
git commit -m "Overnight base load forecasting workflow"
git branch -M main
git remote add origin https://github.com/<your-account>/electricity-demand-forecast.git
git push -u origin main
```

`.gitignore` keeps the 130 MB raw OPSD file out of the commit; `scripts/download_data.py` recreates it. After pushing, paste the URL into the **Repository** line at the top of this README and into `REPO_URL` in the notebook's first code cell. If the code is submitted as a zip instead, leave `data/time_series_60min_singleindex.csv` out of the archive.

## Models

* **Benchmarks**: mean, naive, seasonal naive (m = 52) and drift. The seasonal naive is the reference for every other model.
* **SARIMA**: SARIMA(1, 1, 6)x(0, 1, 1, 52), the lowest AIC (289.17) among the 129 orders that converged with a stationary AR part (147 fitted, 0 failed, 13 not converged). Its regular MA coefficients sum to -1.000 and its seasonal MA is -1.000, so the MA terms cancel both differences: the fit behaves like a stationary model, its interval barely widens with the horizon, and the AIC is a boundary value (invertibility enforced alone reaches the same AIC; with stationarity also enforced the AIC is 519.73, on a different likelihood start). The notebook reports this rather than overriding the AIC choice the brief asks for.
* **SARIMAX**: the same order with standardised HDD, CDD and holiday count as exogenous regressors, no constant (redundant with d = 1 and D = 1). The Berlin-only run (the location the brief specifies, kept as the baseline) scores MASE 1.550 against 1.541 for the five-city index.
* **Feature models**: `RandomForestRegressor` and `GradientBoostingRegressor` on lag, rolling, calendar, holiday and temperature features, forecast recursively.
* **Seasonal profile**: the training-period mean load per month x weekday x hour, binned to the lens. It is the LSTM's baseline and is scored as its own model.
* **LSTM**: PyTorch, best setting `w168_h96_d0.1_lr0.003_L1` from an 18-setting sweep, 43,104 weights; models the anomaly from a training-only month x weekday x hour profile and is rolled forward one day at a time over the two test years.

## Evaluation

All models are scored on the same 104 test weeks (2018-10-09 to 2020-09-29, weeks ending Tuesday) with MAE, RMSE, MAPE, MASE (scaled by the in-sample seasonal-naive MAE, 1.487 GW), bias (forecast minus actual) and 95% interval coverage where a model has an interval. A Diebold-Mariano test (squared-error loss) compares each model with the seasonal naive. All 104 forecasts come from one origin, so the loss differential is autocorrelated; the long-run variance uses a 12-lag Newey-West window, and the short 4-lag window is reported beside it as a sensitivity check. With one origin the test is approximate.

## Train-test split

The final 104 weeks (2018-10-09 to 2020-09-29) are the test set; the 195 weeks before them are training data. The split is time-ordered and nothing is shuffled. Model choices that need validation data take it from inside the training period: `TimeSeriesSplit` folds for the trees, the last 52 training weeks for the tree intervals, and the last 26 training weeks for the LSTM sweep.

## Outputs

```text
outputs/all_forecasts.csv          one row per test week: actual and every model
outputs/model_comparison.csv       the results table below
outputs/sarima_aic_grid.csv        all 147 orders (+ .meta.json with fingerprint and wall time)
outputs/lstm_sweep.csv             every LSTM setting and its validation score
outputs/lstm_hourly_scores.csv     hourly test metrics of the LSTM and the profile
outputs/regime_mae.csv             MAE in festive, lockdown, pre-2020, cold and warm weeks
outputs/stationarity.csv           ADF and KPSS battery
outputs/run_facts.json             every number quoted in the notebook text
figures/*.png                      19 figures (forecast comparison: 18_master_forecasts.png)
```

## Results

| Model | Family | MAE (GW) | RMSE (GW) | MAPE (%) | MASE | MASE / seasonal naive | Bias (GW) | 95% coverage | DM p (12 lags) | DM p (4 lags) | Known at origin |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Seasonal profile | profile (LSTM baseline) | 1.812 | 2.331 | 4.07 | 1.219 | 0.899 | -0.152 |  | 0.589 | 0.483 | yes |
| Gradient boosting | feature ML | 1.854 | 2.387 | 4.37 | 1.247 | 0.920 | 0.979 | 0.71 | 0.216 | 0.186 | no (conditional) |
| Random forest | feature ML | 1.868 | 2.390 | 4.39 | 1.257 | 0.927 | 0.863 | 0.66 | 0.228 | 0.231 | no (conditional) |
| LSTM | neural | 2.011 | 2.630 | 4.66 | 1.352 | 0.998 | 1.211 |  | 0.882 | 0.879 | yes |
| Seasonal naive | benchmark | 2.015 | 2.579 | 4.67 | 1.355 | 1.000 | 1.235 | 0.95 |  |  | yes |
| Random forest [climatology + holidays] | feature ML | 2.018 | 2.548 | 4.70 | 1.357 | 1.001 | 1.300 | 0.69 | 0.696 | 0.773 | yes |
| Gradient boosting [climatology + holidays] | feature ML | 2.054 | 2.585 | 4.80 | 1.381 | 1.019 | 1.461 | 0.73 | 0.952 | 0.953 | yes |
| Random forest [holidays only] | feature ML | 2.132 | 2.728 | 4.98 | 1.434 | 1.058 | 1.520 | 0.69 | 0.337 | 0.421 | yes |
| Random forest [no covariates] | feature ML | 2.133 | 2.745 | 4.99 | 1.434 | 1.059 | 1.446 | 0.71 | 0.273 | 0.387 | yes |
| Gradient boosting [holidays only] | feature ML | 2.145 | 2.688 | 5.00 | 1.442 | 1.065 | 1.483 | 0.72 | 0.301 | 0.360 | yes |
| Gradient boosting [no covariates] | feature ML | 2.234 | 2.815 | 5.21 | 1.502 | 1.109 | 1.620 | 0.82 | 0.067 | 0.134 | yes |
| SARIMAX [temperature + holidays] | statistical | 2.291 | 2.953 | 5.41 | 1.541 | 1.137 | 2.082 | 0.61 | 0.267 | 0.131 | no (conditional) |
| SARIMAX [Berlin temperature + holidays] | statistical | 2.305 | 2.945 | 5.43 | 1.550 | 1.144 | 2.068 | 0.58 | 0.252 | 0.124 | no (conditional) |
| SARIMAX [temperature] | statistical | 2.434 | 3.068 | 5.73 | 1.637 | 1.208 | 2.160 | 0.56 | 0.115 | 0.036 | no (conditional) |
| SARIMAX [holidays] | statistical | 2.549 | 3.113 | 5.94 | 1.714 | 1.265 | 2.296 | 0.58 | 0.059 | 0.012 | yes |
| SARIMA | statistical | 2.636 | 3.225 | 6.16 | 1.773 | 1.308 | 2.317 | 0.58 | 0.023 | 0.003 | yes |
| Mean | benchmark | 3.337 | 3.928 | 7.46 | 2.244 | 1.656 | 0.007 |  | 0.002 | 0.000 | yes |
| Drift | benchmark | 3.360 | 3.909 | 7.58 | 2.260 | 1.668 | 0.414 |  | 0.000 | 0.000 | yes |
| Naive | benchmark | 3.416 | 3.966 | 7.73 | 2.298 | 1.696 | 0.545 |  | 0.000 | 0.000 | yes |

* Lowest MASE overall: **Seasonal profile** (MASE 1.219, 0.899 of the seasonal naive).
* Lowest MASE using only inputs known at the origin: **Seasonal profile** (MASE 1.219).
* Models that beat the seasonal naive by at least 5% on MASE with a Diebold-Mariano p below 0.05 (12-lag window): **none**; with the 4-lag window: **none**.
* Operational recommendation (known inputs, with an interval): **Seasonal naive**.

The master forecast plot is `figures/18_master_forecasts.png`; regime errors are in `figures/19_regime_mae.png`.

## Data leakage

* Target lags, rolling means and `temp_lag1` use `shift(1)` or later; a unit test checks that the target lags and rolling means of row *t* never change when week *t* changes.
* The climatology, SARIMAX standardisation, LSTM profile and scale, MASE scale and all hyper-parameter choices are fitted on training data only.
* Tree and LSTM forecasts are recursive; the notebook replaces every test-window load value with noise and confirms the forest forecast does not change (largest difference below 1e-13 GW, which is floating-point rounding).
* Models that use realised test-period temperature are reported as conditional forecasts.

## Report

The written report (6 to 8 pages) is submitted separately as a Word or PDF file. Every number, table and figure it uses comes from this repository: tables from `outputs/`, figures from `figures/` and quoted values from `outputs/run_facts.json`.

## Good practice followed

```text
Reusable code in src/, one module per job, with docstrings and type hints.
The notebook explains; it does not hide logic in long cells.
Findings text is rendered from computed values, so it cannot drift from the tables.
Large raw data are not committed; a script recreates them.
Seeds are fixed and the LSTM runs on CPU by default for exact reruns.
Every model is compared with the seasonal naive benchmark.
Covariates are flagged as known or not known at the forecast origin.
Every statistic that learns from data uses training weeks only.
```

## Expected submission

```text
README.md                      this file
requirements.txt               pinned versions
src/nightload/                 source code
scripts/run_pipeline.py        command-line entry point
notebooks/                     the executed workflow
outputs/, figures/             generated metrics and figures
report                         submitted separately (Word or PDF)
```

## Limitations

* One 104-week test window that includes the 2020 lockdown; a rolling-origin evaluation would be more reliable.
* Fewer than four training years: the MA terms of SARIMA cancel both differences (an over-differencing signature), and the tree models have 143 training rows after the 52-week lag warm-up.
* Realised temperature is not available two years ahead; the climatology variants show the operational alternative.
* Every model inherits the 2015 to 2018 level, so all of them over-forecast a falling series.

## Testing

```bash
pytest -q
```

The tests check that MASE is zero for a perfect forecast, that benchmark forecasts match the test length and index, that lag and rolling features never see the current week, and that the Diebold-Mariano statistic has the expected sign.

## References

* Bouktif, S., Fiaz, A., Ouni, A. and Serhani, M.A. (2018) Optimal deep learning LSTM model for electric load forecasting using feature selection and genetic algorithm. *Energies*, 11(7), 1636.
* Diebold, F.X. and Mariano, R.S. (1995) Comparing predictive accuracy. *Journal of Business and Economic Statistics*, 13(3), 253-263.
* Harvey, D., Leybourne, S. and Newbold, P. (1997) Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281-291.
* Hochreiter, S. and Schmidhuber, J. (1997) Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
* Hyndman, R.J. and Athanasopoulos, G. (2021) *Forecasting: Principles and Practice*, 3rd edn. OTexts.
* Kong, W., Dong, Z.Y., Jia, Y., Hill, D.J., Xu, Y. and Zhang, Y. (2019) Short-term residential load forecasting based on LSTM recurrent neural network. *IEEE Transactions on Smart Grid*, 10(1), 841-851.
* Marino, D.L., Amarasinghe, K. and Manic, M. (2016) Building energy load forecasting using deep neural networks. *IECON 2016*, IEEE.
* Open Power System Data (2020) Data Package Time series, version 2020-10-06.
