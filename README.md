# Earnings-Based PEAD Strategy

A rule-based earnings research and backtesting project that studies **Post-Earnings Announcement Drift (PEAD)** using historical earnings events, event-level feature engineering, signal generation, and multi-horizon performance evaluation.

## Overview

This project investigates whether earnings surprises can generate **predictable post-announcement returns**. The core idea is based on the PEAD hypothesis: the market does not always fully incorporate earnings information immediately, so prices may continue to drift for several trading days or weeks after the announcement.

To test this idea, the project builds a full research pipeline:

1. Collect historical earnings events for a predefined watchlist
2. Construct a standardized event-level research dataset
3. Engineer trading features such as:
   - EPS surprise
   - Revenue surprise
   - SUE (Standardized Unexpected Earnings)
   - Gap percentage
   - Volume ratio
4. Generate rule-based long signals
5. Backtest different entry-delay and holding-period combinations
6. Evaluate performance using:
   - Mean return
   - Median return
   - Win rate
   - Volatility
   - Sharpe ratio
   - t-statistic
7. Analyze yearly stability and recent alpha decay

## Research Goal

The main objective is to answer the following questions:

- Do strong earnings signals lead to statistically significant post-announcement returns?
- How sensitive is performance to **entry timing** and **holding period**?
- Is the signal stable across years, or does it decay under changing market conditions?

## Strategy Logic

The strategy is a **rule-based long-only PEAD model**.

Each earnings event is scored using a combination of features:

- **SUE**  
  Measures standardized earnings surprise relative to a stock’s own historical earnings surprise volatility.

- **EPS surprise percentage**  
  Captures the magnitude of the earnings beat or miss.

- **Revenue surprise percentage**  
  Adds top-line confirmation to the signal.

- **Gap percentage**  
  Measures whether price action confirms the earnings information.

- **Volume ratio**  
  Confirms whether the event is accompanied by abnormal trading activity.

- **Guidance flag**  
  Reserved for management guidance direction where available.

Each event receives a composite score. If the score passes a threshold, the event is classified as a **LONG** signal; otherwise it is ignored.

### Why this design?

The model is intentionally simple and interpretable. Instead of training a complex black-box model, it uses a transparent scoring system to test whether a small set of economically intuitive features can capture PEAD-like drift.

This design makes it easier to:
- understand which features drive the signal,
- defend the logic in interviews,
- and iterate on the strategy later.

## Key Results

Across all tested entry-lag and holding-period combinations, the strongest configuration was:

- **Entry lag:** 3 trading days
- **Holding period:** 20 trading days

This configuration achieved:

- **Mean return:** 3.11%
- **Median return:** 2.30%
- **Win rate:** 61.40%
- **Volatility (std):** 13.05%
- **Sharpe ratio:** 0.2383
- **t-stat:** 6.0517

### Interpretation

These results support the PEAD hypothesis:

- Performance generally improves as the holding period increases
- The strongest configurations occur at longer holding windows
- The best t-stat is also achieved at the same configuration that has the highest mean return

This suggests that earnings information is **not fully priced immediately**, and the price adjustment continues over subsequent trading days.

## Top Configurations

| entry_lag | hold_days | mean | median | std | sharpe | t_stat | win_rate |
|----------|-----------|------|--------|-----|--------|--------|----------|
| 3 | 20 | 3.11% | 2.30% | 13.05% | 0.2383 | 6.0517 | 61.40% |
| 2 | 20 | 2.71% | 1.76% | 12.55% | 0.2162 | 5.4910 | 58.29% |
| 1 | 20 | 2.82% | 2.22% | 13.22% | 0.2135 | 5.4213 | 58.76% |
| 5 | 20 | 2.51% | 2.09% | 12.69% | 0.1981 | 5.0232 | 60.50% |
| 3 | 10 | 1.80% | 1.69% | 9.90% | 0.1818 | 4.6201 | 58.82% |

## Robustness and Regime Change

A year-by-year breakdown shows that the strategy worked much better in some regimes than others.

Earlier years showed stronger and more persistent post-earnings drift, while recent results suggest **alpha decay**, especially in 2025–2026.

Possible explanations include:

- faster information diffusion,
- more crowded event-driven trading,
- reduced market inefficiency,
- regime shifts in market structure.

This makes the project more realistic: the goal is not just to find a profitable configuration in-sample, but also to understand **when and why the signal weakens**.

## Project Structure

```text
earnings_scanner/
├── README.md
├── watchlist.txt
├── archive/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── tables/
│   └── charts/
├── research/
│   └── earnings_research.md
└── src/
    ├── analysis/
    │   ├── build_summary_by_window.py
    │   └── plot_strategy_results.py
    ├── backtest/
    │   ├── performance_metrics.py
    │   └── run_performance_analysis.py
    ├── data/
    │   ├── build_earnings_features.py
    │   ├── build_earnings_research_dataset.py
    │   ├── build_watchlist_earnings_dataset.py
    │   ├── earnings_api.py
    │   ├── historical_earnings.py
    │   └── price_api.py
    └── signal/
        └── pead_strategy.py



## Data Pipeline

The project follows a structured event-driven research workflow.

---

### Step 1 — Collect Raw Earnings Calendar  

Fetch historical earnings data from FMP API.

**Output**  
- `data/raw/watchlist_earnings_calendar.csv`

---

### Step 2 — Clean Earnings Data  

Clean and standardize earnings events.

**Output**  
- `data/raw/watchlist_earnings_calendar_clean.csv`

---

### Step 3 — Construct Event-Level Dataset  

Compute forward returns for each earnings event.

**Output**  
- `data/processed/earnings_research_dataset.csv`

---

### Step 4 — Feature Engineering  

Generate predictive features (EPS surprise, SUE, gap, volume, etc.).

**Output**  
- `data/processed/earnings_features_2018_2026.csv`

---

### Step 5 — Signal Generation  

Apply rule-based PEAD model to classify events.

**Output**  
- `data/processed/pead_signals_2018_2026.csv`

---

### Step 6 — Signal Validation & Trade Construction  

Validate signal effectiveness and simulate trade outcomes.

- Compute forward returns (1d / 3d / 5d / 10d / 20d)  
- Separate long / short performance  
- Generate trade-level dataset  

**Outputs**  
- `data/processed/validated_signals_2018_2026.csv`  
- `data/processed/validated_summary_all_2018_2026.csv`  
- `data/processed/validated_summary_by_side_2018_2026.csv`  
- `data/processed/validated_summary_by_year_2018_2026.csv`  
- `data/processed/validated_summary_by_year_and_side_2018_2026.csv`  
- `data/processed/entry_lag_detail_2018_2026.csv`

---

### Step 7 — Strategy Backtesting  

Evaluate parameter combinations:

- Entry delay  
- Holding period  

**Input**  
- `data/processed/entry_lag_detail_2018_2026.csv`

**Outputs**  
- `data/processed/summary_by_window_and_entry_day_2018_2026.csv`  
- `data/processed/yearly_summary_by_window_and_entry_day_2018_2026.csv`

---

### Step 8 — Visualization  

Generate charts for performance analysis.

**Outputs**  
- `outputs/charts/mean_return_vs_holding.png`  
- `outputs/charts/tstat_vs_holding.png`  
- `outputs/charts/best_combo_yearly_mean_return.png`  
- `outputs/charts/yearly_mean_heatmap_entry3.png`


## Strategy Explanation

This strategy is based on the Post-Earnings Announcement Drift (PEAD) hypothesis, which suggests that markets do not fully incorporate earnings information immediately. As a result, stock prices may continue to drift in the direction of earnings surprises after the announcement.

To capture this effect, the strategy constructs a rule-based signal using a combination of fundamental and price-based features.

First, **EPS surprise** and **revenue surprise** are used to measure the magnitude of the earnings beat or miss relative to market expectations. These variables capture the core information content of the earnings release.

Second, the model incorporates **SUE (Standardized Unexpected Earnings)**, which normalizes earnings surprises by their historical volatility. This allows the strategy to distinguish between statistically significant surprises and noise.

Third, the strategy includes **price confirmation signals**, such as gap percentage and volume ratio. These features are used to validate whether the market reaction is supported by strong price movement and abnormal trading activity.

All features are combined into a rule-based scoring system. Only events that exceed a predefined threshold are selected as long signals.

The strategy then evaluates performance across different **entry delays** and **holding periods**. Empirical results show that delaying entry by a few days and holding positions for longer horizons generally leads to stronger performance. This is consistent with the gradual information diffusion mechanism implied by PEAD.

However, the performance is not stable across all time periods. More recent data shows weaker results, suggesting potential alpha decay due to increased market efficiency and more crowded event-driven strategies.