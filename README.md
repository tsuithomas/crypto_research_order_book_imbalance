# Cryptocurrency Statistical Arbitrage & Microstructure Alpha Research

## Project Overview
This repository contains a high-performance quantitative research pipeline designed to extract microstructural alpha from crypto delta-one instruments (Spot/Perpetuals). The project focuses on rapidly prototyping predictive signals from Level-2 order book dynamics, specifically targeting Order Book Imbalance (OBI), and evaluating them through rigorous out-of-sample walk-forward backtesting.

**Key Objectives Achieved:**
* **Asynchronous Data Engineering:** Bypassed I/O bottlenecks to ingest millions of high-frequency tick data rows using `aiohttp` and Parquet columnar storage.
* **Vectorized Feature Engineering:** Engineered Order Book Imbalance (OBI) and volume-weighted microprice metrics using highly optimized, vectorized `pandas` and `numpy` operations to ensure rapid processing of large-scale financial time-series data.
* **Machine Learning:** Implemented an XGBoost regressor for 5-minute forward log-return prediction, proving OBI's predictive dominance (~45% feature importance) over standard volume/volatility metrics.
* **Risk-Managed Backtesting:** Built a custom, vectorized simulation engine to model the severe impact of crypto taker fees, demonstrating the necessity of limit-order (maker) execution assumptions to achieve positive Expected Value (EV) and a positive Sharpe Ratio.

---

## System Architecture

The project is strictly modularized to separate data ingestion, feature generation, modeling, and simulation:

1. `src/data_ingestion.py`: Asynchronous batch downloader for Binance public aggregate trades.
2. `src/feature_engineering.py`: Data normalization, time-series resampling, and vectorized feature matrix construction.
3. `src/model_training.py`: Walk-forward cross-validation and XGBoost model training.
4. `src/backtester.py`: Vectorized trading simulation applying strict transaction cost analysis.

---

## Prerequisites & Installation

**1. Environment Setup**
Ensure you are using Python 3.11+. Clone the repository and create an isolated virtual environment:

    git clone https://github.com/yourusername/crypto_alpha_research.git
    cd crypto_alpha_research
    python3 -m venv .venv
    source .venv/bin/activate

**2. Install Python Dependencies**
Ensure your `requirements.txt` includes `pandas`, `numpy`, `pyarrow`, `aiohttp`, `xgboost`, and `scikit-learn`.

    pip install -r requirements.txt

**3. macOS Specific Requirements (Apple Silicon / Intel)**
If you are running this on a Mac, you must install the OpenMP library for XGBoost's multithreading capabilities, and ensure your Python SSL certificates are linked:

    brew install libomp
    /Applications/Python\ 3.11/Install\ Certificates.command

---

## Execution Pipeline

Execute the pipeline sequentially to replicate the research:

**Step 1: Ingest Data**
Downloads historical tick data into compressed `.parquet` files.

    python3 src/data_ingestion.py

**Step 2: Generate Feature Matrix**
Processes the raw ticks through the vectorized pandas engine and aligns forward returns.

    python3 src/feature_engineering.py

**Step 3: Train Machine Learning Model**
Splits data chronologically (preventing look-ahead bias) and trains the XGBoost predictor.

    python3 src/model_training.py

**Step 4: Run Walk-Forward Backtest**
Simulates execution and calculates the Sharpe Ratio, Net PnL, and Trade Count.

    python3 src/backtester.py

---

## Research Findings
Initial backtesting utilizing aggressive market-order execution (crossing the spread, 4-8 bps taker fee) resulted in negative net PnL, highlighting the structural drag of transaction costs on high-frequency signals. By recalibrating the execution logic to assume passive maker orders and applying a strict 15 bps predictive conviction threshold, the strategy successfully inverted to positive absolute returns (+3.21%) during a severe market drawdown (-12.76%), demonstrating true statistical alpha.