# AETHERQUANT // Deep Learning Stock Forecasting & Financial Backtester
### *Traditional Statistical Baselines (Auto-ARIMA, Holt-Winters) vs. PyTorch Sequence Networks (LSTM, GRU, Bahdanau Attention-LSTM, and Multi-Head Self-Attention Transformers) with a Remix.run-Inspired Bento-Grid Dashboard*

---

Aetherquant is an end-to-end quantitative research framework, predictive time-series pipeline, and interactive data visualization cockpit. Designed to model sequential financial asset datasets, it bridges robust statistical modeling with state-of-the-art deep learning architectures. 

The framework is paired with a zero-dependency Python bridge server and a high-fidelity, kinetic-animated bento-grid dashboard drawing design inspiration from the sleek, minimalist, pitch-black visual aesthetic of **[Remix.run](https://remix.run/)**.

---

## 🎨 Interactive Dashboard (Remix.run Visual Language)

The newly integrated web interface provides an immersive, cyberpunk-inspired quantitative cockpit:
* **Pitch-Black Glassmorphism**: Utilizes solid `#000000` base variables, translucent panel backgrounds (`rgba(18, 18, 18, 0.45)`), backdrop blurs, and sharp borders.
* **Kinetic Neural Particle Canvas**: A high-frame-rate background particle canvas running interactive self-attention node links that subtly repel from the user's cursor.
* **Asymmetric 12-Column Bento Grid**: All control panels, dials, and training shell modules align within a clean grid, floating into place via staggered CSS animations.
* **Interactive Attention Weights Heatmap**: Displays Bahdanau sequence attention distributions dynamically. Hovering over sequence elements triggers the active weight inspector card to show predictive significance.
* **Momentum Oscillator Dials**: Gauges real-time momentum indicators (RSI) using animated mechanical gauge dials.
* **Subprocess Compiler logs**: A live, multi-threaded subprocess log terminal displaying epoch progresses and network weight initializations on the fly.

---

## ⚙️ Core Architectures & Modeling Laboratory

Aetherquant equips researchers with a comparative sandbox of 6 core quantitative models:

1. **Auto-ARIMA (Classical Baseline)**: Fits a linear Auto-Regressive Integrated Moving Average parameter grid optimized using Akaike Information Criterion (AIC) minimizations.
2. **Holt-Winters Exponential Smoothing**: A classical baseline configured with damped trends and seasonality matrices to project rolling statistical metrics.
3. **Multivariate LSTM (PyTorch)**: A Recurrent Neural Network (RNN) capturing multi-feature historical sequences (Volume, Close, EMA, SMA, Bollinger Bands, RSI, MACD).
4. **Multivariate GRU (PyTorch)**: A Gated Recurrent Unit network comparing sequence training speed and directional forecasts.
5. **Bahdanau Temporal Attention-LSTM (Seq2Seq)**: An encoder-decoder architecture computing custom mathematical alignment coefficients across historical sequences (e.g. t-20 to t-1), indicating precisely which historical days impact predictions.
6. **Time-Series Transformer (PyTorch Self-Attention)**: Implements query-key-value parameter projections, sinusoidal Positional Encodings, and multi-head self-attention heads to model long-range sequential numeric dependencies.

---

## 📁 Repository Directory Structure

```text
Stock forecasting/
├── .gitignore                <- Excludes Python bytecode, Node modules, build dist, and bulk raw CSV datasets.
├── requirements.txt          <- System dependencies (PyTorch, Pandas, yfinance, Statsmodels).
├── run_pipeline.py           <- Unified CLI orchestrator script running full end-to-end downloads & neural training.
├── server.py                 <- Multi-threaded HTTP JSON REST API server and static file bridge serving the UI bundle.
├── AETHERQUANT_SYSTEM_SPECIFICATIONS.txt  <- Comprehensive system specification sheet.
├── README.md                 <- [This File] Main documentation.
│
├── data/
│   ├── raw/                  <- Temporary cache folder for yfinance downloaded CSVs (git ignored).
│   └── processed/            <- Storage for engineered technical indicator datasets (git ignored).
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py      <- Fetches NSE market data, checks bounds, handles local directory mapping & caching.
│   ├── feature_engineering.py <- Pure Pandas implementations of RSI, MACD, Bollinger Bands, rolling stats, lookback lags.
│   ├── evaluation.py         <- Quantitative backtest simulation engine (calculates Sharpe, Drawdowns, Win Rates).
│   ├── utils.py              <- Console logger, plot styling overrides, and attention weight inspector wrappers.
│   └── models/
│       ├── __init__.py
│       ├── baselines.py      <- ARIMA and Holt-Winters wrappers.
│       ├── lstm_gru.py       <- PyTorch recurrent sequence loops (LSTM and GRU layers).
│       ├── attention.py      <- Gated encoder-decoder network utilizing custom Bahdanau temporal attention scoring.
│       └── transformer.py    <- Time-Series Transformer using multi-head self-attention.
│
├── notebooks/
│   ├── 01_pipeline_and_baselines.ipynb  <- Exploratory data analysis, indicators, ADF test, classical fits.
│   └── 02_deep_learning_forecasting.ipynb  <- PyTorch DataLoader design, training loops, attention maps, backtesting.
│
├── results/
│   ├── metrics/              <- Performance evaluation CSV records.
│   ├── models/               <- Saved PyTorch model checkpoint dictionaries (.pth).
│   └── plots/                <- Standard visual PNG forecast overlays, backtest curves, and static attention heatmaps.
│
└── dashboard/                <- High-fidelity web cockpit
    ├── index.html            <- Heavyweight bento-grid skeleton and Hero section.
    ├── package.json          <- Node scripting for Vite compiler.
    └── src/
        ├── main.js           <- State engine managing REST sync, particle canvas loops, and dynamic attention grids.
        └── style.css         <- Sleek, high-contrast, pitch-black typography system and bento structures.
```

---

## 🚀 Getting Started & Local Deployment

### 1. Prerequisites & Environment Setup
Make sure you have **Python >= 3.8** (compatible with Python 3.13) installed. Open your terminal in the project root directory and run:

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Interactive Bento-Grid Dashboard Cockpit
To view the animated Remix.run-style Bento Grid trading cockpit, start the integrated REST bridge server:

```bash
# Launch the bridge devserver
python server.py
```

Open your browser and navigate to **[http://localhost:8000](http://localhost:8000)**. 
* *Note: The server features dynamic price-scaled fallback metrics for TCS, Infosys (INFY), and HDFC Bank. You can immediately click through tickers, inspect self-attention columns, and evaluate backtests without any pre-existing local files.*

---

## 🏃 Run the Neural Training CLI

You can also trigger data downloads, feature engineering, statistical benchmarks fitting, PyTorch deep learning models training, backtesting simulations, and PNG graphics plotting directly from your terminal using the single-entrypoint orchestrator pipeline:

```bash
# Execute with standard defaults (Reliance - RELIANCE.NS)
python run_pipeline.py

# Execute for custom tickers (e.g. Tata Consultancy Services - TCS) with custom parameters
python run_pipeline.py --ticker TCS.NS --lookback 30 --epochs 25 --batch_size 16 --lr 0.0005
```

### Supported CLI Flags:
* `--ticker`: NSE stock ticker name, defaults to `'RELIANCE.NS'`. Supports `'TCS.NS'`, `'HDFCBANK.NS'`, `'INFY.NS'`, etc.
* `--lookback`: Context sequence days supplied to the PyTorch networks, defaults to `20`.
* `--horizon`: Forecast horizon days ahead, defaults to `1` (predicting tomorrow's price).
* `--epochs`: Epoch training runs for neural architectures, defaults to `15` (increase to 50+ for optimal training).
* `--batch_size`: DataLoader batch size, defaults to `32`.
* `--lr`: Learning rate, defaults to `0.001`.
* `--force_fetch`: Bypass cache and download fresh yfinance market metrics.

---

## 📓 Interactive Exploration (Jupyter Notebooks)

Prefer step-by-step visual exploration? Launch your Jupyter server:
```bash
jupyter notebook
```
And explore:
1. **`notebooks/01_pipeline_and_baselines.ipynb`**: Handles stock loading, technical indicators engineering, ADF series stationarity checks, and statistical fits (ARIMA & Exponential Smoothing).
2. **`notebooks/02_deep_learning_forecasting.ipynb`**: Full tutorial training PyTorch networks, generating Seq2Seq attention maps, and performing cumulative financial trading simulations.
