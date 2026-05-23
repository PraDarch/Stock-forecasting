import numpy as np
import pandas as pd
import os
from statsmodels.tsa.stattools import adfuller
from src.utils import logger, ensure_directories

def add_technical_indicators(df):
    """
    Computes technical indicators using optimized pandas and numpy operations.
    Avoids ta-lib binary dependency for full compatibility on Windows/Python 3.13.
    """
    logger.info("Computing technical indicators and features...")
    data = df.copy()

    # 1. Price Returns
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # 2. Simple and Exponential Moving Averages (SMA/EMA)
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

    # 3. Volatility
    data['Volatility_20'] = data['Returns'].rolling(window=20).std()

    # 4. Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Use exponential moving average for smoothing RSI gains/losses (Wilder's smoothing)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10) # Avoid divide by zero
    data['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    # 5. MACD (Moving Average Convergence Divergence)
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

    # 6. Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']

    # 7. Lag features (Lags: 1, 2, 3, 5, 10)
    for lag in [1, 2, 3, 5, 10]:
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
        data[f'Returns_Lag_{lag}'] = data['Returns'].shift(lag)

    # 8. Rolling statistics
    data['Close_Rolling_Mean_7'] = data['Close'].rolling(window=7).mean()
    data['Close_Rolling_Std_7'] = data['Close'].rolling(window=7).std()

    # 9. Calendar / Time-based features
    data['Day_of_Week'] = data.index.dayofweek
    data['Month'] = data.index.month
    data['Quarter'] = data.index.quarter

    logger.info(f"Feature engineering completed. Columns added. Total features: {len(data.columns)}")
    return data

def run_stationarity_analysis(series, name="Series"):
    """
    Performs Augmented Dickey-Fuller (ADF) test for stationarity and logs the results.
    """
    logger.info(f"Running stationarity analysis on: {name}...")
    clean_series = series.dropna()
    if len(clean_series) < 20:
        logger.warning(f"Series {name} is too short for stationarity analysis.")
        return False
        
    try:
        result = adfuller(clean_series, autolag='AIC')
        adf_stat = result[0]
        p_val = result[1]
        crit_vals = result[4]
        
        is_stationary = p_val <= 0.05
        
        logger.info(f"ADF Results for {name}:")
        logger.info(f"  - ADF Statistic: {adf_stat:.6f}")
        logger.info(f"  - p-value: {p_val:.6f}")
        logger.info(f"  - Critical Values: 1%={crit_vals['1%']:.3f}, 5%={crit_vals['5%']:.3f}, 10%={crit_vals['10%']:.3f}")
        
        if is_stationary:
            logger.info(f"  → {name} is STATIONARY (p <= 0.05). Reject H0.")
        else:
            logger.info(f"  → {name} is NON-STATIONARY (p > 0.05). Fail to reject H0.")
            
        return is_stationary
    except Exception as e:
        logger.error(f"Error performing stationarity analysis on {name}: {e}")
        return False

def generate_and_save_features(ticker, df, force_run=False):
    """
    Generates all features for a ticker and saves the processed dataset to data/processed.
    """
    ensure_directories()
    processed_path = os.path.join('data', 'processed', f"{ticker.replace('.NS', '')}_features.csv")
    
    if os.path.exists(processed_path) and not force_run:
        logger.info(f"Loading pre-existing features from {processed_path}")
        return pd.read_csv(processed_path, index_col=0, parse_dates=True)
        
    # Generate indicators
    df_feat = add_technical_indicators(df)
    
    # Run stationarity analysis on key series as validation
    run_stationarity_analysis(df_feat['Close'], f"{ticker} Close Price")
    run_stationarity_analysis(df_feat['Returns'], f"{ticker} Daily Returns")
    run_stationarity_analysis(df_feat['Log_Returns'], f"{ticker} Log Returns")
    
    # Drop rows with NaN values created by lag calculations/rolling windows
    nan_count = df_feat.isnull().any(axis=1).sum()
    logger.info(f"Dropping {nan_count} rows containing NaN values from initial rolling steps.")
    df_feat = df_feat.dropna()
    
    # Save features
    df_feat.to_csv(processed_path)
    logger.info(f"Saved processed features to {processed_path}. Shape: {df_feat.shape}")
    
    return df_feat

if __name__ == '__main__':
    # Fast standalone validation
    from src.data_pipeline import get_or_fetch_data
    df_raw = get_or_fetch_data('RELIANCE.NS')
    df_features = generate_and_save_features('RELIANCE.NS', df_raw)
