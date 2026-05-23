import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from src.utils import logger, ensure_directories

DEFAULT_TICKERS = [
    'RELIANCE.NS',
    'TCS.NS',
    'HDFCBANK.NS',
    'INFY.NS',
    'ICICIBANK.NS'
]

def validate_and_clean_data(df, ticker):
    """
    Validates and cleans stock market data to ensure pipeline integrity.
    Checks for null values, proper indexing, chronological sorting, and expected columns.
    """
    if df is None or df.empty:
        raise ValueError(f"Downloaded dataframe for {ticker} is empty or None.")

    # Flatten columns in case of multi-indexing (common in yfinance download)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardize columns to standard casing
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in expected_cols:
        if col not in df.columns:
            # Handle minor naming discrepancies
            for actual_col in df.columns:
                if actual_col.lower() == col.lower():
                    df = df.rename(columns={actual_col: col})
                    break
            else:
                if col == 'Adj Close' and 'Close' in df.columns:
                    df['Adj Close'] = df['Close']
                else:
                    raise ValueError(f"Required column '{col}' is missing in downloaded data for {ticker}. Columns: {df.columns.tolist()}")

    # Format Index to DateTime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError(f"Failed to convert DataFrame index to DatetimeIndex: {e}")

    # Remove any timezone info if present to avoid indexing issues in statsmodels/pytorch
    df.index = df.index.tz_localize(None)

    # Sort chronologically
    df = df.sort_index()

    # Drop duplicate index values if any
    if df.index.duplicated().any():
        logger.warning(f"Duplicate dates detected for {ticker}. Dropping duplicates.")
        df = df[~df.index.duplicated(keep='first')]

    # Check for missing values
    null_counts = df[expected_cols].isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Null values detected in {ticker} data:\n{null_counts[null_counts > 0]}. Forward filling...")
        df[expected_cols] = df[expected_cols].ffill().bfill()

    # Verify again
    if df[expected_cols].isnull().any().any():
        raise ValueError(f"Unresolved null values in essential columns for {ticker}.")

    logger.debug(f"Data validation passed for {ticker}. Shape: {df.shape}")
    return df

def get_or_fetch_data(ticker, start_date='2018-01-01', end_date=None, force_fetch=False, cache_expiry_hours=24):
    """
    Retrieves data for a given ticker from yfinance or local cache if available and not expired.
    
    Args:
        ticker: Ticker symbol (e.g. 'RELIANCE.NS')
        start_date: Start date for fetching data (YYYY-MM-DD)
        end_date: End date for fetching data (YYYY-MM-DD, defaults to current date)
        force_fetch: If True, bypass cache and fetch fresh data
        cache_expiry_hours: Hours after which cache is considered stale
    """
    ensure_directories()
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    cache_path = os.path.join('data', 'raw', f"{ticker.replace('.NS', '')}_raw.csv")
    
    # Check cache status
    use_cache = False
    if os.path.exists(cache_path) and not force_fetch:
        file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        cache_age = datetime.now() - file_mtime
        if cache_age < timedelta(hours=cache_expiry_hours):
            use_cache = True
            logger.info(f"Using cached raw data for {ticker}. Cache age: {cache_age.total_seconds() / 3600:.1f} hours.")
        else:
            logger.info(f"Cache for {ticker} is older than {cache_expiry_hours} hours. Re-fetching...")

    if use_cache:
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            # Filter by date range in cache
            df = df.loc[start_date:end_date]
            if not df.empty:
                return df
            else:
                logger.warning(f"Cache read returned empty dataset inside desired dates. Re-fetching...")
        except Exception as e:
            logger.error(f"Failed to read raw cache file {cache_path}: {e}. Fetching fresh...")

    # Fetch fresh data
    logger.info(f"Downloading {ticker} data from yfinance for range {start_date} to {end_date}...")
    try:
        # Download with progress bar disabled for cleaner logs
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        df = validate_and_clean_data(df, ticker)
        
        # Save to cache
        df.to_csv(cache_path)
        logger.info(f"Downloaded and cached {len(df)} records for {ticker} at {cache_path}.")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching stock data for {ticker}: {e}")
        # Fallback to cache even if stale, as backup
        if os.path.exists(cache_path):
            logger.warning(f"Downloading failed. Loading stale cache as fallback.")
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                return df.loc[start_date:end_date]
            except Exception as read_err:
                logger.error(f"Failed loading stale cache: {read_err}")
        
        raise e

def run_ingestion_pipeline(tickers=None, start_date='2018-01-01', end_date=None, force_fetch=False):
    """
    Orchestrates downloading data for multiple stock tickers.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS
        
    logger.info("Initializing Data Ingestion Pipeline...")
    datasets = {}
    
    for ticker in tickers:
        try:
            df = get_or_fetch_data(ticker, start_date, end_date, force_fetch)
            datasets[ticker] = df
        except Exception as e:
            logger.error(f"Skipping ticker {ticker} due to fatal pipeline error: {e}")
            
    logger.info(f"Data Ingestion Pipeline completed. Successfully fetched: {list(datasets.keys())}")
    return datasets

if __name__ == '__main__':
    # Fast standalone verification
    run_ingestion_pipeline(['RELIANCE.NS'])
