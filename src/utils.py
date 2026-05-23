import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def setup_logger(log_file='pipeline.log'):
    """
    Sets up a dual-destination logger: console and file output.
    """
    logger = logging.getLogger('StockForecasting')
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup is run multiple times
    if logger.handlers:
        return logger

    # Create formatters
    detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    return logger

# Get local logger
logger = setup_logger()

def ensure_directories():
    """
    Ensures that the required directories for the pipeline are present.
    """
    dirs = [
        'data/raw',
        'data/processed',
        'results/plots',
        'results/models',
        'results/metrics'
    ]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            logger.info(f"Created directory: {d}")
        else:
            logger.debug(f"Directory already exists: {d}")

def setup_plotting_style():
    """
    Configures standard premium plotting styles using Seaborn and Matplotlib.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (14, 7)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['grid.alpha'] = 0.3
    logger.info("Plotting styles initialized successfully.")

def plot_predictions(dates, y_true, predictions, ticker, output_path=None):
    """
    Plots true values vs model predictions.
    
    Args:
        dates: Array-like index of dates
        y_true: Ground truth stock prices
        predictions: Dictionary of {model_name: predicted_values}
        ticker: String ticker symbol (e.g. 'RELIANCE')
        output_path: Filepath to save the figure
    """
    setup_plotting_style()
    plt.figure(figsize=(14, 7))
    
    # Plot true values
    plt.plot(dates, y_true, label='Actual Close Price', color='#2B2D42', linewidth=2.0, alpha=0.9)
    
    # Visual color palette for multiple models
    colors = {
        'ARIMA': '#D90429',
        'Exp Smoothing': '#F77F00',
        'LSTM': '#00B4D8',
        'GRU': '#7209B7',
        'Attention-LSTM': '#4CC9F0',
        'Transformer': '#38B000'
    }
    
    for name, y_pred in predictions.items():
        color = colors.get(name, np.random.rand(3,))
        plt.plot(dates, y_pred, label=f'{name} Forecast', color=color, linewidth=1.8, linestyle='--')
        
    plt.title(f"{ticker} Multi-Horizon Stock Price Forecast Comparison", fontsize=15, pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (INR)", fontsize=12)
    plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
        logger.info(f"Forecast plot saved to {output_path}")
    plt.close()

def plot_backtest_results(dates, cum_returns_dict, ticker, output_path=None):
    """
    Plots the cumulative returns of various strategies compared to a Buy-and-Hold benchmark.
    
    Args:
        dates: Date indexes
        cum_returns_dict: Dictionary of {strategy_name: cumulative_returns_array}
        ticker: String ticker symbol
        output_path: Filepath to save the figure
    """
    setup_plotting_style()
    plt.figure(figsize=(14, 7))
    
    colors = {
        'Buy & Hold': '#2B2D42',
        'ARIMA Strategy': '#D90429',
        'Exp Smoothing Strategy': '#F77F00',
        'LSTM Strategy': '#00B4D8',
        'GRU Strategy': '#7209B7',
        'Attention-LSTM Strategy': '#4CC9F0',
        'Transformer Strategy': '#38B000'
    }
    
    for name, returns in cum_returns_dict.items():
        color = colors.get(name, np.random.rand(3,))
        # Convert returns to percentage change since start
        pct_returns = (returns - 1.0) * 100.0
        plt.plot(dates, pct_returns, label=name, color=color, linewidth=2.0 if 'Hold' in name else 1.6)
        
    plt.title(f"Trading Strategy Cumulative Returns Backtest ({ticker})", fontsize=15, pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Returns (%)", fontsize=12)
    plt.legend(loc='upper left', frameon=True, facecolor='white')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
        logger.info(f"Backtest cumulative returns plot saved to {output_path}")
    plt.close()

def plot_attention_weights(attention_weights, x_labels, y_labels, output_path=None):
    """
    Plots a heatmap of temporal attention weights to visualize model focus.
    
    Args:
        attention_weights: Numpy array of shape (features or targets, time_steps) or (time_steps, time_steps)
        x_labels: Labels for x axis (e.g. lookback window index or dates)
        y_labels: Labels for y axis (e.g. feature names or output days)
        output_path: Filepath to save the figure
    """
    setup_plotting_style()
    plt.figure(figsize=(12, 6))
    
    sns.heatmap(attention_weights, xticklabels=x_labels, yticklabels=y_labels, 
                cmap='viridis', annot=True, fmt=".2f", cbar_kws={'label': 'Attention Weight'})
                
    plt.title("Temporal Attention Weight Distribution Map", fontsize=14, pad=15)
    plt.xlabel("Lookback Timesteps (t-N to t-1)", fontsize=12)
    plt.ylabel("Prediction Step or Feature", fontsize=12)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
        logger.info(f"Attention weights heatmap saved to {output_path}")
    plt.close()
