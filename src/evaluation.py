import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils import logger

def calculate_forecasting_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculates key time series forecasting performance metrics.
    """
    # Flatten arrays
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()
    
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    
    # Avoid division by zero in MAPE
    non_zero = y_true_flat != 0
    mape = np.mean(np.abs((y_true_flat[non_zero] - y_pred_flat[non_zero]) / y_true_flat[non_zero])) * 100.0
    
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # Directional Accuracy (predicting positive/negative change)
    # y_true_diff = y_true[t+horizon] - y_true[t]
    actual_direction = np.sign(np.diff(y_true_flat))
    pred_direction = np.sign(np.diff(y_pred_flat))
    
    # Pad to match sizes in case of multiple horizons or flat sequences
    # For simplified directional accuracy on steps
    dir_acc = np.mean(actual_direction == pred_direction) * 100.0 if len(actual_direction) > 0 else 0.0
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': dir_acc
    }
    
    logger.info(f"=== {model_name} Metrics ===")
    logger.info(f"  MAE:                  {mae:.4f}")
    logger.info(f"  RMSE:                 {rmse:.4f}")
    logger.info(f"  MAPE:                 {mape:.4f}%")
    logger.info(f"  R2 score:             {r2:.4f}")
    logger.info(f"  Directional Accuracy: {dir_acc:.2f}%")
    
    return metrics

class QuantitativeBacktester:
    """
    Simulates simple trading strategies based on forecasting model predictions
    and calculates financial risk and performance metrics.
    """
    def __init__(self, initial_capital=100000.0, risk_free_rate=0.06):
        """
        Args:
            initial_capital: Float representing baseline cash
            risk_free_rate: Annualized risk free interest rate (defaults to 6% common in India)
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

    def backtest(self, dates, actual_prices, predicted_prices, strategy_type='long_only'):
        """
        Backtests the predicted signals against real price paths.
        
        Args:
            dates: Datetime series corresponding to trading days
            actual_prices: 1D Array of actual closing prices
            predicted_prices: 1D Array of corresponding forecast prices (matching alignment)
            strategy_type: 'long_only' or 'long_short'
            
        Returns:
            summary_stats: Dict of financial performance indicators
            equity_curve: Numpy array representing capital path
        """
        actual_prices = np.array(actual_prices)
        predicted_prices = np.array(predicted_prices)
        
        n_periods = len(actual_prices)
        if n_periods < 2:
            return {}, np.array([])
            
        # Daily actual market returns
        market_returns = np.diff(actual_prices) / actual_prices[:-1]
        
        # Signals generation:
        # If predicted price tomorrow is higher than today's actual close, Buy.
        # predicted_prices represent tomorrow's prediction.
        # Let's align signals: if predicted tomorrow is greater than today's actual close, signal = +1
        predicted_change = predicted_prices[1:] - actual_prices[:-1]
        
        signals = np.zeros(n_periods - 1)
        
        if strategy_type == 'long_only':
            signals[predicted_change > 0] = 1.0  # Buy
        elif strategy_type == 'long_short':
            signals[predicted_change > 0] = 1.0   # Go Long
            signals[predicted_change <= 0] = -1.0 # Short Sell
            
        # Strategy daily returns
        strategy_returns = signals * market_returns
        
        # Portfolio value / equity curve
        equity_curve = np.zeros(n_periods)
        equity_curve[0] = self.initial_capital
        
        for t in range(1, n_periods):
            # return at t-1 applies from close of t-1 to close of t
            ret = strategy_returns[t - 1]
            equity_curve[t] = equity_curve[t - 1] * (1.0 + ret)
            
        # Benchmark equity curve: Buy & Hold
        benchmark_curve = np.zeros(n_periods)
        benchmark_curve[0] = self.initial_capital
        for t in range(1, n_periods):
            benchmark_curve[t] = benchmark_curve[t - 1] * (1.0 + market_returns[t - 1])
            
        # Compute summary statistics
        total_days = n_periods - 1
        years = total_days / 252.0
        
        final_value = equity_curve[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100.0
        annualized_return = ((final_value / self.initial_capital) ** (1.0 / max(years, 0.001)) - 1.0) * 100.0
        
        # Volatility
        daily_vol = np.std(strategy_returns)
        annualized_vol = daily_vol * np.sqrt(252.0) * 100.0
        
        # Sharpe Ratio (annualized)
        daily_rf = (1.0 + self.risk_free_rate) ** (1.0 / 252.0) - 1.0
        excess_returns = strategy_returns - daily_rf
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        
        sharpe_ratio = (mean_excess / (std_excess + 1e-10)) * np.sqrt(252.0)
        
        # Maximum Drawdown
        peaks = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - peaks) / peaks
        max_drawdown = np.min(drawdowns) * 100.0
        
        # Win Rate (positive returns daily count)
        win_rate = np.mean(strategy_returns > 0) * 100.0
        
        # Benchmark comparison
        bench_final = benchmark_curve[-1]
        bench_return = (bench_final - self.initial_capital) / self.initial_capital * 100.0
        
        stats = {
            'Strategy Type': strategy_type,
            'Total Return (%)': total_return,
            'Annualized Return (%)': annualized_return,
            'Annualized Volatility (%)': annualized_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Win Rate (%)': win_rate,
            'Buy & Hold Return (%)': bench_return
        }
        
        logger.info(f"Strategy performance: Total Return={total_return:.2f}%, Sharpe={sharpe_ratio:.2f}, MaxDD={max_drawdown:.2f}%")
        return stats, equity_curve, benchmark_curve
