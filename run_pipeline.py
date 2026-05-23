import argparse
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# Core modules
from src.utils import (
    logger, 
    ensure_directories, 
    plot_predictions, 
    plot_backtest_results, 
    plot_attention_weights
)
from src.data_pipeline import get_or_fetch_data
from src.feature_engineering import generate_and_save_features
from src.models.baselines import BaselineModelWrapper
from src.models.lstm_gru import PyTorchTrainer
from src.models.attention import AttentionLSTMHelper
from src.models.transformer import TransformerHelper
from src.evaluation import calculate_forecasting_metrics, QuantitativeBacktester

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Horizon Stock Forecasting Pipeline Orchestrator")
    parser.add_argument('--ticker', type=str, default='RELIANCE.NS', help="NSE stock ticker (e.g. INFY.NS)")
    parser.add_argument('--start_date', type=str, default='2018-01-01', help="Start date YYYY-MM-DD")
    parser.add_argument('--lookback', type=int, default=20, help="Lookback window size (days)")
    parser.add_argument('--horizon', type=int, default=1, help="Forecasting horizon (days ahead)")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs for deep learning")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--hidden_dim', type=int, default=64, help="LSTM/GRU hidden layer dimension")
    parser.add_argument('--num_layers', type=int, default=2, help="Deep learning layers count")
    parser.add_argument('--force_fetch', action='store_true', help="Force download fresh data from yfinance")
    return parser.parse_args()

def run_pipeline(args):
    logger.info("=" * 60)
    logger.info(f"STARTING PIPELINE FOR {args.ticker.upper()} FORECASTING")
    logger.info("=" * 60)
    
    # 1. Setup folders
    ensure_directories()
    
    # 2. Ingest Data
    df_raw = get_or_fetch_data(args.ticker, start_date=args.start_date, force_fetch=args.force_fetch)
    
    # 3. Feature Engineering
    df_features = generate_and_save_features(args.ticker, df_raw)
    
    # 4. Train-Test Split (Chronological splitting)
    split_idx = int(len(df_features) * 0.8)
    train_val_data = df_features.iloc[:split_idx]
    test_data = df_features.iloc[split_idx:]
    
    # For PyTorch deep learning models, we split train_val further into train and validation (90/10)
    sub_split = int(len(train_val_data) * 0.9)
    train_data = train_val_data.iloc[:sub_split]
    val_data = train_val_data.iloc[sub_split:]
    
    # Define Target and Feature Columns
    target_col = 'Close'
    # Drop future-leaking columns or indices
    feature_cols = [c for c in df_features.columns if c not in ['Close', 'Adj Close', 'Returns', 'Log_Returns']]
    
    logger.info(f"Splits generated:")
    logger.info(f"  - Total samples: {len(df_features)}")
    logger.info(f"  - Training samples: {len(train_data)}")
    logger.info(f"  - Validation samples: {len(val_data)}")
    logger.info(f"  - Test samples: {len(test_data)}")
    logger.info(f"  - Features count: {len(feature_cols)}")
    
    # We will accumulate forecasting predictions and backtesting curves for comparison
    test_predictions = {}
    backtest_curves = {}
    backtest_stats = []
    
    # Setup test targets for matching alignments
    # Deep learning models require lookback padding to make predictions on the test set.
    # Therefore, predicted outputs will start exactly at `test_data.index[args.lookback + args.horizon - 1]`
    alignment_offset = args.lookback + args.horizon - 1
    aligned_dates = test_data.index[alignment_offset:]
    aligned_actuals = test_data[target_col].values[alignment_offset:]
    
    # ------------------ 5. Baseline Models ------------------
    # A. ARIMA
    arima_wrapper = BaselineModelWrapper(model_type='arima')
    arima_wrapper.fit(train_val_data[target_col])
    # Predict exactly matching the aligned test dataset size
    arima_raw_preds = arima_wrapper.predict(steps=len(test_data))
    # Align dates
    test_predictions['ARIMA'] = arima_raw_preds[alignment_offset:]
    
    # B. Exponential Smoothing
    exp_wrapper = BaselineModelWrapper(model_type='expsmoothing')
    exp_wrapper.fit(train_val_data[target_col])
    exp_raw_preds = exp_wrapper.predict(steps=len(test_data))
    test_predictions['Exp Smoothing'] = exp_raw_preds[alignment_offset:]
    
    # ------------------ 6. Deep Learning Models ------------------
    # Initialize trainer
    trainer = PyTorchTrainer(
        lookback=args.lookback,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=5
    )
    
    # A. LSTM
    logger.info("=" * 40)
    logger.info("TRAINING PYTORCH LSTM MODEL...")
    lstm_weights = os.path.join('results', 'models', f"{args.ticker.replace('.NS', '')}_lstm.pth")
    trainer.fit(train_data, val_data, feature_cols, target_col, model_type='lstm', save_path=lstm_weights)
    # Output shape: (samples, horizon). We extract step 1 predictions for daily comparison
    lstm_raw_preds = trainer.predict(test_data, feature_cols)
    test_predictions['LSTM'] = lstm_raw_preds[:, 0]
    
    # B. GRU
    logger.info("=" * 40)
    logger.info("TRAINING PYTORCH GRU MODEL...")
    gru_weights = os.path.join('results', 'models', f"{args.ticker.replace('.NS', '')}_gru.pth")
    trainer.fit(train_data, val_data, feature_cols, target_col, model_type='gru', save_path=gru_weights)
    gru_raw_preds = trainer.predict(test_data, feature_cols)
    test_predictions['GRU'] = gru_raw_preds[:, 0]
    
    # C. Attention-LSTM
    logger.info("=" * 40)
    logger.info("TRAINING ATTENTION-LSTM MODEL...")
    attn_helper = AttentionLSTMHelper(trainer)
    attn_weights_path = os.path.join('results', 'models', f"{args.ticker.replace('.NS', '')}_attn_lstm.pth")
    attn_helper.fit(train_data, val_data, feature_cols, target_col, save_path=attn_weights_path)
    attn_raw_preds = trainer.predict(test_data, feature_cols)
    test_predictions['Attention-LSTM'] = attn_raw_preds[:, 0]
    
    # Extract temporal attention weights for the heatmap plot
    mean_attention, _ = attn_helper.extract_attention_weights(test_data, feature_cols)
    attn_heatmap_path = os.path.join('results', 'plots', f"{args.ticker.replace('.NS', '')}_attention_heatmap.png")
    plot_attention_weights(
        mean_attention.reshape(1, -1), 
        x_labels=[f"t-{args.lookback - i}" for i in range(args.lookback)], 
        y_labels=["Attention Weight"], 
        output_path=attn_heatmap_path
    )
    
    # D. Custom Transformer
    logger.info("=" * 40)
    logger.info("TRAINING CUSTOM TIME-SERIES TRANSFORMER...")
    trans_helper = TransformerHelper(trainer)
    trans_weights_path = os.path.join('results', 'models', f"{args.ticker.replace('.NS', '')}_transformer.pth")
    trans_helper.fit(train_data, val_data, feature_cols, target_col, save_path=trans_weights_path)
    trans_raw_preds = trainer.predict(test_data, feature_cols)
    test_predictions['Transformer'] = trans_raw_preds[:, 0]
    
    # ------------------ 7. Forecasting Metrics & Backtesting ------------------
    logger.info("=" * 40)
    logger.info("EVALUATING MODEL PERFORMANCE AND BACKTESTING...")
    
    backtester = QuantitativeBacktester(initial_capital=100000.0)
    
    forecasting_report = []
    
    # Run Benchmark Buy & Hold on aligned actuals first to establish base equity curve
    _, _, benchmark_curve = backtester.backtest(
        aligned_dates, 
        aligned_actuals, 
        aligned_actuals, # dummy predictions for B&H
        strategy_type='long_only'
    )
    backtest_curves['Buy & Hold'] = benchmark_curve
    
    for model_name, predictions in test_predictions.items():
        # A. Core forecasting metrics
        metrics = calculate_forecasting_metrics(aligned_actuals, predictions, model_name=model_name)
        metrics['Model'] = model_name
        forecasting_report.append(metrics)
        
        # B. Financial Backtest (Long-Only strategy)
        stats, equity_curve, _ = backtester.backtest(
            aligned_dates, 
            aligned_actuals, 
            predictions, 
            strategy_type='long_only'
        )
        
        stats['Model'] = model_name
        backtest_stats.append(stats)
        
        # Store equity curves for plotting
        backtest_curves[f"{model_name} Strategy"] = equity_curve
        
    # ------------------ 8. Visualizations & Reports ------------------
    # Save comparison plot
    forecast_plot_path = os.path.join('results', 'plots', f"{args.ticker.replace('.NS', '')}_forecasts.png")
    plot_predictions(aligned_dates, aligned_actuals, test_predictions, args.ticker.replace('.NS', ''), output_path=forecast_plot_path)
    
    # Save backtest equity curves plot
    backtest_plot_path = os.path.join('results', 'plots', f"{args.ticker.replace('.NS', '')}_backtests.png")
    plot_backtest_results(aligned_dates, backtest_curves, args.ticker.replace('.NS', ''), output_path=backtest_plot_path)
    
    # Generate final tables and save metrics
    df_forecasting = pd.DataFrame(forecasting_report).set_index('Model')
    df_backtest = pd.DataFrame(backtest_stats).set_index('Model')
    
    df_forecasting.to_csv(os.path.join('results', 'metrics', f"{args.ticker.replace('.NS', '')}_forecasting.csv"))
    df_backtest.to_csv(os.path.join('results', 'metrics', f"{args.ticker.replace('.NS', '')}_backtest.csv"))

    # Save detailed predictions for interactive charts
    df_preds_out = pd.DataFrame(index=aligned_dates)
    df_preds_out.index.name = 'Date'
    df_preds_out['Actual'] = aligned_actuals
    for model_name, preds in test_predictions.items():
        df_preds_out[model_name] = preds
    df_preds_out.to_csv(os.path.join('results', 'metrics', f"{args.ticker.replace('.NS', '')}_predictions.csv"))
    
    # Save detailed equity curves for interactive backtest charts
    df_curves_out = pd.DataFrame(index=aligned_dates)
    df_curves_out.index.name = 'Date'
    for curve_name, curve in backtest_curves.items():
        df_curves_out[curve_name] = curve
    df_curves_out.to_csv(os.path.join('results', 'metrics', f"{args.ticker.replace('.NS', '')}_curves.csv"))
    
    # Format and display markdown summaries in logger
    logger.info("=" * 60)
    logger.info("FORECASTING ACCURACY METRICS COMPARISON")
    logger.info("=" * 60)
    try:
        logger.info("\n" + df_forecasting.to_markdown())
    except Exception:
        logger.info("\n" + str(df_forecasting))
    
    logger.info("=" * 60)
    logger.info("TRADING STRATEGY BACKTEST METRICS COMPARISON (LONG-ONLY)")
    logger.info("=" * 60)
    try:
        logger.info("\n" + df_backtest.drop(columns=['Strategy Type']).to_markdown())
    except Exception:
        logger.info("\n" + str(df_backtest.drop(columns=['Strategy Type'])))
    
    logger.info("=" * 60)
    logger.info(f"PIPELINE RUN COMPLETED! ALL ASSETS SAVED SUCCESSFULLY IN 'results/'.")
    logger.info("=" * 60)

if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)
