import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from src.utils import logger

class BaselineModelWrapper:
    """
    Unified class wrapper to manage traditional statistical baseline models
    (ARIMA and Exponential Smoothing) for forecasting.
    """
    def __init__(self, model_type='arima'):
        """
        Args:
            model_type: 'arima' or 'expsmoothing'
        """
        self.model_type = model_type.lower()
        self.model = None
        self.model_fit = None

    def fit(self, train_series):
        """
        Fits the baseline model on a 1D training series (e.g. Close Price).
        """
        logger.info(f"Fitting baseline {self.model_type.upper()} model on {len(train_series)} trading days...")
        
        # Ensure we have a clean pandas Series with no NaNs
        y_train = pd.Series(train_series).dropna()
        
        if self.model_type == 'arima':
            try:
                # auto_arima will automatically search for optimal parameters (p, d, q)
                self.model = auto_arima(
                    y_train,
                    start_p=1, start_q=1,
                    max_p=3, max_q=3,
                    d=None,  # let auto_arima determine differencing
                    seasonal=False,  # Stock prices rarely have standard daily seasonality
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    trace=False
                )
                logger.info(f"ARIMA Model selected: {self.model.order}")
            except Exception as e:
                logger.error(f"Failed to fit Auto-ARIMA: {e}. Falling back to standard ARIMA(1,1,1).")
                # Fallback to a simple statsmodels ARIMA if auto_arima fails
                from statsmodels.tsa.arima.model import ARIMA as SMA_ARIMA
                self.model = SMA_ARIMA(y_train, order=(1, 1, 1))
                self.model_fit = self.model.fit()
                
        elif self.model_type == 'expsmoothing':
            try:
                # Double/Triple exponential smoothing (Holt-Winters)
                # Trend='add' (additive), damped=True to avoid linear projections blowing up
                # seasonal=None (no seasonality) since daily stock data has no reliable short-term calendar seasonality
                self.model = ExponentialSmoothing(
                    y_train,
                    trend='add',
                    damped_trend=True,
                    seasonal=None
                )
                self.model_fit = self.model.fit()
                logger.info("Exponential Smoothing model fitted successfully.")
            except Exception as e:
                logger.error(f"Failed to fit Exponential Smoothing model: {e}")
                raise e
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Select 'arima' or 'expsmoothing'.")

    def predict(self, steps):
        """
        Forecasts out-of-sample values for 'steps' periods.
        """
        if self.model is None and self.model_fit is None:
            raise ValueError("Model is not fitted yet. Call fit() before predict().")
            
        logger.info(f"Generating {steps}-day baseline forecast with {self.model_type.upper()}...")
        
        if self.model_type == 'arima':
            # Check if using fallback statsmodels fit
            if self.model_fit is not None:
                preds = self.model_fit.forecast(steps=steps)
            else:
                preds = self.model.predict(n_periods=steps)
            return np.array(preds)
            
        elif self.model_type == 'expsmoothing':
            preds = self.model_fit.forecast(steps=steps)
            return np.array(preds)
            
        return np.zeros(steps)
