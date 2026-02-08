"""
ARIMA Model for Cryptocurrency Price Prediction
Implements AutoRegressive Integrated Moving Average model
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class ARIMAModel:
    """
    ARIMA model for time series forecasting of cryptocurrency prices
    """
    
    def __init__(self, order=(5, 1, 2), seasonal_order=(0, 0, 0, 0)):
        """
        Initialize ARIMA model
        
        Args:
            order: tuple (p, d, q) for ARIMA
            seasonal_order: tuple for SARIMA model
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.train_data = None
        self.test_data = None
        self.predictions = None
        
    def prepare_data(self, data, train_size=0.8):
        """
        Prepare and split data for training and testing
        
        Args:
            data: pandas Series of price data
            train_size: proportion of data for training
            
        Returns:
            train_data, test_data
        """
        split_idx = int(len(data) * train_size)
        self.train_data = data[:split_idx]
        self.test_data = data[split_idx:]
        
        return self.train_data, self.test_data
    
    def fit(self, train_data):
        """
        Fit ARIMA model to training data
        
        Args:
            train_data: pandas Series of training data
        """
        try:
            self.model = ARIMA(
                train_data,
                order=self.order,
                seasonal_order=self.seasonal_order
            )
            self.fitted_model = self.model.fit()
            print(f"ARIMA{self.order} model fitted successfully")
            return self.fitted_model
        except Exception as e:
            print(f"Error fitting ARIMA model: {str(e)}")
            return None
    
    def predict(self, steps=30):
        """
        Make predictions for future prices
        
        Args:
            steps: number of steps to forecast
            
        Returns:
            forecast values
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        forecast = self.fitted_model.get_forecast(steps=steps)
        forecast_df = forecast.conf_int()
        forecast_df['forecast'] = forecast.predicted_mean
        
        self.predictions = forecast_df['forecast'].values
        return self.predictions, forecast_df
    
    def evaluate(self, test_data):
        """
        Evaluate model performance on test data
        
        Args:
            test_data: pandas Series of test data
            
        Returns:
            dict with evaluation metrics
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before evaluation")
        
        # Get predictions for test period
        start_idx = len(self.train_data)
        end_idx = start_idx + len(test_data) - 1
        predictions = self.fitted_model.get_prediction(start=start_idx, end=end_idx)
        pred_values = predictions.predicted_mean.values
        
        # Calculate metrics
        mse = mean_squared_error(test_data, pred_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data, pred_values)
        r2 = r2_score(test_data, pred_values)
        mape = np.mean(np.abs((test_data.values - pred_values) / test_data.values)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        return metrics, pred_values
    
    def plot_diagnostics(self):
        """
        Plot model diagnostics
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before plotting diagnostics")
        
        self.fitted_model.plot_diagnostics(figsize=(12, 8))
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, test_data, forecast_df, title="ARIMA Price Forecast"):
        """
        Plot actual vs predicted prices
        
        Args:
            test_data: actual test data
            forecast_df: forecast dataframe from predict()
            title: plot title
        """
        plt.figure(figsize=(14, 7))
        
        # Plot actual prices
        plt.plot(range(len(self.train_data)), self.train_data, label='Training Data')
        plt.plot(range(len(self.train_data), len(self.train_data) + len(test_data)), 
                test_data, label='Actual Test Data', color='green')
        
        # Plot predictions
        forecast_range = range(len(self.train_data) + len(test_data), 
                              len(self.train_data) + len(test_data) + len(self.predictions))
        plt.plot(forecast_range, self.predictions, label='Forecast', 
                color='red', linestyle='--', marker='o')
        
        # Plot confidence interval
        if forecast_df is not None:
            plt.fill_between(forecast_range, 
                            forecast_df.iloc[:, 0], 
                            forecast_df.iloc[:, 1], 
                            alpha=0.3, color='red', label='Confidence Interval')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_summary(self):
        """
        Get model summary statistics
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before getting summary")
        
        return self.fitted_model.summary()


def find_optimal_order(data, p_range=range(0, 6), d_range=range(0, 3), q_range=range(0, 6)):
    """
    Find optimal ARIMA order using AIC criterion
    
    Args:
        data: time series data
        p_range: range of p values to try
        d_range: range of d values to try
        q_range: range of q values to try
        
    Returns:
        best_order, best_aic
    """
    best_aic = np.inf
    best_order = None
    
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except:
                    continue
    
    print(f"Best ARIMA order: {best_order} with AIC: {best_aic:.2f}")
    return best_order, best_aic
