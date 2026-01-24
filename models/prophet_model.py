"""
Facebook Prophet Model for Cryptocurrency Price Prediction
Implements Facebook's Prophet for time series forecasting
"""

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class ProphetModel:
    """
    Facebook Prophet model for time series forecasting of cryptocurrency prices
    """
    
    def __init__(self, yearly_seasonality=True, weekly_seasonality=True, 
                 daily_seasonality=False, changepoint_prior_scale=0.05,
                 seasonality_prior_scale=10):
        """
        Initialize Prophet model
        
        Args:
            yearly_seasonality: enable yearly seasonality
            weekly_seasonality: enable weekly seasonality
            daily_seasonality: enable daily seasonality
            changepoint_prior_scale: flexibility of trend changes
            seasonality_prior_scale: strength of seasonality component
        """
        self.model = None
        self.forecast = None
        self.train_data = None
        self.test_data = None
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        
    def prepare_data(self, data, dates, train_size=0.8):
        """
        Prepare data for Prophet (requires 'ds' and 'y' columns)
        
        Args:
            data: pandas Series or array of price data
            dates: pandas DatetimeIndex or array of dates
            train_size: proportion of data for training
            
        Returns:
            train_df, test_df (in Prophet format)
        """
        # Create dataframe in Prophet format
        df = pd.DataFrame({
            'ds': pd.to_datetime(dates),
            'y': data
        })
        
        # Sort by date
        df = df.sort_values('ds').reset_index(drop=True)
        
        split_idx = int(len(df) * train_size)
        self.train_data = df[:split_idx].copy()
        self.test_data = df[split_idx:].copy()
        
        return self.train_data, self.test_data
    
    def fit(self, train_data, interval_width=0.95):
        """
        Fit Prophet model to training data
        
        Args:
            train_data: dataframe with 'ds' and 'y' columns
            interval_width: confidence interval width
        """
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            interval_width=interval_width,
            interval_width_level=0.95
        )
        
        # Fit the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(train_data)
        
        print("Prophet model fitted successfully")
        return self.model
    
    def predict(self, periods=30):
        """
        Make predictions for future prices
        
        Args:
            periods: number of periods to forecast
            
        Returns:
            forecast dataframe
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        future = self.model.make_future_dataframe(periods=periods)
        self.forecast = self.model.predict(future)
        
        return self.forecast
    
    def evaluate(self):
        """
        Evaluate model performance on test data
        
        Returns:
            dict with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be fitted before evaluation")
        
        # Get predictions for test period
        test_forecast = self.model.predict(self.test_data[['ds']])
        
        # Calculate metrics
        y_test = self.test_data['y'].values
        y_pred = test_forecast['yhat'].values
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        return metrics, y_pred
    
    def plot_forecast(self, title="Prophet Price Forecast"):
        """
        Plot forecast with components
        
        Args:
            title: plot title
        """
        if self.forecast is None:
            raise ValueError("Must call predict() before plotting forecast")
        
        fig = self.model.plot(self.forecast, figsize=(14, 7))
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_components(self):
        """
        Plot Prophet components (trend, seasonality)
        """
        if self.forecast is None:
            raise ValueError("Must call predict() before plotting components")
        
        fig = self.model.plot_components(self.forecast, figsize=(14, 10))
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_test, y_pred, forecast_df, title="Prophet Price Forecast"):
        """
        Plot actual vs predicted prices
        
        Args:
            y_test: actual test prices
            y_pred: predicted test prices
            forecast_df: forecast dataframe from predict()
            title: plot title
        """
        plt.figure(figsize=(14, 7))
        
        # Plot training data
        plt.plot(self.train_data['ds'], self.train_data['y'], label='Training Data')
        
        # Plot test data and predictions
        plt.plot(self.test_data['ds'], y_test, label='Actual Test Data', color='green')
        plt.plot(self.test_data['ds'], y_pred, label='Test Predictions', 
                color='orange', linestyle='--')
        
        # Plot future forecast
        future_forecast = forecast_df[forecast_df['ds'] > self.test_data['ds'].max()]
        plt.plot(future_forecast['ds'], future_forecast['yhat'], label='Future Forecast',
                color='red', linestyle='--', marker='o')
        
        # Plot confidence interval
        plt.fill_between(future_forecast['ds'],
                        future_forecast['yhat_lower'],
                        future_forecast['yhat_upper'],
                        alpha=0.3, color='red', label='Confidence Interval')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_forecast_table(self, periods=30):
        """
        Get forecast as a formatted table
        
        Args:
            periods: number of periods to include
            
        Returns:
            pandas dataframe with forecast
        """
        if self.forecast is None:
            raise ValueError("Must call predict() before getting forecast table")
        
        forecast_table = self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        forecast_table.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
        
        return forecast_table

