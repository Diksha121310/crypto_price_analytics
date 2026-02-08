"""
Model Comparison and Evaluation Script
Compares ARIMA, LSTM, and Prophet models for cryptocurrency price prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arima_model import ARIMAModel, find_optimal_order
from lstm_model import LSTMModel
from prophet_model import ProphetModel


class ModelComparison:
    """
    Compare multiple forecasting models for cryptocurrency price prediction
    """
    
    def __init__(self, data_path, coin='BTC'):
        """
        Initialize model comparison
        
        Args:
            data_path: path to processed data CSV
            coin: cryptocurrency symbol to analyze
        """
        self.data_path = data_path
        self.coin = coin
        self.data = None
        self.prices = None
        self.dates = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """
        Load and prepare cryptocurrency data
        """
        df = pd.read_csv(self.data_path)
        
        # Filter by coin
        coin_data = df[df['coin'] == self.coin].copy()
        if coin_data.empty:
            raise ValueError(f"No data found for coin '{self.coin}'. Available: {sorted(df['coin'].unique().tolist())}")
        
        coin_data['date'] = pd.to_datetime(coin_data['date'])
        coin_data = coin_data.sort_values('date')
        
        self.data = coin_data
        self.prices = coin_data['close'].values
        self.dates = coin_data['date'].values
        
        print(f"Loaded {len(self.prices)} records for {self.coin}")
        return coin_data
    
    def train_arima(self, order=None, train_size=0.8):
        """
        Train ARIMA model
        
        Args:
            order: ARIMA order tuple (p, d, q), or None to find optimal
            train_size: proportion of data for training
        """
        if self.prices is None:
            raise ValueError("Must load data first")
        
        price_series = pd.Series(self.prices)
        
        # Find optimal order if not provided
        if order is None:
            print("Finding optimal ARIMA order...")
            order, _ = find_optimal_order(price_series)
        
        # Train model
        arima = ARIMAModel(order=order)
        train_data, test_data = arima.prepare_data(price_series, train_size)
        arima.fit(train_data)
        
        # Evaluate and predict
        metrics, test_pred = arima.evaluate(test_data)
        forecast, forecast_df = arima.predict(steps=30)
        
        self.models['ARIMA'] = {
            'model': arima,
            'metrics': metrics,
            'test_pred': test_pred,
            'forecast': forecast,
            'forecast_df': forecast_df
        }
        
        print(f"\nARIMA Model Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return arima
    
    def train_lstm(self, lookback=60, units=[50, 50], epochs=50, train_size=0.8):
        """
        Train LSTM model
        
        Args:
            lookback: number of previous time steps to use
            units: list of LSTM units
            epochs: training epochs
            train_size: proportion of data for training
        """
        if self.prices is None:
            raise ValueError("Must load data first")
        
        price_series = pd.Series(self.prices)
        
        # Train model
        lstm = LSTMModel(lookback=lookback, units=units)
        lstm.prepare_data(price_series, train_size)
        lstm.build_model()
        history = lstm.fit(epochs=epochs, batch_size=32, verbose=0)
        
        # Evaluate and predict
        metrics, test_pred, y_test_actual = lstm.evaluate()
        forecast = lstm.predict(steps=30)
        
        self.models['LSTM'] = {
            'model': lstm,
            'metrics': metrics,
            'test_pred': test_pred,
            'forecast': forecast,
            'history': history
        }
        
        print(f"\nLSTM Model Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return lstm
    
    def train_prophet(self, train_size=0.8):
        """
        Train Prophet model
        
        Args:
            train_size: proportion of data for training
        """
        if self.prices is None:
            raise ValueError("Must load data first")
        
        # Train model
        prophet = ProphetModel()
        train_data, test_data = prophet.prepare_data(self.prices, self.dates, train_size)
        prophet.fit(train_data)
        
        # Evaluate and predict
        metrics, test_pred = prophet.evaluate()
        forecast = prophet.predict(periods=30)
        
        self.models['Prophet'] = {
            'model': prophet,
            'metrics': metrics,
            'test_pred': test_pred,
            'forecast': forecast
        }
        
        print(f"\nProphet Model Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return prophet
    
    def compare_models(self):
        """
        Compare all trained models
        
        Returns:
            comparison dataframe
        """
        if not self.models:
            raise ValueError("Must train models first")
        
        comparison = pd.DataFrame({
            model_name: model_info['metrics'] 
            for model_name, model_info in self.models.items()
        }).T
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(comparison.to_string())
        print("="*60)
        
        return comparison
    
    def plot_comparison(self):
        """
        Plot comparison of model predictions
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        models_list = list(self.models.keys())
        
        for idx, model_name in enumerate(models_list):
            model_info = self.models[model_name]
            metrics = model_info['metrics']
            
            ax = axes[idx]
            
            # Plot metrics as bar chart
            metrics_df = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Value': list(metrics.values())
            })
            
            ax.bar(metrics_df['Metric'], metrics_df['Value'])
            ax.set_title(f'{model_name} Model Performance', fontweight='bold')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def plot_forecasts(self):
        """
        Plot all model forecasts side by side
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Use last 100 training points for context
        last_points = 100
        x_range = np.arange(-last_points, 30)
        
        models_list = list(self.models.keys())
        
        for idx, model_name in enumerate(models_list):
            model_info = self.models[model_name]
            forecast = model_info['forecast']
            
            ax = axes[idx]
            
            # Plot recent historical data
            ax.plot(range(-last_points, 0), self.prices[-last_points:], 
                   label='Historical', color='blue')
            
            # Plot forecast
            if model_name == 'ARIMA':
                ax.plot(range(0, 30), forecast, label='Forecast', 
                       color='red', linestyle='--', marker='o')
            else:  # LSTM and Prophet
                ax.plot(range(0, 30), forecast.flatten(), label='Forecast', 
                       color='red', linestyle='--', marker='o')
            
            ax.set_title(f'{model_name} Forecast', fontweight='bold')
            ax.set_xlabel('Days')
            ax.set_ylabel('Price ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_forecast_summary(self, days=30):
        """
        Get summary of all model forecasts
        
        Returns:
            dataframe with forecast summary
        """
        summary_data = {}
        
        for model_name, model_info in self.models.items():
            forecast = model_info['forecast'].flatten()
            
            summary_data[model_name] = {
                'Min Price': forecast.min(),
                'Max Price': forecast.max(),
                'Avg Price': forecast.mean(),
                'Final Price': forecast[-1],
                'Change from Last': ((forecast[-1] - self.prices[-1]) / self.prices[-1]) * 100
            }
        
        summary_df = pd.DataFrame(summary_data).T
        
        print("\n" + "="*60)
        print(f"FORECAST SUMMARY (Next {days} Days)")
        print("="*60)
        print(summary_df.to_string())
        print("="*60)
        
        return summary_df


def get_available_coins(data_path):
    """Get list of unique coins from the data."""
    df = pd.read_csv(data_path)
    return sorted(df['coin'].unique().tolist())


def main(coins=None):
    """
    Main execution function.
    
    Args:
        coins: List of coin symbols to analyze (e.g. ['BTC', 'ETH']), 
               or None/'all' to run for all coins in the data.
    """
    data_path = '../data/processed/processed_data.csv'
    
    if coins is None or coins == 'all':
        coins = get_available_coins(data_path)
        print(f"Running for all {len(coins)} coins: {coins}\n")
    else:
        coins = [c.upper() for c in coins] if isinstance(coins, list) else [str(coins).upper()]
    
    all_results = {}
    
    for coin in coins:
        print("\n" + "="*70)
        print(f"ANALYZING {coin}")
        print("="*70)
        
        comparison = ModelComparison(data_path, coin=coin)
        
        try:
            comparison.load_data()
        except Exception as e:
            print(f"Skipping {coin}: {e}")
            continue
        
        # Train all models
        print("\n" + "-"*60)
        print("TRAINING MODELS")
        print("-"*60 + "\n")
        
        try:
            comparison.train_arima(order=(5, 1, 2))
            comparison.train_lstm(epochs=50)
            comparison.train_prophet()
        except Exception as e:
            print(f"Error training models for {coin}: {e}")
            continue
        
        # Store results
        comparison.compare_models()
        summary = comparison.get_forecast_summary()
        all_results[coin] = {
            'comparison': comparison,
            'metrics': {m: comparison.models[m]['metrics'] for m in comparison.models}
        }
        
        # Plot results for this coin
        print(f"\nGenerating plots for {coin}...")
        comparison.plot_comparison()
        comparison.plot_forecasts()
    
    # Summary across all coins
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("CROSS-COIN SUMMARY (RMSE by model)")
        print("="*70)
        for model_name in ['ARIMA', 'LSTM', 'Prophet']:
            model_metrics = {coin: data['metrics'].get(model_name, {}) for coin, data in all_results.items()}
            rmse_by_coin = {coin: m.get('RMSE', np.nan) for coin, m in model_metrics.items()}
            print(f"\n{model_name}: {rmse_by_coin}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compare forecasting models for crypto prices')
    parser.add_argument('--coin', nargs='*', default=None,
                        help='Coin(s) to analyze (e.g. BTC ETH). Omit to run for all coins.')
    args = parser.parse_args()
    main(coins=args.coin if args.coin else 'all')
