"""
LSTM Model for Cryptocurrency Price Prediction
Implements Long Short-Term Memory neural network for time series forecasting
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class LSTMModel:
    """
    LSTM model for time series forecasting of cryptocurrency prices
    """
    
    def __init__(self, lookback=60, units=[50, 50], dropout_rate=0.2):
        """
        Initialize LSTM model
        
        Args:
            lookback: number of previous time steps to use as input
            units: list of LSTM units for each layer
            dropout_rate: dropout rate for regularization
        """
        self.lookback = lookback
        self.units = units
        self.dropout_rate = dropout_rate
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_data = None
        self.test_data = None
        self.predictions = None
        
    def prepare_data(self, data, train_size=0.8):
        """
        Prepare data for LSTM training
        
        Args:
            data: pandas Series of price data
            train_size: proportion of data for training
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Normalize the data
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        split_idx = int(len(scaled_data) * train_size)
        self.train_data = scaled_data[:split_idx]
        self.test_data = scaled_data[split_idx:]
        
        # Create sequences for training
        def create_sequences(data, lookback):
            X, y = [], []
            for i in range(len(data) - lookback):
                X.append(data[i:(i + lookback), 0])
                y.append(data[i + lookback, 0])
            return np.array(X), np.array(y)
        
        self.X_train, self.y_train = create_sequences(self.train_data, self.lookback)
        self.X_test, self.y_test = create_sequences(self.test_data, self.lookback)
        
        # Reshape for LSTM input (samples, time steps, features)
        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
        
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Test data shape: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model(self):
        """
        Build LSTM neural network model
        """
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(units=self.units[0], return_sequences=True, 
                           input_shape=(self.lookback, 1)))
        self.model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for unit in self.units[1:]:
            self.model.add(LSTM(units=unit, return_sequences=True))
            self.model.add(Dropout(self.dropout_rate))
        
        # Final LSTM layer
        self.model.add(LSTM(units=self.units[-1]))
        self.model.add(Dropout(self.dropout_rate))
        
        # Dense output layer
        self.model.add(Dense(units=1))
        
        # Compile model
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        print("LSTM model built successfully")
        return self.model
    
    def fit(self, epochs=50, batch_size=32, validation_split=0.1, verbose=1):
        """
        Train LSTM model
        
        Args:
            epochs: number of training epochs
            batch_size: training batch size
            validation_split: proportion of training data for validation
            verbose: verbosity level
            
        Returns:
            training history
        """
        if self.model is None:
            self.build_model()
        
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        
        print("LSTM model trained successfully")
        return history
    
    def predict(self, steps=30):
        """
        Make predictions for future prices
        
        Args:
            steps: number of steps to forecast
            
        Returns:
            forecast values in original scale
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Use last lookback values to start prediction
        last_sequence = self.test_data[-self.lookback:].reshape(1, self.lookback, 1)
        predictions = []
        
        for _ in range(steps):
            next_pred = self.model.predict(last_sequence, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            last_sequence = np.append(last_sequence[0, 1:], next_pred)
            last_sequence = last_sequence.reshape(1, self.lookback, 1)
        
        # Convert back to original scale
        predictions = np.array(predictions).reshape(-1, 1)
        self.predictions = self.scaler.inverse_transform(predictions)
        
        return self.predictions
    
    def evaluate(self):
        """
        Evaluate model performance on test data
        
        Returns:
            dict with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Get predictions for test set
        test_predictions = self.model.predict(self.X_test, verbose=0)
        
        # Convert back to original scale
        y_test_actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        test_predictions_actual = self.scaler.inverse_transform(test_predictions)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_actual, test_predictions_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, test_predictions_actual)
        r2 = r2_score(y_test_actual, test_predictions_actual)
        mape = np.mean(np.abs((y_test_actual - test_predictions_actual) / y_test_actual)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        return metrics, test_predictions_actual, y_test_actual
    
    def plot_training_history(self, history):
        """
        Plot training and validation loss
        
        Args:
            history: training history object from fit()
        """
        plt.figure(figsize=(12, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('LSTM Model Training History', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_test_actual, test_predictions, forecast, title="LSTM Price Forecast"):
        """
        Plot actual vs predicted prices
        
        Args:
            y_test_actual: actual test prices in original scale
            test_predictions: predicted test prices
            forecast: future forecast prices
            title: plot title
        """
        plt.figure(figsize=(14, 7))
        
        # Plot test predictions
        test_range = range(len(y_test_actual))
        plt.plot(test_range, y_test_actual, label='Actual Test Data', color='green')
        plt.plot(test_range, test_predictions, label='Test Predictions', 
                color='orange', linestyle='--')
        
        # Plot future forecast
        forecast_range = range(len(y_test_actual), len(y_test_actual) + len(forecast))
        plt.plot(forecast_range, forecast, label='Future Forecast', 
                color='red', linestyle='--', marker='o')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_model_summary(self):
        """
        Get model architecture summary
        """
        if self.model is None:
            raise ValueError("Model must be built before getting summary")
        
        return self.model.summary()

